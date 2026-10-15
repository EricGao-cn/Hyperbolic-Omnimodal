"""E1a: how many clips does a caption describe? (annotations only, no learned model)

For each clip's first caption, fan-out = number of *other* clips that have at
least one caption as similar to it as a typical paraphrase of the same clip.
Similarity is TF-IDF cosine over content words; the paraphrase threshold is the
median similarity between two captions of the same clip (needs datasets with
several captions per clip; otherwise pass --threshold).

Also reports how fan-out correlates with simple specificity proxies (caption
length, content words, IDF mass, WordNet depth when nltk+wordnet are installed),
and, with --labels, clips per category node by depth.

    python experiments/pre_method/e1a_caption_fanout.py --out OUT/e1a \
        msrvtt=/.../msrvtt/descs_ret_test.json vatex=/.../vatex/descs_ret_test.json

Writes e1a_summary.json, e1a_summary.md and e1a_fanout_<dataset>.json
({clip_id: fan-out}), which E2 and the pilot read.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from common import quantiles, spearman, write_json

STOPWORDS = set("""a an the and or but of to in on at for with by from up down out over under is are was
were be been being am it its this that these those there here he she they them his her their him i you we
our your as into onto than then so very just while who whom which what when where how all any some no not
do does did doing has have had having can could will would should may might must also about after before
s t one two""".split())


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())]


def content(tokens):
    return [w for w in tokens if w not in STOPWORDS and not w.isdigit()]


def clip_captions(annotations):
    """{clip_id: [captions]} with the first caption first, in file order."""
    grouped = defaultdict(list)
    for a in annotations:
        identifier = next(str(a[k]) for k in ("clip_id", "video_id", "image_id", "image", "id") if k in a)
        for key in ("desc", "caption", "vast_cap"):
            if key in a:
                value = a[key]
                grouped[identifier].extend(value if isinstance(value, list) else [value])
                break
    return {k: v for k, v in grouped.items() if v}


def tfidf_matrix(docs, vocab, idf):
    rows, cols, vals = [], [], []
    for r, doc in enumerate(docs):
        counts = Counter(w for w in doc if w in vocab)
        norm = math.sqrt(sum((c * idf[w]) ** 2 for w, c in counts.items())) or 1.0
        for w, c in counts.items():
            rows.append(r)
            cols.append(vocab[w])
            vals.append(c * idf[w] / norm)
    index = torch.tensor([rows, cols], dtype=torch.long)
    return torch.sparse_coo_tensor(index, torch.tensor(vals, dtype=torch.float32), (len(docs), len(vocab))).coalesce()


def wordnet_depth_fn():
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
    except Exception:
        return None

    def depth(words):
        depths = []
        for w in words:
            synsets = wn.synsets(w, pos=wn.NOUN) or wn.synsets(w, pos=wn.VERB)
            if synsets:
                depths.append(synsets[0].min_depth())
        return max(depths) if depths else float("nan")
    return depth


def analyse(name, captions_by_clip, threshold, chunk, labels):
    clips = list(captions_by_clip)
    first = [captions_by_clip[c][0] for c in clips]
    all_caps, owner = [], []
    for i, c in enumerate(clips):
        for caption in captions_by_clip[c]:
            all_caps.append(content(tokenize(caption)))
            owner.append(i)
    df = Counter(w for doc in all_caps for w in set(doc))
    vocab = {w: i for i, w in enumerate(sorted(df))}
    n_docs = len(all_caps)
    idf = {w: math.log((1 + n_docs) / (1 + d)) + 1 for w, d in df.items()}
    first_docs = [content(tokenize(c)) for c in first]
    query = tfidf_matrix(first_docs, vocab, idf).to_dense()
    owner = torch.tensor(owner)

    # same-clip paraphrase similarity: first caption vs the clip's other captions
    same = []
    best = torch.full((len(clips), len(clips)), -1.0)
    corpus = tfidf_matrix(all_caps, vocab, idf)
    for start in range(0, n_docs, chunk):
        stop = min(start + chunk, n_docs)
        rows = torch.index_select(corpus, 0, torch.arange(start, stop)).to_dense()
        sim = rows @ query.T                                  # [chunk captions, clips]
        idx = owner[start:stop]
        best.index_reduce_(0, idx, sim, "amax", include_self=True)
        own = sim[torch.arange(stop - start), idx]
        is_first = torch.ones(stop - start, dtype=torch.bool)
        for j in range(start, stop):
            is_first[j - start] = j == 0 or owner[j - 1] != owner[j]
        same.append(own[~is_first])
    same = torch.cat(same)
    source = "given"
    if threshold is None:
        source = "median same-clip paraphrase similarity"
        if same.numel() < 20:
            raise SystemExit(f"{name}: fewer than 20 same-clip caption pairs; pass --threshold")
        threshold = torch.quantile(same, 0.5).item()
    best.fill_diagonal_(-1.0)                                  # best[j, i]: clip j's best caption vs first caption of i
    fanout = (best >= threshold).sum(0)

    lengths = torch.tensor([len(tokenize(c)) for c in first], dtype=torch.float64)
    n_content = torch.tensor([len(d) for d in first_docs], dtype=torch.float64)
    idf_mass = torch.tensor([sum(idf.get(w, 0) for w in d) for d in first_docs], dtype=torch.float64)
    proxies = {"caption_length": lengths, "content_words": n_content, "idf_mass": idf_mass}
    depth = wordnet_depth_fn()
    if depth is not None:
        proxies["wordnet_max_depth"] = torch.tensor([depth(d) for d in first_docs], dtype=torch.float64)

    f = fanout.double()
    summary = {
        "clips": len(clips), "captions": n_docs, "threshold": threshold,
        "threshold_source": source,
        "same_clip_similarity": quantiles(same) if same.numel() else {},
        "fanout": quantiles(f, (0.5, 0.9, 0.99)),
        "fraction_fanout_0": (f == 0).double().mean().item(),
        "fraction_fanout_ge_1": (f >= 1).double().mean().item(),
        "fraction_fanout_ge_5": (f >= 5).double().mean().item(),
        "fraction_fanout_ge_10": (f >= 10).double().mean().item(),
        "spearman_fanout_vs": {k: spearman(f, v) for k, v in proxies.items()},
    }
    if labels:
        summary["label_tree"] = label_tree(clips, labels)
    return summary, {c: int(x) for c, x in zip(clips, fanout.tolist())}


def label_tree(clips, labels):
    """labels: {clip_id: [root, ..., leaf]}; clips and distinct nodes per depth."""
    per_depth = defaultdict(Counter)
    for c in clips:
        path = labels.get(c)
        if not path:
            continue
        for depth in range(len(path)):
            per_depth[depth][tuple(path[:depth + 1])] += 1
    return {str(d): {"nodes": len(n), "clips_per_node": quantiles(torch.tensor(list(n.values()), dtype=torch.float64))}
            for d, n in sorted(per_depth.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="fixed TF-IDF cosine threshold")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--labels", nargs="*", default=[], help="dataset=labels.json ({clip_id: [root,...,leaf]})")
    parser.add_argument("annotations", nargs="+", help="dataset=annotation.json")
    args = parser.parse_args()

    labels = {}
    for spec in args.labels:
        name, path = spec.split("=", 1)
        labels[name] = {str(k): v for k, v in json.loads(Path(path).read_text()).items()}
    out = Path(args.out)
    report = {}
    lines = ["| dataset | clips | threshold | median fan-out | 90% | ≥1 | ≥10 | ρ(fan-out, length) | ρ(fan-out, IDF mass) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for spec in args.annotations:
        name, path = spec.split("=", 1)
        captions = clip_captions(json.loads(Path(path).read_text()))
        summary, fanout = analyse(name, captions, args.threshold, args.chunk, labels.get(name))
        report[name] = summary
        write_json(out / f"e1a_fanout_{name}.json", fanout)
        rho = summary["spearman_fanout_vs"]
        lines.append(f"| {name} | {summary['clips']} | {summary['threshold']:.3f} | {summary['fanout']['q50']:.1f} | "
                     f"{summary['fanout']['q90']:.1f} | {summary['fraction_fanout_ge_1']:.2f} | "
                     f"{summary['fraction_fanout_ge_10']:.2f} | {rho['caption_length']:.2f} | {rho['idf_mass']:.2f} |")
    write_json(out / "e1a_summary.json", report)
    (out / "e1a_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
