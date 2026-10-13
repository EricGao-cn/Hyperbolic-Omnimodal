"""Split a fixed validation holdout off the VAST training annotations.

Every trial in the next-steps suite trains on train.json and is selected on
holdout.json, so no configuration is chosen by looking at test sets. Only clips
whose video and audio both exist are eligible (the same filter AnnoIndexedDataset
applies to `finetune_area`), so the holdout size is exact.

    python experiments/next_steps/holdout_split.py \
        --annotations .../vast150k/VASTannotations150k.accepted.json \
        --vision .../vast150k/videos --audio .../vast150k/audios --out DIR
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--vision', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--vision-sample-num', type=int, default=8)
    args = parser.parse_args()

    annotations = json.loads(Path(args.annotations).read_text())
    vision, audio = Path(args.vision), Path(args.audio)
    usable = [i for i, a in enumerate(annotations)
              if (audio / f"{a['clip_id']}.wav").exists() and (vision / f"{a['clip_id']}.mp4").exists()]
    clips = sorted({annotations[i]['clip_id'] for i in usable})
    if len(clips) <= args.size:
        raise SystemExit(f'only {len(clips)} usable clips')
    held = set(random.Random(args.seed).sample(clips, args.size))
    train = [a for a in annotations if a['clip_id'] not in held]
    holdout = [a for a in annotations if a['clip_id'] in held]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('train.json', train), ('holdout.json', holdout)):
        (out / name).write_text(json.dumps(rows))
    manifest = dict(source=str(Path(args.annotations).resolve()),
                    source_sha256=hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
                    seed=args.seed, holdout_clips=args.size, usable_clips=len(clips),
                    train_annotations=len(train), holdout_annotations=len(holdout),
                    train_txt=str((out / 'train.json').resolve()), holdout_txt=str((out / 'holdout.json').resolve()),
                    vision=str(vision.resolve()), audio=str(audio.resolve()),
                    vision_sample_num=args.vision_sample_num)
    (out / 'holdout_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
