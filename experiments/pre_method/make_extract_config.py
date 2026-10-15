"""Write a run.py-style config for extract_features.py / e1b.

model_cfg and run_cfg come from --base: the config the checkpoint was trained
or evaluated with, so the model is rebuilt exactly as trained. data_cfg.val is
replaced by the video test sets of --tests (audio-only sets are skipped, since
every dump needs t, v and a), plus the VAST holdout and, with --with-train,
the VAST training split from holdout_split.py. Train and holdout entries are
both named finetune_area, which AnnoIndexedDataset's media filter requires;
`dump_name` keeps their dumps apart.

    python experiments/pre_method/make_extract_config.py --base TRIAL_EVAL.json \
        --holdout-dir $S/holdout --with-train --out OUT/configs/k_e0.json
"""
import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="config the checkpoint was trained/evaluated with")
    parser.add_argument("--tests", default="config/pmrl/minimal_hyperbolic_spectral_eval.json",
                        help="config whose data_cfg.val lists the test sets")
    parser.add_argument("--holdout-dir", required=True, help="output directory of holdout_split.py")
    parser.add_argument("--with-train", action="store_true", help="also dump the VAST training split")
    parser.add_argument("--train-vision-sample-num", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text())
    tests = json.loads(Path(args.tests).read_text())["data_cfg"]["val"]
    manifest = json.loads((Path(args.holdout_dir) / "holdout_manifest.json").read_text())
    config = {"run_cfg": copy.deepcopy(base["run_cfg"]), "model_cfg": copy.deepcopy(base["model_cfg"])}
    config["run_cfg"]["mode"] = "testing"
    config["model_cfg"]["checkpointing"] = False

    val = [dict(t, dump_name=t["name"]) for t in tests if "vision" in t and "audio" in t]
    common = dict(type="annoindexed", training=False, name="finetune_area", vision=manifest["vision"],
                  audio=manifest["audio"], vision_format="video_rawvideo", audio_sample_num=1,
                  task="ret%tva", n_workers=8, batch_size=args.batch_size, drop_subtitles=True)
    val.append(dict(common, dump_name="vast_holdout", txt=manifest["holdout_txt"],
                    vision_sample_num=manifest.get("vision_sample_num", 8)))
    if args.with_train:
        val.append(dict(common, dump_name="vast_train", txt=manifest["train_txt"],
                        vision_sample_num=args.train_vision_sample_num))
    config["data_cfg"] = {"train": [], "val": val}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {out}: " + ", ".join(v["dump_name"] for v in val))


if __name__ == "__main__":
    main()
