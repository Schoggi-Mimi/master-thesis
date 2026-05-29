from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper for PanDerm full finetuning with HA/DAL losses")

    parser.add_argument("--panderm-classification-dir", type=str, default="external/PanDerm/classification")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--pretrained-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model", type=str, default="PanDerm_Base_FT")
    parser.add_argument("--nb-classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--layer-decay", type=float, default=0.65)
    parser.add_argument("--drop-path", type=float, default=0.2)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--monitor", type=str, default="recall", choices=["acc", "recall"])
    parser.add_argument("--weights", action="store_true")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="Pooling mode. 'mean' uses GAP/mean pooling; 'cls' uses the CLS token.",
    )
    parser.add_argument("--image-key", type=str, default="image_rel_path")
    parser.add_argument("--mask-key", type=str, default="mask_rel_path")
    parser.add_argument("--ha-lambda", type=float, default=0.5)
    parser.add_argument(
        "--ha-start-epoch",
        type=int,
        default=0,
        help="Epoch from which HA/DAL losses are activated. Before this epoch only CE is used.",
    )
    parser.add_argument("--ha-fp-weight", type=float, default=1.0)
    parser.add_argument("--dal-lambda", type=float, default=0.0)
    parser.add_argument(
        "--dal-mode",
        type=str,
        default="all_classes",
        choices=["all_classes", "topk_non_target"],
    )
    parser.add_argument("--dal-topk", type=int, default=3)
    parser.add_argument("--init-checkpoint", type=str, default="")

    parser.add_argument("--wandb-name", type=str, default="panderm_full_finetune_ham_ha")
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", "master-thesis-panderm-ha"))
    parser.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-mode", type=str, default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--debug-batches", type=int, default=2)
    parser.add_argument("--ha-loss-type", type=str, default="paper_dice", choices=["paper_dice", "alignment"])
    parser.add_argument(
        "--disable-color-jitter",
        action="store_true",
        help="Disable color jitter in both HA paired transforms and CE-only transforms.",
    )
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.0,
        help="MixUp alpha passed to the PanDerm training script. Keep 0.0 for clean baseline/HA comparison.",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=0.0,
        help="CutMix alpha passed to the PanDerm training script. Keep 0.0 for clean baseline/HA comparison.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Label smoothing passed to the PanDerm training script. Keep 0.0 for clean baseline/HA comparison.",
    )

    return parser.parse_args()


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def main():
    args = parse_args()

    classification_dir = Path(args.panderm_classification_dir).resolve()
    script_path = classification_dir / "run_class_finetuning_ha.py"

    csv_path = Path(args.csv_path).resolve()
    root_path = Path(args.root_path).resolve()
    pretrained_ckpt = Path(args.pretrained_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    python_exe = sys.executable

    cmd = [
        python_exe,
        str(script_path),
        "--csv_path", str(csv_path),
        "--root_path", str(root_path),
        "--pretrained_checkpoint", str(pretrained_ckpt),
        "--output_dir", str(output_dir),
        "--model", args.model,
        "--nb_classes", str(args.nb_classes),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--warmup_epochs", str(args.warmup_epochs),
        "--layer_decay", str(args.layer_decay),
        "--drop_path", str(args.drop_path),
        "--update_freq", str(args.update_freq),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--monitor", args.monitor,
        "--device", args.device,
        "--pooling", args.pooling,
        "--image_key", args.image_key,
        "--mask_key", args.mask_key,
        "--mixup", str(args.mixup),
        "--cutmix", str(args.cutmix),
        "--smoothing", str(args.smoothing),
        "--ha_lambda", str(args.ha_lambda),
        "--ha_start_epoch", str(args.ha_start_epoch),
        "--ha_fp_weight", str(args.ha_fp_weight),
        "--ha_loss_type", str(args.ha_loss_type),
        "--dal_lambda", str(args.dal_lambda),
        "--dal_mode", str(args.dal_mode),
        "--dal_topk", str(args.dal_topk),
        "--disable_amp" if args.disable_amp else None,
    ]

    cmd = [x for x in cmd if x is not None]

    if args.init_checkpoint:
        cmd.extend(["--init_checkpoint", str(Path(args.init_checkpoint).resolve())])

    if args.disable_color_jitter:
        cmd.append("--disable_color_jitter")

    cmd.extend([
        "--debug_batches", str(args.debug_batches),
        "--wandb_name", args.wandb_name,
        "--wandb_project", args.wandb_project,
        "--wandb_entity", args.wandb_entity,
        "--wandb_mode", args.wandb_mode,
    ])

    if args.weights:
        cmd.append("--weights")
    if args.eval_only:
        cmd.append("--eval")

    print("=" * 80)
    print("RUNNING COMMAND")
    print("=" * 80)
    print(quote_cmd(cmd))
    print(f"[working dir] {classification_dir}")

    subprocess.run(cmd, cwd=str(classification_dir), env=env, check=True)


if __name__ == "__main__":
    main()