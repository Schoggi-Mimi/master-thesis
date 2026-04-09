"""
Script to run the official PanDerm full fine-tuning baseline with a convenient interface.

python -m scripts.run_panderm_full_finetune_official \
  --csv-path data/HAM10000/HAM10000.csv \
  --root-path data/HAM10000/images \
  --pretrained-checkpoint external/weights/panderm_bb_data6_checkpoint-499.pth \
  --output-dir outputs/panderm_full_finetune/ham \
  --model PanDerm_Base_FT \
  --nb-classes 7 \
  --weights \
  --monitor recall \
  --device cuda


python -m scripts.run_panderm_full_finetune_official \
  --csv-path data/BCN20000/bcn20000.csv \
  --root-path data/BCN20000/images \
  --pretrained-checkpoint external/weights/panderm_bb_data6_checkpoint-499.pth \
  --output-dir outputs/panderm_full_finetune/bcn \
  --model PanDerm_Base_FT \
  --nb-classes 9 \
  --weights \
  --monitor recall \
  --device cuda

"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper for official PanDerm full fine-tuning baseline."
    )

    parser.add_argument(
        "--panderm-classification-dir",
        type=str,
        default="external/PanDerm/classification",
        help="Path to external/PanDerm/classification",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="CSV with columns image,label,split",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        required=True,
        help="Root folder containing the images referenced by the CSV",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to PanDerm pretrained checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to save checkpoints and outputs",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="PanDerm_Base_FT",
        choices=["PanDerm_Base_FT", "PanDerm_Large_FT"],
        help="Official PanDerm finetuning model name",
    )
    parser.add_argument(
        "--nb-classes",
        type=int,
        required=True,
        help="Number of classes in the dataset",
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--layer-decay", type=float, default=0.65)
    parser.add_argument("--drop-path", type=float, default=0.2)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--monitor",
        type=str,
        default="recall",
        choices=["acc", "recall"],
        help=(
            "Official script supports only acc or recall for checkpoint selection. "
            "For imbalanced dermatology data, recall is usually the better choice."
        ),
    )

    parser.add_argument(
        "--weights",
        action="store_true",
        help="Enable weighted random sampler",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable TTA during final evaluation",
    )

    parser.add_argument(
        "--wandb-name",
        type=str,
        default="panderm_full_finetune",
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="WandB mode",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device passed to the official script, usually cuda on cluster",
    )

    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run training only, skip final eval call",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run only evaluation from checkpoint-best.pth",
    )

    return parser.parse_args()


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("=" * 80)
    print("RUNNING COMMAND")
    print("=" * 80)
    print(quote_cmd(cmd))
    print(f"\n[working dir] {cwd}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def build_train_cmd(
    python_exe: str,
    official_script: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        python_exe,
        str(official_script),
        "--model",
        args.model,
        "--pretrained_checkpoint",
        args.pretrained_checkpoint,
        "--nb_classes",
        str(args.nb_classes),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--update_freq",
        str(args.update_freq),
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--epochs",
        str(args.epochs),
        "--layer_decay",
        str(args.layer_decay),
        "--drop_path",
        str(args.drop_path),
        "--weight_decay",
        str(args.weight_decay),
        "--mixup",
        "0.8",
        "--cutmix",
        "1.0",
        "--sin_pos_emb",
        "--no_auto_resume",
        "--monitor",
        args.monitor,
        "--imagenet_default_mean_and_std",
        "--output_dir",
        args.output_dir,
        "--csv_path",
        args.csv_path,
        "--root_path",
        args.root_path,
        "--wandb_name",
        args.wandb_name,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
    ]

    if args.weights:
        cmd.append("--weights")

    return cmd


def build_eval_cmd(
    python_exe: str,
    official_script: Path,
    args: argparse.Namespace,
    best_ckpt: Path,
) -> list[str]:
    cmd = [
        python_exe,
        str(official_script),
        "--eval",
        "--model",
        args.model,
        "--pretrained_checkpoint",
        args.pretrained_checkpoint,
        "--resume",
        str(best_ckpt),
        "--nb_classes",
        str(args.nb_classes),
        "--batch_size",
        str(args.batch_size),
        "--output_dir",
        args.output_dir,
        "--csv_path",
        args.csv_path,
        "--root_path",
        args.root_path,
        "--wandb_name",
        f"{args.wandb_name}_eval",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
    ]

    if args.tta:
        cmd.append("--TTA")

    return cmd


def main() -> None:
    args = parse_args()

    if args.train_only and args.eval_only:
        raise ValueError("Use only one of --train-only or --eval-only.")

    classification_dir = Path(args.panderm_classification_dir).resolve()
    official_script = classification_dir / "run_class_finetuning.py"

    if not classification_dir.exists():
        raise FileNotFoundError(f"PanDerm classification dir not found: {classification_dir}")
    if not official_script.exists():
        raise FileNotFoundError(f"Official script not found: {official_script}")

    csv_path = Path(args.csv_path).resolve()
    root_path = Path(args.root_path).resolve()
    pretrained_ckpt = Path(args.pretrained_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not root_path.exists():
        raise FileNotFoundError(f"Root path not found: {root_path}")
    if not pretrained_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_ckpt}")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "checkpoint-best.pth"

    args.csv_path = str(csv_path)
    args.root_path = str(root_path)
    args.pretrained_checkpoint = str(pretrained_ckpt)
    args.output_dir = str(output_dir)

    env = os.environ.copy()
    env["WANDB_MODE"] = args.wandb_mode

    python_exe = sys.executable

    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    print(f"panderm_classification_dir={classification_dir}")
    print(f"official_script={official_script}")
    print(f"csv_path={csv_path}")
    print(f"root_path={root_path}")
    print(f"pretrained_checkpoint={pretrained_ckpt}")
    print(f"output_dir={output_dir}")
    print(f"model={args.model}")
    print(f"nb_classes={args.nb_classes}")
    print(f"batch_size={args.batch_size}")
    print(f"epochs={args.epochs}")
    print(f"lr={args.lr}")
    print(f"weight_decay={args.weight_decay}")
    print(f"warmup_epochs={args.warmup_epochs}")
    print(f"layer_decay={args.layer_decay}")
    print(f"drop_path={args.drop_path}")
    print(f"monitor={args.monitor}")
    print(f"weights={args.weights}")
    print(f"tta={args.tta}")
    print(f"device={args.device}")
    print(f"wandb_mode={args.wandb_mode}")

    if not args.eval_only:
        train_cmd = build_train_cmd(python_exe, official_script, args)
        run_command(train_cmd, cwd=classification_dir, env=env)

    if args.train_only:
        print("\nTraining finished. Skipping eval because --train-only was set.")
        return

    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"Expected best checkpoint not found after training: {best_ckpt}"
        )

    eval_cmd = build_eval_cmd(python_exe, official_script, args, best_ckpt)
    run_command(eval_cmd, cwd=classification_dir, env=env)

    print("\nDone.")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()

# python run_panderm_full_finetune_official.py \
#   --panderm-classification-dir ../external/PanDerm/classification \
#   --csv-path ../data/HAM10000/HAM10000.csv \
#   --root-path ../data/HAM10000/images/ \
#   --pretrained-checkpoint ../external/weights/panderm_bb_data6_checkpoint-499.pth \
#   --output-dir ../outputs/panderm_full_finetune/ham \
#   --model PanDerm_Base_FT \
#   --nb-classes 7 \
#   --batch-size 128 \
#   --epochs 50 \
#   --lr 5e-4 \
#   --weight-decay 0.05 \
#   --warmup-epochs 10 \
#   --layer-decay 0.65 \
#   --drop-path 0.2 \
#   --update-freq 1 \
#   --weights \
#   --monitor recall \
#   --wandb-name panderm_full_finetune_ham \
#   --wandb-mode disabled \
#   --device cuda



# python run_panderm_full_finetune_official.py \
#   --panderm-classification-dir ../external/PanDerm/classification \
#   --csv-path ../data/BCN20000/bcn20000.csv \
#   --root-path ../data/BCN20000/images/ \
#   --pretrained-checkpoint ../external/weights/panderm_bb_data6_checkpoint-499.pth \
#   --output-dir ../outputs/panderm_full_finetune/bcn \
#   --model PanDerm_Base_FT \
#   --nb-classes 9 \
#   --batch-size 128 \
#   --epochs 50 \
#   --lr 5e-4 \
#   --weight-decay 0.05 \
#   --warmup-epochs 10 \
#   --layer-decay 0.65 \
#   --drop-path 0.2 \
#   --update-freq 1 \
#   --weights \
#   --monitor recall \
#   --wandb-name panderm_full_finetune_bcn \
#   --wandb-mode disabled \
#   --device cuda