

"""
Build a compact qualitative CAM comparison PDF from already generated panel PNGs.

Typical workflow:
1) Run scripts.generate_finer_cam_panderm once per experiment/checkpoint.
2) Point this script to the output folders.
3) This script creates one PDF page per image, with one row per experiment.

Example:
python -m scripts.make_qualitative_comparison_pdf \
  --csv data/HAM10000/ham_test_cam_qualitative_stratified_10.csv \
  --image_col image_rel_path \
  --gt_col gt_label \
  --out_pdf outputs/qual/qualitative_comparison_gt_topk3.pdf \
  --experiments_json_path configs/qualitative_gt_topk3_baseline_ha_seggate.json \
  --num_samples 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_EXPERIMENTS = [
    {
        "name": "Baseline",
        "folder": "outputs/qual/cam_baseline_gt_topk3",
    },
    {
        "name": "HA",
        "folder": "outputs/qual/cam_ha_gt_topk3",
    },
    {
        "name": "SegGate",
        "folder": "outputs/qual/cam_seggate_gt_topk3",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a qualitative comparison PDF from generated CAM panel PNGs."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV used for qualitative CAM generation. Used only for image order and metadata.",
    )
    parser.add_argument(
        "--image_col",
        type=str,
        default="image_rel_path",
        help="Column containing the image id/path used by generate_finer_cam_panderm.py.",
    )
    parser.add_argument(
        "--gt_col",
        type=str,
        default="gt_label",
        help="Ground-truth label column for page titles.",
    )
    parser.add_argument(
        "--out_pdf",
        type=str,
        required=True,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--experiments_json",
        type=str,
        default=None,
        help=(
            "JSON list of experiments. Each item needs {'name': ..., 'folder': ...}. "
            "Usually prefer --experiments_json_path for reproducible runs. "
            "If both experiment arguments are omitted, DEFAULT_EXPERIMENTS inside this script are used."
        ),
    )
    parser.add_argument(
        "--experiments_json_path",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing the experiment list. "
            "Each item needs {'name': ..., 'folder': ...}. "
            "Use this instead of --experiments_json for reproducible runs."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional maximum number of CSV rows/images to include.",
    )
    parser.add_argument(
        "--panel_glob",
        type=str,
        default="*.png",
        help="Glob pattern for panel PNGs inside each experiment folder. Default: *.png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PDF rendering DPI. Default: 180.",
    )
    parser.add_argument(
        "--row_height",
        type=float,
        default=2.25,
        help="Height in inches per experiment row. Increase if text is too small.",
    )
    parser.add_argument(
        "--page_width",
        type=float,
        default=14.0,
        help="Page width in inches. Default: 14.",
    )
    parser.add_argument(
        "--missing_policy",
        type=str,
        default="placeholder",
        choices=["placeholder", "skip_image", "error"],
        help=(
            "What to do if a panel is missing for one experiment. "
            "placeholder = show empty row; skip_image = skip whole page; error = stop."
        ),
    )
    return parser.parse_args()


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def make_safe_output_stem(image_id: str) -> str:
    image_path = Path(str(image_id))
    stem = image_path.stem if image_path.suffix else image_path.name
    stem = stem.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return stem


def load_experiments(
    experiments_json: str | None,
    experiments_json_path: str | None = None,
) -> list[dict[str, str]]:
    if experiments_json is not None and experiments_json_path is not None:
        raise ValueError("Use either --experiments_json or --experiments_json_path, not both.")

    if experiments_json_path is not None:
        path = resolve_path(experiments_json_path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment JSON config not found: {path}")
        experiments = json.loads(path.read_text())
    elif experiments_json is not None:
        experiments = json.loads(experiments_json)
    else:
        experiments = DEFAULT_EXPERIMENTS

    if not isinstance(experiments, list) or len(experiments) == 0:
        raise ValueError("Experiment config must be a non-empty JSON list.")

    cleaned = []
    for exp in experiments:
        if "name" not in exp or "folder" not in exp:
            raise ValueError(f"Each experiment needs 'name' and 'folder'. Got: {exp}")
        cleaned.append({"name": str(exp["name"]), "folder": str(exp["folder"])})

    return cleaned


def find_panel_path(folder: Path, output_stem: str, panel_glob: str) -> Path | None:
    """
    Finds the generated panel for one image.

    generate_finer_cam_panderm.py saves files like:
    ISIC_0027776_rgb_gt_mask_gradcam_a_gradcam_b_map_diff_finercam.png

    This function matches by output_stem prefix so it also works for SegGate panels
    with different suffixes.
    """
    candidates = sorted(folder.glob(panel_glob))
    stem_matches = [p for p in candidates if p.name.startswith(f"{output_stem}_")]

    if len(stem_matches) == 0:
        # Fallback for older names that may only contain the stem somewhere.
        stem_matches = [p for p in candidates if output_stem in p.stem]

    if len(stem_matches) == 0:
        return None

    if len(stem_matches) > 1:
        # Prefer newest in case several panels exist for different panel_items.
        return max(stem_matches, key=lambda p: p.stat().st_mtime)

    return stem_matches[0]


def add_placeholder_axis(ax, experiment_name: str, message: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"{experiment_name}\n{message}",
        ha="center",
        va="center",
        fontsize=11,
    )


def add_panel_axis(ax, panel_path: Path, experiment_name: str) -> None:
    img = Image.open(panel_path).convert("RGB")
    ax.imshow(img)
    ax.axis("off")
    ax.text(
        0.01,
        0.98,
        experiment_name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
    )


def make_pdf(args: argparse.Namespace) -> None:
    csv_path = resolve_path(args.csv)
    out_pdf = resolve_path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(
        experiments_json=args.experiments_json,
        experiments_json_path=args.experiments_json_path,
    )
    for exp in experiments:
        exp["folder_resolved"] = resolve_path(exp["folder"])

    df = pd.read_csv(csv_path)
    if args.image_col not in df.columns:
        raise ValueError(
            f"CSV is missing image_col '{args.image_col}'. Found columns: {df.columns.tolist()}"
        )

    if args.num_samples is not None:
        df = df.head(args.num_samples)

    from matplotlib.backends.backend_pdf import PdfPages

    pages_written = 0
    missing_records: list[dict[str, str]] = []

    with PdfPages(out_pdf) as pdf:
        for row_idx, row in df.iterrows():
            image_id = str(row[args.image_col])
            output_stem = make_safe_output_stem(image_id)
            gt = str(row[args.gt_col]) if args.gt_col in row and pd.notna(row[args.gt_col]) else "unknown"

            panel_paths = []
            missing_for_image = []

            for exp in experiments:
                panel_path = find_panel_path(
                    folder=Path(exp["folder_resolved"]),
                    output_stem=output_stem,
                    panel_glob=args.panel_glob,
                )
                panel_paths.append(panel_path)
                if panel_path is None:
                    missing_for_image.append(exp["name"])
                    missing_records.append(
                        {
                            "image_id": image_id,
                            "output_stem": output_stem,
                            "experiment": exp["name"],
                            "folder": str(exp["folder_resolved"]),
                        }
                    )

            if missing_for_image:
                msg = f"Missing panels for {output_stem}: {missing_for_image}"
                if args.missing_policy == "error":
                    raise FileNotFoundError(msg)
                if args.missing_policy == "skip_image":
                    print(f"[skip] {msg}")
                    continue
                print(f"[warn] {msg}")

            n_rows = len(experiments)
            fig_height = max(3.0, args.row_height * n_rows + 0.7)
            fig, axes = plt.subplots(
                n_rows,
                1,
                figsize=(args.page_width, fig_height),
                squeeze=False,
            )

            fig.suptitle(
                f"{output_stem} | GT={gt} | row={row_idx}",
                fontsize=13,
                fontweight="bold",
                y=0.995,
            )

            for ax, exp, panel_path in zip(axes[:, 0], experiments, panel_paths):
                if panel_path is None:
                    add_placeholder_axis(ax, exp["name"], "missing panel")
                else:
                    add_panel_axis(ax, panel_path, exp["name"])

            plt.tight_layout(rect=(0, 0, 1, 0.975))
            pdf.savefig(fig, dpi=args.dpi)
            plt.close(fig)
            pages_written += 1

    print(f"Saved PDF: {out_pdf}")
    print(f"Pages written: {pages_written}")

    if missing_records:
        missing_csv = out_pdf.with_suffix(".missing_panels.csv")
        pd.DataFrame(missing_records).to_csv(missing_csv, index=False)
        print(f"Missing panel report: {missing_csv}")


def main() -> None:
    args = parse_args()
    make_pdf(args)


if __name__ == "__main__":
    main()