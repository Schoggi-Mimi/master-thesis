# Fine-Grained Grounding for Differential Diagnosis in Dermatology

This repository contains the code of my MSc thesis on explanation guided learning for dermoscopy. A PanDerm foundation model is fine-tuned to separate melanoma from melanocytic nevus on HAM10000. Two kinds of alignment shape where the model's explanation maps point. Global alignment matches the maps to lesion masks. Expert alignment matches them to structure masks drawn by a dermatologist. The maps are computed with Grad-CAM, Diff-CAM and FinerCAM. They are evaluated for agreement with the lesion and with the dermatologist's structures, and for faithfulness by deleting the cells they mark.

Main finding: global alignment moves the maps onto the lesion, but the melanoma and nevus maps become nearly identical and no longer rank the cells the model relies on. Expert alignment and a blinded dermatologist study show that lesion level agreement alone does not give a differential explanation.

## Repository structure

```
master-thesis/
├── environment.yml        conda environment
├── patches/               changes to the PanDerm training code, and the upstream commit they apply to
├── splits/                image level splits of the MEL and NV subset, and the expert alignment folds
├── scripts/
│   ├── train/             SLURM launchers for training and weight sweeps
│   ├── data/              split preparation and annotation quality control
│   ├── eval/              map generation and overlap metrics
│   ├── analysis/          lesion alignment, explanation methods, expert alignment
│   ├── clinician/         pages and answer keys for the dermatologist study
│   └── figures/           thesis figures, tables and the statistics appendix
├── src/
│   ├── thesis/            analysis package used by all scripts
│   ├── cam/, utils/       map computation and panels used by the checkpoint loader
│   └── eval/              shared evaluation constants
└── tools/                 the converter that turned the analysis notebooks into scripts
```

The scripts in `scripts/data`, `eval`, `analysis`, `clinician` and `figures` were converted from Jupyter notebooks by `tools/nb_to_script.py`. Their code cells are unchanged and in their original order. Each script sets its working directory to `notebooks/mel_nv`, where the notebooks ran, so relative paths resolve as before.

## Installation

Python 3.11. Create the environment:

```bash
conda env create -f environment.yml
conda activate thesis
```

Get PanDerm, check out the commit the patch was made against and apply the patch:

```bash
mkdir -p external
git clone https://github.com/SiyuanYan1/PanDerm external/PanDerm
git -C external/PanDerm checkout $(cat patches/PANDERM_COMMIT)
git -C external/PanDerm apply ../../patches/panderm_classification.patch
```

The patch adds the training script with the alignment losses (`classification/run_class_finetuning_ha.py`) and the changes it needs in the PanDerm model and data code. The analysis package imports from it, so this step is needed for training and for evaluation.

Download the pretrained PanDerm Base weights `panderm_bb_data6_checkpoint-499.pth` as described in the PanDerm README, and place them at `external/weights/panderm_bb_data6_checkpoint-499.pth`, the path the training launcher uses.

## Data

HAM10000 is not distributed here. Download it yourself:

- Images and metadata: Harvard Dataverse, https://doi.org/10.7910/DVN/DBW86T
- Lesion segmentation masks: Kaggle, https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations

Place the files like this. The split files store paths relative to `data/HAM10000/`:

```
data/
├── HAM10000/
│   ├── images/                                  ISIC_*.jpg
│   └── mel_nv/                                  split files (see below)
└── HAM10000_segmentations_lesion_tschandl/      ISIC_*_segmentation.png
```

Copy the split files to where the scripts read them:

```bash
mkdir -p data/HAM10000/mel_nv
cp splits/*.csv data/HAM10000/mel_nv/
```

| file | content |
|---|---|
| `ham_mel_nv_clean.csv` | melanoma and nevus images with a mask, with train, validation and test split by lesion |
| `f2_fold0.csv` to `f2_fold4.csv` | training data of the five expert alignment folds |
| `f2_fold0_heldout.csv` to `f2_fold4_heldout.csv` | the annotated images held out in each fold |

Paths stored in the annotation quality control file are mapped onto the local repository by `rebase()` in `src/thesis/config.py`, so files written on another machine resolve without editing.

The structure masks drawn by the dermatologist are not public. They are available from the author on request, together with the structure masks used for expert alignment training. They are needed by `scripts/data/annotation_qc.py`, by the expert alignment training, and by every script in `scripts/eval`, `analysis`, `clinician` and `figures`. Without them, the global alignment models can be trained and evaluated for classification only.

## Reproducing the results

Training ran on UBELIX, the HPC cluster of the University of Bern, with one NVIDIA H100 GPU per job. One training run took less than an hour. The launchers are SLURM scripts. Run them from `scripts/train/`, because they call `run_panderm_full_finetune_ha.sh` by its relative name.

```bash
cd scripts/train
sbatch run_panderm_full_finetune_ha.sh      # one training run, settings via environment variables (see the script header)
bash run_ha_alpha_ablation.sh               # global alignment weight sweep
bash run_f2_sweep.sh                        # expert alignment folds
bash run_cls_lambda_sweep.sh                # CLS pooling sweep (Appendix F)
bash run_dal_mean_sweep.sh                  # class separation term with alignment (Appendix F)
bash run_dal_only_mean_sweep.sh             # class separation term without alignment (Appendix F)
cd ../..
```

The analysis scripts ran on a MacBook Pro with an M1 Pro chip and 16 GB memory. Run them from the repository root in this order:

```bash
python scripts/data/prepare_split.py              # not needed, the split files are provided
python scripts/data/annotation_qc.py
python scripts/eval/generate_eval_cams.py
python scripts/eval/overlap_eval_cams.py
python scripts/analysis/lesion_alignment.py
python scripts/analysis/explanations.py
python scripts/analysis/expert_alignment.py
python scripts/clinician/build_study.py
python scripts/figures/figures_tables.py
python scripts/figures/final_figures_statistics.py
```

Tables are written to `results/final/tables/`. Figures are written to `thesis/img/`, set by `FIG` in `src/thesis/config.py`. `final_figures_statistics.py` writes most figures of Chapters 3 and 4, the appendix tables and the statistics appendix.

## Model weights

Model weights are not distributed. They are available from the author on request.

## Licence

No licence is granted for the code in this repository. All rights are reserved by the author. You may read the code, but reuse requires the author's permission.

HAM10000 is licensed under CC BY-NC 4.0. PanDerm and the patch derived from it are subject to the PanDerm licence.