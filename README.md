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
git clone https://github.com/SiyuanYan1/PanDerm external/PanDerm      # [CHECK] URL
git -C external/PanDerm checkout $(cat patches/PANDERM_COMMIT)
git -C external/PanDerm apply ../../patches/panderm_classification.patch
```

The patch adds the training script with the alignment losses (`classification/run_class_finetuning_ha.py`) and the changes it needs in the PanDerm model and data code. The analysis package imports from it, so this step is needed for training and for evaluation.

Download the pretrained PanDerm Base weights from the PanDerm repository and place them at `external/weights/panderm_bb_data6_checkpoint-499.pth`, the path the training launcher uses. [CHECK] file name and download link in the PanDerm README.

## Data

HAM10000 is not distributed here. Download it yourself:

- Images and metadata: Harvard Dataverse, https://doi.org/10.7910/DVN/DBW86T
- Lesion segmentation masks (Tschandl): [CHECK] download link

Place the data under `data/HAM10000/`. The split files store image and mask paths relative to this folder. [CHECK] Name the image and mask subfolders as in the `image_rel_path` and `mask_rel_path` columns of `splits/ham_mel_nv_clean.csv`.

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

The structure masks drawn by the dermatologist are not public. They are available from the author on request. Scripts that read them need these masks. [CHECK] These are `scripts/data/annotation_qc.py`, the expert alignment training and most of the analysis scripts.

## Reproducing the results

Training ran on UBELIX, the HPC cluster of the University of Bern, with one NVIDIA H100 GPU per job. The launchers are SLURM scripts. Run them from `scripts/train/`, because they call `run_panderm_full_finetune_ha.sh` by its relative name.

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

Each training job has a wall time limit of 6 hours. [CHECK] actual run times.

Evaluation and figures, run from the repository root in this order:

```bash
python scripts/data/prepare_split.py              # not needed, the split files are provided
python scripts/data/annotation_qc.py              # needs the annotation masks
python scripts/eval/generate_eval_cams.py
python scripts/eval/overlap_eval_cams.py
python scripts/analysis/lesion_alignment.py
python scripts/analysis/explanations.py
python scripts/analysis/expert_alignment.py
python scripts/clinician/build_study.py
python scripts/figures/figures_tables.py
python scripts/figures/final_figures_statistics.py
```

Tables and figures are written to the output folders set in `src/thesis/config.py`. [CHECK] folder names. `final_figures_statistics.py` writes most figures of Chapters 3 and 4, the appendix tables and the statistics appendix. [CHECK] hardware used for the analysis scripts.

## Model weights

Model weights are not distributed. They are available from the author on request.

## Licence

No licence is granted for the code in this repository. All rights are reserved by the author. You may read the code, but reuse requires the author's permission.

HAM10000 is licensed under CC BY-NC 4.0. PanDerm and the patch derived from it are subject to the PanDerm licence.