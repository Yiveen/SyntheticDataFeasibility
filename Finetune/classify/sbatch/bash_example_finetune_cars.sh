#!/bin/bash

# === Read job index from argument ===
TASK_ID=$1
if [ -z "$TASK_ID" ]; then
  echo "Usage: bash run_finetune.sh <task_id>"
  exit 1
fi

# === Define parameter combinations ===
FEASIBLE_VALUES=(True True False False True True False False False True False True False)
INFEASIBLE_VALUES=(False False True True False False True True True False True False True)
BACK_VALUES=(True False True False True False True False False False False False False)
COLOR_VALUES=(False True False True False True False True True False False False False)
TEXTURE_VALUES=(False False False False False False False False False True True True True)
NO_MASK_VALUES=(False False False False False False False False False False False False False)

SYNTHETIC_TRAIN_FLAG=(True True True True True True True True False True True True True)
REAL_TRAIN_FLAG=(False False False False True True True True False False False True True)

# === Retrieve parameters for this task ===
FEASIBLE=${FEASIBLE_VALUES[$TASK_ID]}
INFEASIBLE=${INFEASIBLE_VALUES[$TASK_ID]}
BACK=${BACK_VALUES[$TASK_ID]}
COLOR=${COLOR_VALUES[$TASK_ID]}
TEXTURE=${TEXTURE_VALUES[$TASK_ID]}
NO_MASK=${NO_MASK_VALUES[$TASK_ID]}

SYNTHETIC_TRAIN=${SYNTHETIC_TRAIN_FLAG[$TASK_ID]}
REAL_TRAIN=${REAL_TRAIN_FLAG[$TASK_ID]}

# === Paths and hyperparameters ===
PROJECT_ROOT='/dss/dsshome1/0C/ge42lor2/projects/Test/OODData/'
CSV_DIR="${PROJECT_ROOT}/Finetune/artifacts"
YAML="${PROJECT_ROOT}/Finetune/classify/local_yaml/local_cars.yaml"
OUTPUT_DIR="output_ft/cars"

TOTAL_ITER=78720
VAL_ITER=656
DATASET="cars"
NIPC=500
N_SHOT=100
LAMBDA_1=0.5
LR=5e-4
MIN_LR=1e-8
WD=1e-4
EPOCH=25
WARMUP_EPOCH=3
IS_MIX_AUG=False
N_TEMPLATE=1

# === Conda setup ===
CONDA_ROOT='/dss/dsshome1/0C/ge42lor2/miniconda3'
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate varireal

# === GPU Info ===
echo "CUDA version from nvidia-smi:"
nvidia-smi

# === Run Python script ===
python $PROJECT_ROOT/Finetune/classify/main.py \
  --model_type=clip \
  --output_dir=$OUTPUT_DIR \
  --n_img_per_cls=$NIPC \
  --is_lora_image=True \
  --is_lora_text=True \
  --is_synth_train=$SYNTHETIC_TRAIN \
  --is_real_shots=$REAL_TRAIN \
  --lambda_1=$LAMBDA_1 \
  --warmup_epochs=$WARMUP_EPOCH \
  --log=wandb \
  --wandb_project=Finetune_test \
  --dataset=$DATASET \
  --n_shot=$N_SHOT \
  --lr=$LR \
  --wd=$WD \
  --min_lr=$MIN_LR \
  --is_mix_aug=$IS_MIX_AUG \
  --infeasible=$INFEASIBLE \
  --feasible=$FEASIBLE \
  --back=$BACK \
  --color=$COLOR \
  --texture=$TEXTURE \
  --no_mask=$NO_MASK \
  --csv_path=$CSV_DIR \
  --yaml_file=$YAML \
  --total_iterations=$TOTAL_ITER \
  --val_iter=$VAL_ITER

# === Done ===
echo "Done."