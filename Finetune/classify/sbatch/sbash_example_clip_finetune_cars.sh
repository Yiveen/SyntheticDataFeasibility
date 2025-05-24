#!/bin/bash
#SBATCH --output=your/path/to/.../run_output_finetune/cars_1e_4_job_%A_%a.out
#SBATCH --error=your/path/to/.../run_output_finetune/cars_1e_4_job_%A_%a.err
#SBATCH --time=8:00:00 # maybe change
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --job-name=1e4_cars_experiments
#SBATCH --array=10
#SBATCH --mem=30G
# print info about current job
scontrol show job $SLURM_JOB_ID


# define combinations
FEASIBLE_VALUES=(True True False False True True False False False True False True False)
INFEASIBLE_VALUES=(False False True True False False True True True False True False True)
BACK_VALUES=(True False True False True False True False False False False False False)
COLOR_VALUES=(False True False True False True False True True False False False False)
TEXTURE_VALUES=(False False False False False False False False False True True True True)
NO_MASK_VALUES=(False False False False False False False False False False False False False)

SYNTHETIC_TRAIN_FLAG=(True True True True True True True True False True True True True)
REAL_TRAIN_FLAG=(False False False False True True True True False False False True True)

FEASIBLE=${FEASIBLE_VALUES[$SLURM_ARRAY_TASK_ID]}
INFEASIBLE=${INFEASIBLE_VALUES[$SLURM_ARRAY_TASK_ID]}
BACK=${BACK_VALUES[$SLURM_ARRAY_TASK_ID]}
COLOR=${COLOR_VALUES[$SLURM_ARRAY_TASK_ID]}
TEXTURE=${TEXTURE_VALUES[$SLURM_ARRAY_TASK_ID]}
NO_MASK=${NO_MASK_VALUES[$SLURM_ARRAY_TASK_ID]}

SYNTHETIC_TRAIN=${SYNTHETIC_TRAIN_FLAG[$SLURM_ARRAY_TASK_ID]}
REAL_TRAIN=${REAL_TRAIN_FLAG[$SLURM_ARRAY_TASK_ID]}

PROJECT_ROOT='/dss/dsshome1/0C/ge42lor2/projects/OODData'
CSV_DIR="${PROJECT_ROOT}/Finetune/artifacts"

YAML="${PROJECT_ROOT}/Finetune/classify/local_yaml/local_cars.yaml"

OUTPUT_DIR="output"
TOTAL_ITER=78720
VAL_ITER=656
DATASET="cars"
NIPC=500
N_SHOT=100
LAMBDA_1=0.5
LR=1e-4
MIN_LR=1e-8
WD=1e-4
EPOCH=25
WARMUP_EPOCH=3
IS_MIX_AUG=False
N_TEMPLATE=1

echo "Running with:"
echo "FEASIBLE=$FEASIBLE, INFEASIBLE=$INFEASIBLE, BACK=$BACK, COLOR=$COLOR, NO_MASK=$NO_MASK"

CONDA_ROOT='/dss/dsshome1/0C/ge42lor2/miniconda3' # change
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate varireal

echo "CUDA version from nvidia-smi:"
nvidia-smi

srun python $PROJECT_ROOT/Finetune/classify/main.py \
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
--wandb_project=Finetune_0325 \
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
