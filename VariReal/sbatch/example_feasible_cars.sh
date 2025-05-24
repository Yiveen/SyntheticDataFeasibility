#!/bin/bash

# === Path Configuration, should be changed according to yours ===
PROJECT_ROOT='your/path/to/.../OODData'
STORAGE_ROOT='your/path/to/.../VariReal_Gen'

ROOT_DIR="${STORAGE_ROOT}/data"
OUT_DIR="${STORAGE_ROOT}/output"
REAL_TRAIN_DIR="${STORAGE_ROOT}/real_train"
CSV_DIR="${PROJECT_ROOT}/Finetune/artifacts"
YAML_DIR="${PROJECT_ROOT}/VariReal/src/local.yaml"

# === Generation Parameters ===
PER_REAL=5                      # Number of generated images per real image
BS=1                            # Batch size
DATASET="cars"                  # Dataset name
GENERAL_CLS="cars"              # General class label
CLS="Ferrari FF Coupe 2012"     # Target class for generation

# === Conda Environment Setup ===
CONDA_ROOT='your/path/to/your/conda/.../miniconda3'

# Check if conda exists
if [ ! -d "$CONDA_ROOT" ]; then
  echo "ERROR: Conda root not found at $CONDA_ROOT"
  exit 1
fi

# Initialize conda and activate environment
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate varireal || {
  echo "ERROR: Failed to activate conda environment 'Finetune'"
  exit 1
}

# === Check CUDA status ===
echo "CUDA version from nvidia-smi:"
nvidia-smi || {
  echo "ERROR: Failed to run nvidia-smi"
  exit 1
}

# === Run Main VariReal Pipeline ===
echo "Running VariReal pipeline..."
python "${PROJECT_ROOT}/VariReal/src/main_pipeline.py" \
  --datasets_root "$ROOT_DIR" \
  --real_train_dir "$REAL_TRAIN_DIR" \
  --output_dir "$OUT_DIR" \
  --csv_path "$CSV_DIR" \
  --feasiblity \
  --texture \
  --images_per_real "$PER_REAL" \
  --batch_size "$BS" \
  --dataset "$DATASET" \
  --gen_class "$CLS" \
  --yaml_file "$YAML_DIR" \
  --use_vlm_filtering 



# Like for infeasible texture setting: 

# python "${PROJECT_ROOT}/VariReal/src/main_pipeline.py" \
#   --datasets_root "$ROOT_DIR" \
#   --real_train_dir "$REAL_TRAIN_DIR" \
#   --output_dir "$OUT_DIR" \
#   --csv_path "$CSV_DIR" \
#   --texture \    # change this for other attributes
#   --images_per_real "$PER_REAL" \
#   --batch_size "$BS" \
#   --dataset "$DATASET" \
#   --gen_class "$CLS" \
#   --yaml_file "$YAML_DIR"

# === Done ===
echo "Done."
