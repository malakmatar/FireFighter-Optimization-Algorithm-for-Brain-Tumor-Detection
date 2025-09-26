#!/bin/bash
#SBATCH --job-name=ffo_full
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ASCEND
#SBATCH --output=job_%j.out

# ---- in your sbatch, BEFORE srun ----
module purge
module load python
source ~/.swing_venv/bin/activate

# If you installed tensorflow[and-cuda], expose its CUDA libs (skip if using cluster CUDA modules)
sitepkg="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$sitepkg/nvidia/cudnn/lib:$sitepkg/nvidia/cublas/lib:$sitepkg/nvidia/cuda_runtime/lib:$sitepkg/nvidia/cusolver/lib:$sitepkg/nvidia/cufft/lib:$sitepkg/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

# More stable GPU memory during many trials
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true     # "true"/"false" (string), not 1/0
export TF_CPP_MIN_LOG_LEVEL=1

# Nice-to-haves
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export KERAS_HOME="$SLURM_SUBMIT_DIR/.keras"
export NO_PROXY="*"



echo "===================== NVIDIA SMI ====================="
nvidia-smi || { echo "nvidia-smi not found or GPU not visible"; exit 1 }


echo "===================== LAUNCHING FFO ====================="
srun --gpus=1 python -u src/ffo.py --mode full \
  --dense-min 128 --dense-max 512 \
  --dropout-min 0.25 --dropout-max 0.55 \
  --lr-min 5e-5 --lr-max 3e-4 \
  --batch-min 16 --batch-max 64 \
  --l2-min 1e-7 --l2-max 1e-5 \
  --epochs-fixed-full 10 \
  --seed 42
