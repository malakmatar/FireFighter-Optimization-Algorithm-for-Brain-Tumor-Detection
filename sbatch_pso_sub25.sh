#!/bin/bash
#SBATCH --job-name=pso_sub25
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ASCEND
#SBATCH --output=job_pso_sub25_%j.out

module purge
module load python
source ~/.swing_venv/bin/activate

sitepkg="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$sitepkg/nvidia/cudnn/lib:$sitepkg/nvidia/cublas/lib:$sitepkg/nvidia/cuda_runtime/lib:$sitepkg/nvidia/cusolver/lib:$sitepkg/nvidia/cufft/lib:$sitepkg/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export KERAS_HOME="$HOME/FireFighter-Optimization-Algorithm-for-Brain-Tumor-Detection/.keras"
export NO_PROXY="*"

echo "===================== NVIDIA SMI ====================="
nvidia-smi || { echo "nvidia-smi not found or GPU not visible"; exit 1; }

echo "===================== PSO FINAL (25% data, particles=10, iters=10) ====================="
srun --ntasks=1 --cpu-bind=cores --gpu-bind=map_gpu:0 python -u src/pso.py --mode full \
  --data-dir . --results-dir runs/pso_final_sub25 \
  --subset-frac 0.25 --particles 10 --iters 10 --cache "" \
  --dense-min 128 --dense-max 512 \
  --dropout-min 0.25 --dropout-max 0.55 \
  --lr-min 1e-5 --lr-max 5e-4 \
  --batch-min 16 --batch-max 32 \
  --l2-min 1e-6 --l2-max 1e-4 \
  --epochs-fixed-full 8 \
  --seed 42
