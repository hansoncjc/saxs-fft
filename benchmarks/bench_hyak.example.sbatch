#!/bin/bash
#SBATCH --job-name=saxsfft-bench
#SBATCH --account=zeelab              # <- confirm with `hyakalloc`
#SBATCH --partition=gpu-a40           # <- confirm with `hyakalloc` / `sinfo -s`
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16            # the CPU side of the comparison
#SBATCH --gpus=1                      # the GPU side
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=bench_%j.out

# Both halves of the comparison run in ONE allocation on ONE node, so the CPU
# and GPU numbers share the same hardware, the same data and the same software
# stack.  Node-to-node variation is the largest confounder in a two-job
# benchmark; this removes it.

set -euo pipefail

module load cuda/11.8 || true          # adjust to whatever `module avail cuda` lists
source /gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate

cd "$SLURM_SUBMIT_DIR"
CSV=benchmark_results_${SLURM_JOB_ID}.csv

echo "== node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.get_device_name(0))"

# Grid sizes: the FFT cost scales as n^3 log n, so sweeping N_grid shows where
# the GPU starts to win rather than asserting a single number.
GRIDS="64 100 150 200 300"
DTYPES="float64 float32"

for DT in $DTYPES; do
  for NG in $GRIDS; do

    # --- CPU: thread scaling, so the headline speedup is quoted against a
    # --- saturated CPU rather than against one artificially idle core.
    for T in 1 2 4 8 16; do
      OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T \
        python benchmark_sq.py --device cpu --dtype "$DT" --threads "$T" \
                               --ngrid "$NG" --csv "$CSV" --label hyak
    done

    # --- GPU
    python benchmark_sq.py --device cuda --dtype "$DT" \
                           --ngrid "$NG" --csv "$CSV" --label hyak
  done
done

echo "== results: $CSV"
column -s, -t < "$CSV"
