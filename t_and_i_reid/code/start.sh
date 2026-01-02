#!/bin/bash
 
#SBATCH --job-name=vlmreid
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=GSEN018842
#SBATCH --partition gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --exclude=bp1-gpu030  # Exclude the problematic node
 
echo start time is "$(date)"
echo Slurm job ID is "${SLURM_JOBID}"
 
#show gpu info
nvidia-smi

# Echo the GPU node name(s)
echo "Allocated GPU node(s): $SLURM_JOB_NODELIST"

# Optional: Get the exact hostname of the node running the job
echo "Current hostname: $(hostname)"

#add env lib to path
LD_LIBRARY_PATH=/user/work/zy23930/anaconda3/envs/myenv/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

#add env python to path
export PYTHONPATH=/user/work/zy23930/anaconda3/envs/myenv/lib/python3.11/site-packages:$PYTHONPATH

#module add lang/python/anaconda/pytorch

#load cuda
module add lang/cuda/11.8

#export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

#confirm env
conda info --envs

#run code
python train.py --config_file configs/AmurTiger/vit_base.yml MODEL.DEVICE_ID "('0')"

echo end time is "$(date)"
hostname

