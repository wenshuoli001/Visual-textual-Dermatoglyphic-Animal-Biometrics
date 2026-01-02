#!/bin/bash
 
#SBATCH --job-name=vlmreid_imageonly
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=GSEN018842
#SBATCH --partition gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=64G
 
 
echo start time is "$(date)"
echo Slurm job ID is "${SLURM_JOBID}"


#show gpu info
nvidia-smi


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
python test.py --config_file configs/AmurTiger/vit_base.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT 'result/clipmix_650.pth'

echo end time is "$(date)"
hostname

