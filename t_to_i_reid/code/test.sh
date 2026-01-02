#!/bin/bash
 
#SBATCH --job-name=real
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=GSEN018842
#SBATCH --partition gpu
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=64G

 
echo start time is "$(date)"
echo Slurm job ID is "${SLURM_JOBID}"

echo "SLURM Job Name: $SLURM_JOB_NAME"

#show gpu info
nvidia-smi
 

#add env lib to path
LD_LIBRARY_PATH=/user/work/zy23930/anaconda3/envs/myenv/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

#add env python to path
export PYTHONPATH=/user/work/zy23930/anaconda3/envs/myenv/lib/python3.11/site-packages:$PYTHONPATH

#module add lang/python/anaconda/pytorch

#load cuda
#module add lang/cuda/11.8

#export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

#confirm env
conda info --envs

#activate envs
conda activate /user/work/zy23930/anaconda3/envs/myenv

#confirm env
conda info --envs

#print info 


#run code
python test.py --config_file 'logs/RSTPReid/20250312_133553_iira/configs.yaml'

echo end time is "$(date)"
hostname

