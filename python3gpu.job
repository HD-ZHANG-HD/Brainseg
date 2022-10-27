#!/usr/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=46:00:00
#SBATCH --account=ajoshi_27
#SBATCH --partition=gpu 


eval "$(conda shell.bash hook)"

conda activate brainseg
cd /scratch1/hedongzh/brainseg

echo "Checking Cuda, GPU USED?"
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.current_device()); print(torch.cuda.get_device_name(0))'
nvidia-smi


echo "Running: python " $1

python $1

