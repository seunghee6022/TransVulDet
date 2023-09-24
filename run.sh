#!/bin/bash             
#SBATCH --mem=64000M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=seunghee.jeong@dlr.de
#SBATCH --mail-type=END
#SBATCH --time=1-00:00

source activate myenv

python main_deep_classifier.py