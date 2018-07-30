#!/bin/bash
#SBATCH -c 3 # number of cores
#SBATCH --gres=gpu:1 # number of gpu requested
#SBATCH -J "DARTS"
#SBATCH -t 03-00:00:00
#SBATCH -p gip,all
#SBATCH --mail-user=yochaiz@cs.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

cd ~/darts-UNIQ/cnn
source ~/tf3/bin/activate # activate python3 virtual environment
python3 sbatch.py

