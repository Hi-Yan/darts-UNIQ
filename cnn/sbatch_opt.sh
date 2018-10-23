#!/bin/bash
#SBATCH -J "OPT"
#SBATCH -t 01-00:00:00
#SBATCH -p all
#SBATCH --mail-user=yochaiz@cs.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

# source /etc/profile.d/modules.sh
# module load cuda
source ~/venv/bin/activate # activate python3 virtual environment
cd ~/F-BANNAS/cnn
echo $1
echo $2
PYTHONPATH=../ python3 sbatch_opt.py $1 "$2"