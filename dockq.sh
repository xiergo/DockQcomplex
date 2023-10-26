#!/usr/bin/bash

#SBATCH -p hygon
## SBATCH --gres=gpu:1
#SBATCH -c 64
#SBATCH -o dockq_%j.out

set -e
set -u
set -o pipefail

echo Start: $(date)
# python cal_dockq.py # af-multimer
# python cal_dockq_myinfer.py # myinfer
python cal_dockq_5model.py #5model
echo Finished! $(date)