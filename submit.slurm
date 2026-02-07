#!/bin/bash
#SBATCH --job-name=canary-generation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uddalok.sarkar@mail.utoronto.ca

#SBATCH --gpus-per-node=1
#SBATCH --time=0-2:00

module load gcc/12.3 arrow/21.0.0
module load StdEnv/2023
module load openmpi/4.1.5
module load python
source ~/.venvs/udda/bin/activate

./grid_search.sh