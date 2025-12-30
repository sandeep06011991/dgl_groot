#!/bin/bash
#SBATCH --job-name=download    # Job name
#SBATCH --output=download-%j.out # Standard output and error log
#SBATCH --error=download-%j.err  # Standard error log
#SBATCH --time=05:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=200G                    # Memory per node (e.g., 4GB)
#SBATCH --partition=cpu     # Specify the partition/queue (e.g., interactive, general)

# Load necessary modules (if any)
# module load <module_name>/<version>

# Your job commands go here     
PWD=/work/pi_huiguan_umass_edu/sandeep   
source activate ${PWD}/conda_environment/dgl_groot/

cd $PWD/code/dgl_groot
# cd -- "$(dirname -- "$0")"/../
# source init.sh

# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# echo export PYTHONPATH=${SCRIPT_DIR}/../../python
# python -c "import dgl; print ('all done')"

echo "Running on filename" $1
bash experiment/prepare_dataset/download.sh $1 

# echo "Starting my Unity job on hostname: $(hostname)"
# For example, running a Python script:
# python my_script.py
# Or executing a compiled program:
# ./my_program

echo "Unity job finished."
