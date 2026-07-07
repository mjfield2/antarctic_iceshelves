#!/bin/bash
#SBATCH --job-name=hubs        # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=michael.field@ufl.edu  # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=8             # CPUs for data loading (adjust as needed)
#SBATCH --mem=14gb                    # Job memory request
#SBATCH --time=28:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/hubs_logs_%A.log      # Standard output and error log
#SBATCH --partition=hpg-default          
#SBATCH --account=emackie     # Your UFRC account/group
#SBATCH --qos=emackie         # Your QOS (usually same as group name)

ml conda
conda activate gravity

python /blue/emackie/michael.field/antarctic_iceshelves/scripts/calculate_merged_hubs.py --low 35 --high 40