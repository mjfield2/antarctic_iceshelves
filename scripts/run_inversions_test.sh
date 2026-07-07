#!/bin/bash
#SBATCH --job-name=bathymetry_inversion        # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=michael.field@ufl.edu  # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=4             # CPUs for data loading (adjust as needed)
#SBATCH --mem=10gb                    # Job memory request
#SBATCH --time=00:30:00               # Time limit hrs:min:sec
#SBATCH --output=logs/inversions/bathy_logs_%A.log      # Standard output and error log
#SBATCH --partition=hpg-default          
#SBATCH --account=emackie     # Your UFRC account/group
#SBATCH --qos=emackie         # Your QOS (usually same as group name)

pwd; hostname; date

ml conda
conda activate gravity

python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name pineisland_new --low 35 --high 36

date