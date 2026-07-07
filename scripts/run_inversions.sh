#!/bin/bash
#SBATCH --job-name=bathymetry_inversion                 # Job name
#SBATCH --mail-type=END,FAIL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=michael.field@ufl.edu               # Where to send mail	
#SBATCH --ntasks=1                                      # Run on a single CPU
#SBATCH --cpus-per-task=4                               # CPUs for data loading (adjust as needed)
#SBATCH --mem=8gb                                       # Job memory request
#SBATCH --time=8:00:00                                 # Time limit hrs:min:sec
#SBATCH --output=logs/inversions/bathy_logs_%A_%a.log      # Standard output and error log
#SBATCH --partition=hpg-default                         # Hipergator partition
#SBATCH --account=emackie                               # Your UFRC account/group
#SBATCH --qos=emackie                                   # Your QOS (usually same as group name)
#SBATCH --array=1-20                                     # Array range

pwd; hostname; date

ml conda
conda activate gravity

# Each task will do 10 inversions
PER_TASK=10

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name abbot --low $START_NUM --high $END_NUM
# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name george --low $START_NUM --high $END_NUM
# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name getz --low $START_NUM --high $END_NUM
# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name larsen --low $START_NUM --high $END_NUM
# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name maudeast --low $START_NUM --high $END_NUM
# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name maudwest --low $START_NUM --high $END_NUM
# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name pineisland --low $START_NUM --high $END_NUM
python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name pineisland_new --low $START_NUM --high $END_NUM
# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name salzberger --low $START_NUM --high $END_NUM
# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name shackleton --low $START_NUM --high $END_NUM
# python /blue/emackie/michael.field/antarctic_iceshelves/scripts/run_inversion.py --name totten --low $START_NUM --high $END_NUM

date