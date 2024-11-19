#!/bin/bash
#SBATCH --job-name=gndvae          # Job name
#SBATCH --output=train_gndvae-%j.out  # Standard output log file (%j = JobID)
#SBATCH --error=train_gndvae-%j.err   # Standard error log file
#SBATCH --time=02:00:00             # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=16          # Increase CPUs to support multiple GPUs
#SBATCH --mem=64G                   # Increase memory for larger data processing
#SBATCH --gres=gpu:4                # Request 4 GPUs
#SBATCH --account=def-soosan        # Group account name
#SBATCH --mail-user=khbayati@torontomu.ca
#SBATCH --mail-type=ALL             # Notifications for job (BEGIN, END, FAIL, ALL)

# Load required modules
module load python/3.12
module load cuda/12.2

# Activate the virtual environment
source /home/khbayati/envs/gndvae/bin/activate

# Navigate to the project directory
cd /home/khbayati/gndvae_project/GNet_DVAE

# Run the Python script
python train_gndvae_ar.py
