#!/bin/bash
#SBATCH  --job-name=mcmc_train
#SBATCH  --output=/scratch_net/biwidl214/ecetin_scratch/GSCompression/log/log-%j.out
#SBATCH --error=/scratch_net/biwidl214/ecetin_scratch/GSCompression/log/errors-%j.err
#SBATCH  --gres=gpu:1
#SBATCH  --cpus-per-task=4
#SBATCH  --mem=30G
#SBATCH  --time=24:00:00
#SBATCH  --nodes=1
#SBATCH  --nodelist=bmicgpu0[1-9]
##SBATCH  --partition=gpu.medium
#SBATCH  --constraint='a6000'

# echo "Starting job"
# cd /scratch_net/biwidl214/ecetin_scratch/REPROD/GSCompression
# source /scratch_net/biwidl214/ecetin/conda/etc/profile.d/conda.sh
# conda activate gscodec

# From here, it's just what you executed in srun
scene_names=( 
    "mipnerf360/garden" "mipnerf360/kitchen" \
    "mipnerf360/room" "mipnerf360/stump" "mipnerf360/treehill" \
    "db/playroom" "db/drjohnson" \
    "tandt/truck" "tandt/train" 
)

# scene_names=( tandt/train )

for scene_name in "${scene_names[@]}"
do
    python train_gaussians.py --scene_name $scene_name \
        --config ./config/preset_configs/base_gaussian.yaml \
        --model_path ./output/base_model/$scene_name \
        --model base

    python test.py --scene_name $scene_name \
        --config ./config/preset_configs/base_gaussian.yaml \
        --model_path ./output/base_model/$scene_name \
        --load_iteration 30000 --model base

done
