#!/bin/bash
#SBATCH  --job-name=mcmc_train
#SBATCH  --output=/scratch_net/biwidl214/ecetin_scratch/renewed_repo/log/log-%j.out
#SBATCH --error=/scratch_net/biwidl214/ecetin_scratch/renewed_repo/log/errors-%j.err
#SBATCH  --gres=gpu:1
#SBATCH  --cpus-per-task=4
#SBATCH  --mem=30G
#SBATCH  --time=24:00:00
#SBATCH  --nodes=1
#SBATCH  --nodelist=bmicgpu0[1-9]
##SBATCH  --partition=gpu.medium
#SBATCH  --constraint='a6000'

# echo "Starting job"
cd /scratch_net/biwidl214/ecetin_scratch/renewed_repo
source /scratch_net/biwidl214/ecetin/conda/etc/profile.d/conda.sh
conda activate gscodec

# From here, it's just what you executed in srun
# "mipnerf360/bicycle" "mipnerf360/bonsai" "mipnerf360/counter" "mipnerf360/flowers"
scene_names=( 
    "mipnerf360/garden" "mipnerf360/kitchen" \
    "mipnerf360/room" "mipnerf360/stump" "mipnerf360/treehill" \
    "db/playroom" "db/drjohnson" \
    "tandt/truck" "tandt/train" 
)

for scene_name in "${scene_names[@]}"
do
    # python train_gaussians.py --scene_name $scene_name \
    # --config ./config/preset_configs/mcmc_gaussian.yaml \
    # --model_path ./output/mcmc_1M/$scene_name \
    # --model mcmc --cap_max 2500000 \
    # --radsplat_prune_at 1000000

    # python test.py --scene_name $scene_name \
    #     --config ./config/preset_configs/mcmc_gaussian.yaml \
    #     --model_path ./output/mcmc_1M/$scene_name \
    #     --load_iteration 30000 --model mcmc

    python train_gaussians.py --scene_name $scene_name \
        --config ./config/preset_configs/base_gaussian.yaml \
        --model_path ./output/base_model/$scene_name \
        --model base

    python test.py --scene_name $scene_name \
        --config ./config/preset_configs/base_gaussian.yaml \
        --model_path ./output/base_model/$scene_name \
        --load_iteration 30000 --model base

    # python train_gaussians.py --scene_name tandt/$scene_name \
    #     --config ./config/preset_configs/mcmc_gaussian.yaml \
    #     --model_path ./output/mcmc_masked_1M/$scene_name \
    #     --model masked_mcmc --cap_max 2500000

    # python test.py --scene_name tandt/$scene_name \
    #     --config ./config/preset_configs/mcmc_gaussian.yaml \
    #     --model_path ./output/mcmc_masked_1M/$scene_name \
    #     --load_iteration 30000 --model masked_mcmc

    # python train_gaussians.py --scene_name tandt/$scene_name \
    #     --config ./config/preset_configs/mcmc_gaussian.yaml \
    #     --model_path ./output/mcmc_radsplat_1M/$scene_name \
    #     --model mcmc --cap_max 2500000

    # python test.py --scene_name tandt/$scene_name \
    #     --config ./config/preset_configs/mcmc_gaussian.yaml \
    #     --model_path ./output/mcmc_radsplat_1M/$scene_name \
    #     --load_iteration 30000 --model mcmc
done
