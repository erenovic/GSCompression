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
# cd /scratch_net/biwidl214/ecetin_scratch/GSCompression
# source /scratch_net/biwidl214/ecetin/conda/etc/profile.d/conda.sh
# conda activate gscodec

# From here, it's just what you executed in srun
scene_names=( 
    "tandt/truck" "tandt/train" \
    "mipnerf360/bicycle" "mipnerf360/bonsai" "mipnerf360/counter" \
    "mipnerf360/flowers" "mipnerf360/garden" "mipnerf360/kitchen" \
    "mipnerf360/room" "mipnerf360/stump" "mipnerf360/treehill" \
    "db/playroom" "db/drjohnson"
)

# scene_names=( "tandt/train" )

for scene_name in "${scene_names[@]}"
do
    
    if [ "$scene_name" == "mipnerf360/bicycle" ]; then
        capmax=6000000
    fi
    if [ "$scene_name" == "mipnerf360/bonsai" ]; then
        capmax=1200000
    fi
    if [ "$scene_name" == "mipnerf360/counter" ]; then
        capmax=1200000
    fi
    if [ "$scene_name" == "mipnerf360/flowers" ]; then
        capmax=3500000
    fi
    if [ "$scene_name" == "mipnerf360/garden" ]; then
        capmax=5800000
    fi
    if [ "$scene_name" == "mipnerf360/kitchen" ]; then
        capmax=1800000
    fi
    if [ "$scene_name" == "mipnerf360/room" ]; then
        capmax=1500000
    fi
    if [ "$scene_name" == "mipnerf360/stump" ]; then
        capmax=4800000
    fi
    if [ "$scene_name" == "mipnerf360/treehill" ]; then
        capmax=3700000
    fi
    if [ "$scene_name" == "db/playroom" ]; then
        capmax=2300000
    fi
    if [ "$scene_name" == "db/drjohnson" ]; then
        capmax=3200000
    fi
    if [ "$scene_name" == "tandt/truck" ]; then
        capmax=2500000
    fi
    if [ "$scene_name" == "tandt/train" ]; then
        capmax=1000000
    fi

    # Mean scale hyperprior from 25k
    python train_compression.py --scene_name $scene_name \
    --config ./config/preset_configs/hierarchical_mcmc_gaussian.yaml \
    --model_path ./output/mcmc_compress/$scene_name/hierarchical_0.001_from25k_freeze \
    --model complete --cap_max $capmax \
    --compressor complete_ms \
    --extra_iterations 5000 \
    --freeze_geometry \
    --checkpoint ./output/mcmc_radsplat/$scene_name/checkpoints/chkpnt25000.pth

    python test.py --scene_name $scene_name \
        --config ./config/preset_configs/hierarchical_mcmc_gaussian.yaml \
        --model_path ./output/mcmc_compress/$scene_name/hierarchical_0.001_from25k_freeze \
        --load_iteration 30000 --model complete \
        --compressor meanscale

done
