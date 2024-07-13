# Gaussian Splatting Representation Compression

## Abstract

3D Gaussian splatting has recently gained immense popularity due to its high parallelizability and efficiency, allowing 3D scenes to be rendered much faster than neural radiance field-based methods while maintaining comparable quality. However, representing a scene with 3D Gaussian splatting requires a large number of Gaussian primitives, from hundreds of thousands to several millions, resulting in high storage complexity and substantial memory usage.

To address this issue, we investigate the use of learned entropy models from image compression literature and residual coding for Gaussian attribute compression. We also explore enhancements to the 3D Gaussian splatting algorithm using a Markov Chain Monte Carlo framework and investigate methods to reduce the number of Gaussian primitives through learned primitive masking and importance-based pruning.
Our experiments show that optimizing Gaussian primitives with the Markov Chain Monte Carlo framework significantly improves the visual quality of novel view synthesis. Additionally, learned primitive masking and importance-based pruning can reduce the number of Gaussian primitives by up to half without notable quality loss. Furthermore, we demonstrate that learned entropy modeling, combined with a hyperprior network, can integrate seamlessly into optimized Gaussian primitives, reducing their size by up to 10 times without degrading visual quality. As the integration does not require any modification in Gaussian primitives, it is an easy method to adopt. Further investigation into hierarchy generation and residual coding reveals that hierarachy structure with an octree representation and weighted averaging does not allow for higher compression efficiency due to one-to-many mapping between covariance and scale-rotation pair.
These findings highlight the potential for substantial storage and memory improvements in 3D Gaussian splatting while maintaining high visual quality, paving the way for scalable rendering techniques.

## Environment Setup

### Requirements
For our experiments, we used the following software versions:
- Debian 11
- Python 3.10
- CUDA 11.8
- PyTorch 2.0.0
- CompressAI 1.2.6

To set the environment using the provided `environment.yml` file, you can run the following command (making sure a GPU is available for the installation process):
```bash
conda env create -f environment.yaml
conda activate gscodec
```

Please refer to [3D-MCMC Codebase](https://github.com/ubc-vision/3dgs-mcmc) for installation instructions. Make sure to install CUDA packages with the following command on an available 3DGS environment as this library has been modified:
```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/build_octree
```

As an addition to the regular packages `diff-gaussian-rasterization`, and `simple-knn`, we have added the `build_octree` package to the submodules. This package is used for building an octree structure for the hierarchical Gaussian splatting compression experiments. For installing all these packages, make sure that CUDA is available during environment setup.

## Datasets

For our experiments, we used [Tanks and Temples](https://github.com/graphdeco-inria/gaussian-splatting), [DeepBlending](https://github.com/graphdeco-inria/gaussian-splatting) and [MipNeRF](https://jonbarron.info/mipnerf360/) datasets. 

- For **Tanks and Temples**, you can install the public data from the Inria repository from the link above.
- For **DeepBlending**, you can download the scenes from the Inria repository from the link above as well.
- For **MipNeRF**, you can download the scenes from the MipNeRF repository from the link above. Note that you would require to install both parts of the dataset to run the tests.

For the `data` folder, you need to provide following folder structure: 
```
data
├── db
│   ├── drjohnson
│   │   ├── images
│   │   └── sparse
│   └── playroom
│       ├── images
│       └── sparse
├── mipnerf360
│   ├── bicycle
│   │   ├── images
│   │   ├── poses_bounds.npy
│   │   └── sparse
│   ├── bonsai
│   │   ├── images
│   │   ├── poses_bounds.npy
│   │   └── sparse
│   ├── ...
│   └── treehill
│       ├── images
│       ├── poses_bounds.npy
│       └── sparse
└── tandt
    ├── train
    │   ├── cfg_args
    │   ├── images
    │   ├── random.ply
    │   └── sparse
    └── truck
        ├── cfg_args
        ├── images
        └── sparse
```

In addition, you need to provide the base path to the `data` folder in the `config.yaml` file. For sample `config.yaml` files, please refer to the samples in the `config/preset_configs` folder.

## Running code
Running code is similar to the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) and [Original 3D-MCMC code base](https://github.com/ubc-vision/3dgs-mcmc).

Additional to the code skeleton adopted from the original 3DGS code base, we developed an experiment base for quick experimentation. Specifically, we have following folder structure:

```
.
├── README.md
├── config
│   ├── build_config_spaces.py
│   └── preset_configs
├── convert.py
├── environment.yml
├── gaussian_renderer
│   ├── __init__.py
│   ├── compression_renderer.py
│   ├── covariance_renderer.py
│   ├── masked_renderer.py
│   └── network_gui.py
├── linting.sh
├── lpipsPyTorch
│   ├── __init__.py
│   └── modules
├── models
│   ├── compression
│   └── splatting
├── remove_checkpoints.ipynb
├── scene
│   ├── __init__.py
│   ├── cameras.py
│   ├── colmap_loader.py
│   └── dataset_readers.py
├── submodules
│   ├── build_octree
│   ├── diff-gaussian-rasterization
│   ├── hierarchy_generation
│   └── simple-knn
├── test.py
├── testing
│   ├── testing_base.py
│   ├── testing_complete.py
│   ├── testing_compressed.py
│   ├── testing_hierarchical_compressed.py
│   ├── testing_masked.py
│   └── testing_res_pos_compressed.py
├── train_compression.py
├── train_gaussians.py
├── training
│   ├── compression
│   ├── log_training.py
│   ├── training_base.py
│   ├── training_masked.py
│   ├── training_masked_mcmc.py
│   ├── training_mcmc.py
│   ├── training_radsplat.py
│   └── training_utils.py
└── utils
    ├── camera_utils.py
    ├── general_utils.py
    ├── graphics_utils.py
    ├── image_utils.py
    ├── loss_utils.py
    ├── reloc_utils.py
    ├── sh_utils.py
    └── system_utils.py
```

As important components of this codebase, we have the following important folders:
- `models`: Contains the models for Gaussian splatting and compression.
    - `splatting`: Contains the Gaussian splatting models. These models include finished models such as base 3D-GS, 3D-MCMC, masked 3D-MCMC, and RadSplat 3D-MCMC. In addition, there are experimental (unfinished) models such as latent 3D-MCMC model.
    - `compression`: Contains the compression models. These models include finished models such as EntropyBottleneck and MeanScaleHyperprior. In addition, there are experimental (unfinished) models which are not covered in the report. The compression models take `base_model` as a basis for inheritance. 
    
        Finally, `mpeg-pcc-tmc13` folder includes the MPEG PCC TMC13 model which has been used for experiments and trials for Gaussian primitive position compression.
- `training`: Contains the training scripts for Gaussian splatting and compression.
    - `compression`: Contains the training scripts for compression models.
        - `training_base.py`: Contains the base training script for 3D-GS model with the entropy model of choice.
        - `training_mcmc.py`: Contains the training script for 3D-MCMC Gaussian splatting models with the entropy model of choice.
        - `training_complete.py`: Contains the training script for entropy model training with the hierarchical structure and residual coding.
    - `training_base.py`: Contains the training script for base 3D-GS model.
    - `training_masked.py`: Contains the training script for masked Gaussian splatting models.
    - `training_masked_mcmc.py`: Contains the training script for masked 3D-MCMC Gaussian splatting models. 
    - `training_mcmc.py`: Contains the training script for 3D-MCMC Gaussian splatting models.
    - `training_radsplat.py`: Contains the training script for RadSplat 3D-MCMC Gaussian splatting models. 
    - `training_utils.py`: Contains the utility functions for training scripts.
    - `log_training.py`: Contains the logging functions for training scripts.
- `testing`: Contains the testing scripts for Gaussian splatting and compression.
    - `testing_base.py`: Contains the base testing script for 3D-GS model without any compression.
    - `testing_masked.py`: Contains the testing script for masked 3D-GS model without any compression.
    - `testing_compressed.py`: Contains the testing script for compressed 3D-GS model with the entropy model of choice (meanscale or entropybottleneck). This script is NOT used for compression with the hierarchical structure.
    - `testing_complete.py`: Contains the testing script for compressed 3D-GS model with the entropy model of choice with the hierarchical structure and residual coding.
    - `testing_utils.py`: Contains the utility functions for testing scripts.

### Gaussian Splatting Optimization
To compress a 3D Gaussian splat representation, first requirement is to optimize the Gaussian splat representation.
With this repository, you can optimize Gaussian splat representations with following options:
1. Regular Gaussian Splatting (3D-GS)
2. Gaussian Splatting with MCMC (3D-MCMC)
3. Gaussian Splatting with MCMC and Learned Masking
4. Gaussian Splatting with MCMC and RadSplat pruning method

For further details on how to run the experiments with these 4 methods, you can refer to slurm scipt `./slurm_scripts/mcmc_train.sh`.

#### 1. Regular Gaussian Splatting (3D-GS):
This is the original optimization method proposed in the 3DGS paper. To optimize a Gaussian splat representation with 3D-GS, you can run the following command:

```bash
python train_gaussians.py --scene_name $scene_name \
    --config_path ./config/preset_configs/base_gaussian.yaml \
    --model_path $model_path \
    --model base
```
where `$scene_name` is the name of the scene, `$config_path` is the path to the configuration file, and `$model_path` is the path to the model file.

An example for the `scene_name` is `tandt/train`, and for `model_path` is `./output/base_model/$scene_name`.

#### 2. Gaussian Splatting with MCMC (3D-MCMC):
This is the optimization method proposed in the [3D-MCMC paper](https://ubc-vision.github.io/3dgs-mcmc/). To optimize a Gaussian splat representation with 3D-MCMC, you can run the following command:

```bash
python train_gaussians.py --scene_name $scene_name \
    --config_path ./config/preset_configs/mcmc_gaussian.yaml \
    --model_path $model_path \
    --model mcmc \
    --cap_max $cap_max \
    --radsplat_prune_at 1000000
```
where `$scene_name` is the name of the scene,  and `$model_path` is the path to the model file. The `cap_max` parameter is the maximum number of Gaussians to be optimized. For a list of suitable values, please refer to the `./slurm_scripts/mcmc_train.sh` file.

An example for the `scene_name` is `tandt/train`, and for `model_path` is `./output/mcmc_model/$scene_name`.

**NOTE:** By setting `radsplat_prune_at` to a value greater than `30000` which is the default total iterations, we are disabling RadSplat pruning method. The sample config files in `./config/preset_configs` use `radsplat_prune_at` as `[16000, 24000]` to apply pruning twice as described in RadSplat pruning section.

#### 3. Gaussian Splatting with MCMC and Learned Masking:
This is the optimization method proposed in the [3D-MCMC paper](https://ubc-vision.github.io/3dgs-mcmc) with additional learned masking which was introduced in [Compact3D-GS](https://maincold2.github.io/c3dgs/). To optimize a Gaussian splat representation with 3D-MCMC and learned masking, you can run the following command:

```bash
python train_gaussians.py --scene_name $scene_name \
    --config_path ./config/preset_configs/mcmc_gaussian.yaml \
    --model_path $model_path \
    --model masked_mcmc \
    --cap_max $cap_max
```
where `$scene_name` is the name of the scene,  and `$model_path` is the path to the model file.

#### 4. Gaussian Splatting with MCMC and RadSplat pruning method:
This is the optimization method proposed in the [3D-MCMC paper](https://ubc-vision.github.io/3dgs-mcmc) with additional RadSplat pruning method which was introduced in [RadSplat](https://m-niemeyer.github.io/radsplat/). To optimize a Gaussian splat representation with 3D-MCMC and RadSplat pruning method, you can run the following command:

```bash
python train_gaussians.py --scene_name $scene_name \
    --config_path ./config/preset_configs/mcmc_gaussian.yaml \
    --model_path $model_path \
    --model mcmc
```
where `$scene_name` is the name of the scene,  and `$model_path` is the path to the model file.

**NOTE:** The RadSplat pruning method specifications are listed in `./config/preset_configs/mcmc_gaussian.yaml` by not providing any `radsplat_prune_at` value, we use the default values set in the config file.

### Gaussian Splatting Compression

To compress Gaussian primitives and Gaussian splat representation, we need to provide the checkpoint to train the entropy model. With this repository, you can train entropy models with following options:
1. Compress with Fully-Factorized entropy model (EntropyBottleneck)
2. Compress with Mean-Scale Hyperprior entropy model (MeanScaleHyperprior)

For further details on how to run the experiments with these 2 methods, you can refer to slurm scipt `./slurm_scripts/mcmc_compression_from25k_train.sh`.

To train the entropy model in accordance to descriptions in the report provided, you can run the following command:

```bash
python train_compression.py --scene_name $scene_name \
    --config_path ./config/preset_configs/mcmc_compress.yaml \
    --model_path $model_path \
    --model mcmc \
    --cap_max $cap_max \
    --compressor $compressor \
    --extra_iterations 5000 \
    --freeze_geometry \
    --checkpoint $checkpoint
```
where `$scene_name` is the name of the scene, `$model_path` is the path to the model file, `$cap_max` is the maximum number of Gaussians, `$compressor` is the entropy model type (either `entropybottleneck` or `meanscale`), and `$checkpoint` is the path to the Gaussian splat checkpoint provided to be compressed.

Multiple examples for how to run the compression can be found in the `./slurm_scripts/mcmc_compression_from25k_train.sh` script. An additional example is provided below:

```bash
python train_compression.py --scene_name tandt/train \
    --config_path ./config/preset_configs/mcmc_compress.yaml \
    --model_path ./output/mcmc_compress/tandt/train \
    --model mcmc \
    --cap_max 600000 \
    --compressor meanscale \
    --extra_iterations 5000 \
    --freeze_geometry \
    --checkpoint ./output/mcmc_model/tandt/train/chkpnt25000.pth
```

### Hierarchical Gaussian Splatting Compression

To repeat experiments with hierarchical Gaussian splat compression, you can use the same pretrained Gaussian splat models and compress them with the hierarchical structure. For this, you can use the following command:

```bash
python train_compression.py --scene_name $scene_name \
    --config_path ./config/preset_configs/hierarchical_mcmc_gaussian.yaml \
    --model_path $model_path \
    --model complete \
    --cap_max $cap_max \
    --compressor complete_ms \
    --extra_iterations 5000 \
    --freeze_geometry \
    --checkpoint $checkpoint
```

where `$scene_name` is the name of the scene, `$model_path` is the path to the model file, `$cap_max` is the maximum number of Gaussians, `$compressor` is the entropy model type (`complete_ms`, we use the mean-scale hyperprior due to its superior compression efficiency), and `$checkpoint` is the path to the Gaussian splat checkpoint provided to be compressed.

Multiple examples for how to run the compression can be found in the `./slurm_scripts/hierarchical_from25k_train.sh` script. An additional example is provided below:

```bash
python train_compression.py --scene_name tandt/train \
    --config_path ./config/preset_configs/hierarchical_mcmc_gaussian.yaml \
    --model_path ./output/residual_coding/tandt/train \
    --model mcmc \
    --cap_max 600000 \
    --compressor complete_ms \
    --extra_iterations 5000 \
    --freeze_geometry \
    --checkpoint ./output/mcmc_model/tandt/train/chkpnt25000.pth
```

## Testing

To test the compressed Gaussian splat representation, you can use `test.py` script with respective config file. To run the script, you need to provide following arguments:

- `--scene_name`: Name of the scene to be tested. (e.g. `tandt/train`)
- `--config_path`: Path to the config file. (e.g. `./config/preset_configs/mcmc_radsplat.yaml`)
- `--model_path`: Path to the model training folder. (e.g. `./output/mcmc_radsplat/tandt/train`)
- `--model`: Model type to be tested. (e.g. `mcmc`, `radsplat`, `complete`,...)
- `--load_iteration`: Iteration number to be loaded from the checkpoints `folder` in `model_path`. (e.g. `30000`)
- `--compressor`: Compressor type to be used for testing. (e.g. `meanscale`, `entropybottleneck`, nothing if no compression)

For further examples, please check the `./slurm_scripts/` folder where training and testing pairs are provided.

## Acknowledgement
This repository is based on the [3DGS codebase](https://github.com/graphdeco-inria/gaussian-splatting) and [3D-MCMC codebase](https://ubc-vision.github.io/3dgs-mcmc). We would like to thank the authors of these repositories for their contributions.