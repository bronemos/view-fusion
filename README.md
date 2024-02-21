<h2 align="center"> ViewFusion: Learning Composable Diffusion Models for Novel View Synthesis
</h2>

<div align="center">
  <img width="100%" alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/9f313e8f-be97-4d5b-942e-0ffe862d1374">
</div>
<div align="center" width="70%">
  <sub><strong>Fig. 1. Architecture Overview.</strong> <em>ViewFusion</em> takes an <strong>arbitrary number</strong> of
                <strong>unordered</strong> and <strong>pose-free views</strong> coupled with the noise at timestep
                <em>t-1</em>. The inputs are denoised in
                parallel using the U-Net conditioned on timestep <em>t</em> and target viewing angle. The
                model then produces noise predictions and corresponding weights for timestep <em>t</em>. A composed
                noise prediction,
                computed as a weighted sum of individual contributions, is then subtracted from the previous timestep
                prediction. Ultimately, after <em>T</em> timesteps, a fully denoised target view is obtained. </sub>
</div>
<br>
This is the official implementation of 
"<a href="https://arxiv.org/abs/2402.02906">ViewFusion: Learning Composable Diffusion Models for Novel View Synthesis</a>".
<br><br>

```
@misc{spiegl2024viewfusion,
      title={ViewFusion: Learning Composable Diffusion Models for Novel View Synthesis},
      author={Bernard Spiegl and Andrea Perin and Stéphane Deny and Alexander Ilin},
      year={2024},
      eprint={2402.02906},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Setup

### Environment

You can install and activate the conda environment by simply running:

```
conda env create -f environment.yml
conda activate view-fusion
```

For ARM-based macOS run:

```
conda env create -f environment_osx.yml
conda activate view-fusion
```

### Dataset

Version of the NMR ShapeNet dataset we use is hosted by [(Niemeyer et al.)](https://github.com/autonomousvision/differentiable_volumetric_rendering). Downloadable [here](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip).<br>
Please note that our current setup is optimized for use in a cluster computing environment and requires sharding.

To shard the dataset, place the `NMR_Dataset.zip` in `data/nmr/` and run `python data/dataset_prep.py` command. The default sharding will split the dataset into four shards. In order to enable parallelization, the number of shards has to be divisible by the number of GPUs you use.

## Experiments - Work In Progress!

Configurations for various experiments are located in `configs/`.

### Training

To launch training on a single GPU run:

```
python main.py -c configs/multi-view-composable-variable-small-v100.yaml -g -t --wandb
```

For a distributed setup run:

```
torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS main.py -c configs/multi-view-composable-variable-small-v100-4.yaml -g -t --wandb
```

where `$NUM_NODES` and `$NUM_GPUS` can, for instance, be replaced by 1 and 4, respectively. This would correspond to a single-node setup with four V100 GPUs.

(In case you are using Slurm, more example scripts are available in `slurm/`.)

### Inference

Coming soon.

### Eval

Coming soon.

### Using Only the Model

In case you want to implement separate data pipelines or training procedures, all the architecture details are available in `model/`.

At training time, the model receives:

- `y_0` which is the target (ground truth) of shape `(B C H W)`,
- `y_cond` which contains all the input views and is of shape `(B N C H W)` where N denotes the total number of views (24 in our case),
- `view_count` of shape `(B,)` which contains the number of views used as conditioning for each sample in the batch,
- `angle` also of shape `(B,)` indicating the target angle for each sample.

At inference time, `y_0` is omitted, with everything else remaining the same as training. <br>
See paper for full implementation details.

### Resource Requirements

**NB** Training configurations require significant amount of VRAM.<br>
The model referenced in the paper was trained using `configs/multi-view-composable-variable-small-v100-4.yaml` configuration for 710k steps (approx. 6.5 days) on 4x V100 GPUs, each with 32GB VRAM.<br>
Pretrained model weights will be made available soon.

## Results

<img width="49.7%" alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/c2d4adbb-0c48-439e-9f25-5275b36b2049">
<img width="49.7%" alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/be8b3b3e-06c3-4b68-a33c-b55942019432">
<img alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/faae6b84-a3fa-4163-ac64-953201c3287a">
