<h2 align="center"> ViewFusion: Learning Composable Diffusion Models for Novel View Synthesis
</h2>

<div align="center">
  <img width="100%" alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/9f313e8f-be97-4d5b-942e-0ffe862d1374">
</div>
<div align="center" width="70%">
  <sub>Figure 1: <strong>Architecture Overview.</strong> <em>ViewFusion</em> takes an <strong>arbitrary number</strong> of
                <strong>unordered</strong> and <strong>pose-free views</strong> coupled with the noise at timestep
                <em>t-1</em>. The inputs are denoised in
                parallel using the U-Net conditioned on timestep <em>t</em> and target viewing angle. The
                model then produces noise predictions and corresponding weights for timestep <em>t</em>. A composed
                noise prediction,
                computed as a weighted sum of individual contributions, is then subtracted from the previous timestep
                prediction. Ultimately, after <em>T</em> timesteps, a fully denoised target view is obtained. </sub>
</div>
<br>
<img width="49.7%" alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/c2d4adbb-0c48-439e-9f25-5275b36b2049">
<img width="49.7%" alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/be8b3b3e-06c3-4b68-a33c-b55942019432">
<div align="center" width="70%">
  <sub>Figure 2: <strong>Adaptive Weight Shifting.</strong> The model shifts its weighting adaptively based on the most informative input view w.r.t. the desired target output. In the examples, six evenly spaced out views of the object are passed in, depending on the target view the model puts most emphasis on the closest views. </sub>
</div>
<br>
<img alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/faae6b84-a3fa-4163-ac64-953201c3287a">
<div align="center" width="70%">
  <sub>Figure 3: <strong>Autoregressive 3D Consistency.</strong> Our approach is capable of maintaining 3D consistency through autoregressive generation even when primed solely with a single input view. We start by priming the model with a single input view, and incrementally rotate the target viewing direction to produce novel views. During the autoregressive generation, each consecutively generated view is added to the flexible conditioning for producing the next view. </sub>
</div>
<br>
This is the official implementation of 
<a href="https://arxiv.org/abs/2402.02906">ViewFusion: Learning Composable Diffusion Models for Novel View Synthesis</a>.
<br><br>

```bibtex
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

```bash
conda env create -f environment.yml
conda activate view-fusion
```

For ARM-based macOS (not tested extensively) run:

```bash
conda env create -f environment_osx.yml
conda activate view-fusion
```

### Dataset

Version of the NMR ShapeNet dataset we use is hosted by [(Niemeyer et al.)](https://github.com/autonomousvision/differentiable_volumetric_rendering). Downloadable [here](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip).<br>
Please note that our current setup is optimized for use in a cluster computing environment and requires sharding.

To ensure correct placement (in `data/nmr/`), you can download the dataset using `fetch_dataset.sh`.

Afterwards, to shard the dataset, run

```
python data/dataset_prep.py
```

The default sharding will split the dataset into four shards. In order to enable parallelization, the number of shards has to be divisible by the number of GPUs you use.

## Experiments

Configurations for various experiments can be found in `configs/`.

### Training

To launch training on a single GPU run:

```bash
python main.py -c configs/small-v100.yaml -g -t --wandb
```

For a distributed training setup run:

```bash
torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS main.py -c configs/small-v100-4.yaml -g -t --wandb
```

where `$NUM_NODES` and `$NUM_GPUS` can, for instance, be replaced by 1 and 4, respectively. This would correspond to a single-node setup with four V100 GPUs. Please note that all of the experiments were run on a single-node setup, multi-node environments have not been tested thoroughly.

(In case you are using Slurm, some example scripts are available in `slurm/`.)

### Inference

Inference mode supports a variety of visualization options that can be executed by applying their corresponding flags:

- `-gif` &ensp; produces animated generation around the axis along with the weights as shown in Figure 2.
- `-ar` &emsp; produces animated autoregressive generation as shown in Figure 3.
- `-ex` &emsp; performs extrapolation beyond six input views that are given at training time.

Pretrained model weights are available [here](https://huggingface.co/bronemos/view-fusion/resolve/main/best_model_all.pt) via HuggingFace. For running the model using provided inference script fetch the weights by running `fetch_checkpoint.sh`.

Inference can be performed on a saved checkpoint by running:

```bash
python main.py -g -i -s ./logs/pretrained --wandb -gif -ar
```

which produces GIFs as shown in Figure 2 and 3. The outputs are saved to Weights & Biases.

The setup draws random samples from validation visualisation dataloader.

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

## Repository Structure

```
view-fusion
├── configs                    # various experiment configurations
├── data                       # everything data preparation and loading related
│   ├── __init__.py
│   ├── dataset_prep.py           # script to shard the dataset
│   ├── msn_dataset.py            # not in use
│   └── nmr_dataset.py            # sample processing, dataloaders, nodesplitters
├── logs                       # default loging directory
│   └── pretrained                # default pretrained model directory
│       └── config.yaml           # pretrained model configuration
├── model                      # everything model related
│   ├── unet.py                   # unet architecture (used for denoising)
│   └── view_fusion.py            # DDPM and composable weighting logic
├── slurm                      # some slurm script examples
├── utils                      # various utilities
│   ├── __init__.py
│   ├── analysis.ipynb            # obsolete
│   ├── checkpoint.py             # checkpointing logic
│   ├── compute_metrics.py        # computes metrics on a directory containing all generated test samples
│   ├── dist.py                   # distributed training helpers
│   ├── metrics.py                # SSIM and PSNR functions
│   ├── nerf.py                   # not in use
│   ├── plot_dataset.py           # obsolete
│   └── schedulers.py             # learning rate scheduler
├── .gitignore
├── LICENSE
├── README.md
├── demo.ipynb
├── environment.yml
├── environment_osx.yml
├── experiment.py              # full experiment logic, including training, validation and inference
├── fetch_dataset.sh           # script for downloading dataset
├── fetch_pretrained.sh        # script for downloading pretrained model weights
├── inference.py               # obsolete
└── main.py                    # main
```

<!-- ## Results

<img width="49.7%" alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/c2d4adbb-0c48-439e-9f25-5275b36b2049">
<img width="49.7%" alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/be8b3b3e-06c3-4b68-a33c-b55942019432">
<div align="center" width="70%">
  <sub>Figure 2: <strong>Adaptive Weight Shifting.</strong> The model shifts its weighting adaptively based on the most informative input view w.r.t. the desired target output. In the examples, six evenly spaced out views of the object are passed in, depending on the target view the model puts most emphasis on the closest views. </sub>
</div>
<br>
<img alt="image" src="https://github.com/bronemos/view-fusion/assets/72516979/faae6b84-a3fa-4163-ac64-953201c3287a">
<div align="center" width="70%">
  <sub>Figure 3: <strong>Autoregressive 3D Consistency.</strong> Our approach is capable of maintaining 3D consistency through autoregressive generation even when primed solely with a single input view. We start by priming the model with a single input view, and incrementally rotate the target viewing direction to produce novel views. During the autoregressive generation, each consecutively generated view is added to the flexible conditioning for producing the next view. </sub>
</div>
<br> -->
