#!/bin/bash

srun -p interactive --time=03:00:00 --mem=32G --gres=gpu:v100:1 --partition=gpu-v100-32g --pty bash
# srun -p dgx-common --time=05:00:00 --mem=16G --gres=gpu:v100:2 --pty bash