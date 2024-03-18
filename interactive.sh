#!/bin/bash

srun -p interactive --time=01:00:00 --mem=16G --gres=gpu:v100:4 --pty bash
# srun -p dgx-common --time=05:00:00 --mem=16G --gres=gpu:v100:2 --pty bash