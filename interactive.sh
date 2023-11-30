#!/bin/bash

srun -p interactive --time=02:00:00 --mem=16G --gres=gpu:v100:2 --pty bash