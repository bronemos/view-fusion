#!/bin/bash

srun -p interactive --time=06:00:00 --mem=16G --gres=gpu:a100:1 --pty bash