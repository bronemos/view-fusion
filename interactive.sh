#!/bin/bash

srun -p interactive --time=03:00:00 --mem=16G --gres=gpu:v100:1 --pty bash