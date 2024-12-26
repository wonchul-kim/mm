#!/bin/bash

PYTHONPATH=$1
GPUS=$2

python -m torch.distributed.launch --nproc_per_node=$GPUS $PYTHONPATH --launcher pytorch