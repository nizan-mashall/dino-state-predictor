#!/bin/bash

source ~/dino-state-predictor/myenv/bin/activate

python -m robomimic.scripts.download_datasets \
    --tasks lift \
    --dataset_types mg \
    --download_dir /users/ogal/nmashall/dino-state-predictor/data

# --tasks:        lift, can, square, transport, tool_hang
# --dataset_types:
#   ph  = proficient human (200 demos, 1 operator)   ← start here
#   mh  = mixed human (300 demos, 6 operators)       ← more multimodality
#   mg  = machine generated (RL policy)              ← consistent but not human
#   paired = ph + mh combined