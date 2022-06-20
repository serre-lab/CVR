#!/bin/bash

checkpoint="<path to experiment>"

test_set='<"gen" if the test is over the generalization split>'

config="${checkpoint}/config.yaml"

python inference.py \
    --config $config \
    --test_set $test_set \
    --load_checkpoint \
