#!/bin/bash

DATA=data
PRED=results/nts_step_20000

python evaluate.py $DATA/test_uni.en $DATA/test_references.tsv $PRED