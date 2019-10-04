#!/bin/bash

DATA=data
PRED=results/nts_step_15000

python evaluate.py $DATA/test_uni.en $DATA/references/test_references.tsv $PRED