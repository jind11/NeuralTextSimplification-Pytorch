#!/bin/bash

CUDA_VISIBLE_DEVICES=2 onmt_translate \
-model models/nts_step_15000.pt \
-src data/test_uni.en \
-replace_unk \
-verbose \
-beam_size 5 \
-n_best 4 \
-share_vocab \
-output results/nts_step_15000