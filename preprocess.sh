#!/bin/bash

onmt_preprocess \
-train_src data/train.en \
-train_tgt data/train.sen \
-valid_src data/valid_uni.en \
-valid_tgt data/valid_uni.sen \
-share_vocab \
-save_data data/EW-SEW-NTS/ew-sew