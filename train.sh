#!/bin/bash

CUDA_VISIBLE_DEVICES=2 onmt_train \
--data data/EW-SEW-NTS/ew-sew \
--save_model models/nts \
--world_size 1 \
--gpu_ranks 0 \
--batch_size 64 \
--train_steps 60000 \
--valid_steps 5000 \
--early_stopping 2 \
--max_grad_norm 5 \
--dropout 0.3 \
--feat_vec_size 20 \
--learning_rate_decay 0.7 \
--word_vec_size 500 \
--share_embeddings \
--share_decoder_embeddings \
--model_type text \
--encoder_type rnn \
--rnn_type LSTM \
--layers 2 \
--rnn_size 500 \
--global_attention general \

