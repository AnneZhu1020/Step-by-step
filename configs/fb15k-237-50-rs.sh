#!/usr/bin/env bash

data_dir="data/FB15K-237-50"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="False"
use_relation_space_bucketing="False"

bandwidth=400
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
bucket_interval=10
num_epochs=240
num_wait_epochs=30
num_peek_epochs=2
batch_size=96
train_batch_size=256
dev_batch_size=16
learning_rate=0.001
baseline="n/a"
grad_norm=0
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.5
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.02
relation_only="False"
#beam_size=128
#emb_2D_d1=10
#emb_2D_d2=20

ptranse_state_dict_path="model/FB15K-237-50-PTransE-xavier-200-200-0.001-0.3-0.1/checkpoint-999.tar"
distmult_state_dict_path="model/FB15K-237-50-distmult-xavier-200-200-0.003-0.3-0.1/checkpoint-15.tar"
complex_state_dict_path="model/FB15K-237-50-complex-RV-xavier-200-200-0.003-0.3-0.1/checkpoint-999.tar"
tucker_state_dict_path="model/FB15K-237-50-tucker-RV-xavier-200-200-0.0005-32-3-0.3-0.3-0.2-0.1/model_best.tar"
conve_state_dict_path="model/FB15K-237-50-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1-addtailentity-lastunique/model_best.tar"

num_paths_per_entity=-1
margin=-1

gamma=0.99
rl_module='hrl'

relation_dropout_rate=0.7
tailentity_dropout_rate=0.7

beam_size_high=8
beam_size_low=16
beam_size=128

additional_tailentity_size=96