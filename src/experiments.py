#!/usr/bin/env python3

"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Experiment Portal.
"""

import copy
import itertools
import numpy as np
import os, sys
import random
import logging

import torch

from src.parse_args import parser
from src.parse_args import args
import src.data_utils as data_utils
import src.eval
from src.knowledge_graph import KnowledgeGraph
from src.emb.fact_network import ComplEx, ConvE, DistMult
from src.emb.fact_network import get_conve_kg_state_dict, get_complex_kg_state_dict, get_distmult_kg_state_dict
from src.emb.emb import EmbeddingBasedMethod
from src.rl.graph_search.pn import GraphSearchPolicy
from src.rl.graph_search.pg import PolicyGradient
from src.rl.graph_search.rs_pg import RewardShapingPolicyGradient

torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test,
                                     args.add_reverse_relations)


def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''
    entire_graph_tag = '-EG' if args.train_entire_graph else ''
    if args.xavier_initialization:
        initialization_tag = '-xavier'
    elif args.uniform_entity_initialization:
        initialization_tag = '-uniform'
    else:
        initialization_tag = ''

    # Hyperparameter signature

    if args.model in ['rule']:
        print("rule")
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.history_num_layers,
            args.learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.bandwidth,
            args.beta,
            args.rl_module,
            args.beam_size_high,
            args.beam_size_low
        )
    elif args.model.startswith('point'):
        print("point")
        if args.baseline == 'avg_reward':
            logging.info('* Policy Gradient Baseline: average reward')
        elif args.baseline == 'avg_reward_normalized':
            logging.info('* Policy Gradient Baseline: average reward baseline plus normalization')
        else:
            logging.info('* Policy Gradient Baseline: None')
        if args.action_dropout_anneal_interval < 1000:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.action_dropout_anneal_factor,
                args.action_dropout_anneal_interval,
                args.bandwidth,
                args.beta,
                args.gamma,
                args.num_rollout_steps,
                args.num_rollouts,
                args.rl_module,
                args.beam_size_high,
                args.beam_size_low,
                args.beam_size,
                args.relation_dropout_rate,
                args.tailentity_dropout_rate,
                args.train_batch_size,
                args.dev_batch_size,
                args.additional_tailentity_size
            )
            if args.mu != 1.0:
                hyperparam_sig += '-{}'.format(args.mu)
        else:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.bandwidth,
                args.beta,
                args.gamma,
                args.num_rollout_steps,
                args.num_rollouts,
                args.rl_module,
                args.beam_size_high,
                args.beam_size_low,
                args.beam_size,
                args.relation_dropout_rate,
                args.tailentity_dropout_rate,
                args.train_batch_size,
                args.dev_batch_size,
                args.additional_tailentity_size
            )
        if args.reward_shaping_threshold > 0:
            hyperparam_sig += '-{}'.format(args.reward_shaping_threshold)
    elif args.model == 'distmult':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model == 'complex':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model in ['conve', 'hypere', 'triplee']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.num_out_channels,
            args.kernel_size,
            args.emb_dropout_rate,
            args.hidden_dropout_rate,
            args.feat_dropout_rate,
            args.label_smoothing_epsilon
        )
    else:
        raise NotImplementedError
    model_sub_dir = '{}-{}{}{}{}-{}'.format(
        dataset,
        args.model,
        reverse_edge_tag,
        entire_graph_tag,
        initialization_tag,
        hyperparam_sig
    )

    if args.model == 'set':
        model_sub_dir += '-{}'.format(args.beam_size)
        model_sub_dir += '-{}'.format(args.num_paths_per_entity)
    if args.relation_only:
        model_sub_dir += '-ro'
    elif args.relation_only_in_path:
        model_sub_dir += '-rpo'
    elif args.type_only:
        model_sub_dir += '-to'

    if args.test:
        model_sub_dir += '-test'

    if random_seed:
        model_sub_dir += '.{}'.format(random_seed)

    model_dir = os.path.join(model_root_dir, 'test/' + model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logging.info('Model directory created: {}'.format(model_dir))
    else:
        logging.info('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir


def construct_model(args):
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)
    if args.model.endswith('.gc'):
        kg.load_fuzzy_facts()

    if args.model in ['point', 'point.gc']:
        # pn => policy network
        pn = GraphSearchPolicy(args)
        # lf => learn framework
        lf = PolicyGradient(args, kg, pn)
    elif args.model.startswith('point.rs'):
        fn_model = args.model.split('.')[2]
        fn_args = copy.deepcopy(args)
        fn_args.model = fn_model
        fn_args.relation_only = False
        if fn_model == 'complex':
            fn = ComplEx(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'distmult':
            fn = DistMult(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'conve':
            fn = ConvE(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        pn = GraphSearchPolicy(args, fn, fn_kg)
        lf = RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn)
    elif args.model == 'complex':
        fn = ComplEx(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'distmult':
        fn = DistMult(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'conve':
        fn = ConvE(args, kg.num_entities)
        lf = EmbeddingBasedMethod(args, kg, fn)
    else:
        raise NotImplementedError
    return lf


def train(lf):
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    test_path = os.path.join(args.data_dir, 'test.triples')

    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')

    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()

    train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges)
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path,
                                       seen_entities=seen_entities)

    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    lf.run_train(train_data, dev_data)


def inference(lf):
    lf.batch_size = args.dev_batch_size
    lf.eval()
    if args.model == 'hypere':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        secondary_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(secondary_kg_state_dict)
    elif args.model == 'triplee':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        complex_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(complex_kg_state_dict)
        distmult_kg_state_dict = get_distmult_kg_state_dict(torch.load(args.distmult_state_dict_path))
        lf.tertiary_kg.load_state_dict(distmult_kg_state_dict)
    else:
        lf.load_checkpoint(get_checkpoint_path(args))
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()

    eval_metrics = {
        'dev': {},
        'test': {}
    }

    dev_path = os.path.join(args.data_dir, 'dev.triples')
    test_path = os.path.join(args.data_dir, 'test.triples')
    dev_data = data_utils.load_triples(
        dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
    test_data = data_utils.load_triples(
        test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
    logging.info('Dev set performance:')
    pred_scores = lf.forward(dev_data, verbose=False)
    dev_metrics = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
    eval_metrics['dev'] = {}
    eval_metrics['dev']['hits_at_1'] = dev_metrics[0]
    eval_metrics['dev']['hits_at_3'] = dev_metrics[1]
    eval_metrics['dev']['hits_at_5'] = dev_metrics[2]
    eval_metrics['dev']['hits_at_10'] = dev_metrics[3]
    eval_metrics['dev']['mrr'] = dev_metrics[4]
    src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
    logging.info('Test set performance:')
    pred_scores = lf.forward(test_data, verbose=args.save_beam_search_paths)
    test_metrics = src.eval.hits_and_ranks(test_data, pred_scores, lf.kg.all_objects, verbose=True)
    eval_metrics['test']['hits_at_1'] = test_metrics[0]
    eval_metrics['test']['hits_at_3'] = test_metrics[1]
    eval_metrics['test']['hits_at_5'] = test_metrics[2]
    eval_metrics['test']['hits_at_10'] = test_metrics[3]
    eval_metrics['test']['mrr'] = test_metrics[4]

    return eval_metrics


def run_ablation_studies(args):
    """
    Run the ablation study experiments reported in the paper.
    """

    def set_up_lf_for_inference(args):
        initialize_model_directory(args)
        lf = construct_model(args)
        lf.cuda()
        lf.batch_size = args.dev_batch_size
        lf.load_checkpoint(get_checkpoint_path(args))
        lf.eval()
        return lf

    def rel_change(metrics, ab_system, kg_portion):
        ab_system_metrics = metrics[ab_system][kg_portion]
        base_metrics = metrics['ours'][kg_portion]
        return int(np.round((ab_system_metrics - base_metrics) / base_metrics * 100))

    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dataset = os.path.basename(args.data_dir)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(
        dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
    to_m_rels, to_1_rels, (to_m_ratio, to_1_ratio) = data_utils.get_relations_by_type(args.data_dir,
                                                                                      relation_index_path)
    relation_by_types = (to_m_rels, to_1_rels)
    to_m_ratio *= 100
    to_1_ratio *= 100
    seen_queries, (seen_ratio, unseen_ratio) = data_utils.get_seen_queries(args.data_dir, entity_index_path,
                                                                           relation_index_path)
    seen_ratio *= 100
    unseen_ratio *= 100

    systems = ['ours', '-ad', '-rs']
    mrrs, to_m_mrrs, to_1_mrrs, seen_mrrs, unseen_mrrs = {}, {}, {}, {}, {}
    for system in systems:
        logging.info('** Evaluating {} system **'.format(system))
        if system == '-ad':
            args.action_dropout_rate = 0.0
            if dataset == 'umls':
                # adjust dropout hyperparameters
                args.emb_dropout_rate = 0.3
                args.ff_dropout_rate = 0.1
        elif system == '-rs':
            config_path = os.path.join('configs', '{}.sh'.format(dataset.lower()))
            args = parser.parse_args()
            args = data_utils.load_configs(args, config_path)

        lf = set_up_lf_for_inference(args)
        pred_scores = lf.forward(dev_data, verbose=False)
        _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
        if to_1_ratio == 0:
            to_m_mrr = mrr
            to_1_mrr = -1
        else:
            to_m_mrr, to_1_mrr = src.eval.hits_and_ranks_by_relation_type(
                dev_data, pred_scores, lf.kg.dev_objects, relation_by_types, verbose=True)
        seen_mrr, unseen_mrr = src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries, verbose=True)
        mrrs[system] = {'': mrr * 100}
        to_m_mrrs[system] = {'': to_m_mrr * 100}
        to_1_mrrs[system] = {'': to_1_mrr * 100}
        seen_mrrs[system] = {'': seen_mrr * 100}
        unseen_mrrs[system] = {'': unseen_mrr * 100}
        _, _, _, _, mrr_full_kg = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
        if to_1_ratio == 0:
            to_m_mrr_full_kg = mrr_full_kg
            to_1_mrr_full_kg = -1
        else:
            to_m_mrr_full_kg, to_1_mrr_full_kg = src.eval.hits_and_ranks_by_relation_type(
                dev_data, pred_scores, lf.kg.all_objects, relation_by_types, verbose=True)
        seen_mrr_full_kg, unseen_mrr_full_kg = src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries, verbose=True)
        mrrs[system]['full_kg'] = mrr_full_kg * 100
        to_m_mrrs[system]['full_kg'] = to_m_mrr_full_kg * 100
        to_1_mrrs[system]['full_kg'] = to_1_mrr_full_kg * 100
        seen_mrrs[system]['full_kg'] = seen_mrr_full_kg * 100
        unseen_mrrs[system]['full_kg'] = unseen_mrr_full_kg * 100

    # overall system comparison (table 3)
    logging.info('Partial graph evaluation')
    logging.info('--------------------------')
    logging.info('Overall system performance')
    logging.info('Ours(ConvE)\t-RS\t-AD')
    logging.info('{:.1f}\t{:.1f}\t{:.1f}'.format(mrrs['ours'][''], mrrs['-rs'][''], mrrs['-ad']['']))
    logging.info('--------------------------')
    # performance w.r.t. relation types (table 4, 6)
    logging.info('Performance w.r.t. relation types')
    logging.info('\tTo-many\t\t\t\tTo-one\t\t')
    logging.info('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    logging.info('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        to_m_ratio, to_m_mrrs['ours'][''], to_m_mrrs['-rs'][''], rel_change(to_m_mrrs, '-rs', ''), to_m_mrrs['-ad'][''],
        rel_change(to_m_mrrs, '-ad', ''),
        to_1_ratio, to_1_mrrs['ours'][''], to_1_mrrs['-rs'][''], rel_change(to_1_mrrs, '-rs', ''), to_1_mrrs['-ad'][''],
        rel_change(to_1_mrrs, '-ad', '')))
    logging.info('--------------------------')
    # performance w.r.t. seen queries (table 5, 7)
    logging.info('Performance w.r.t. seen/unseen queries')
    logging.info('\tSeen\t\t\t\tUnseen\t\t')
    logging.info('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    logging.info('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        seen_ratio, seen_mrrs['ours'][''], seen_mrrs['-rs'][''], rel_change(seen_mrrs, '-rs', ''), seen_mrrs['-ad'][''],
        rel_change(seen_mrrs, '-ad', ''),
        unseen_ratio, unseen_mrrs['ours'][''], unseen_mrrs['-rs'][''], rel_change(unseen_mrrs, '-rs', ''),
        unseen_mrrs['-ad'][''], rel_change(unseen_mrrs, '-ad', '')))
    logging.info()
    logging.info('Full graph evaluation')
    logging.info('--------------------------')
    logging.info('Overall system performance')
    logging.info('Ours(ConvE)\t-RS\t-AD')
    logging.info(
        '{:.1f}\t{:.1f}\t{:.1f}'.format(mrrs['ours']['full_kg'], mrrs['-rs']['full_kg'], mrrs['-ad']['full_kg']))
    logging.info('--------------------------')
    logging.info('Performance w.r.t. relation types')
    logging.info('\tTo-many\t\t\t\tTo-one\t\t')
    logging.info('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    logging.info('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        to_m_ratio, to_m_mrrs['ours']['full_kg'], to_m_mrrs['-rs']['full_kg'], rel_change(to_m_mrrs, '-rs', 'full_kg'),
        to_m_mrrs['-ad']['full_kg'], rel_change(to_m_mrrs, '-ad', 'full_kg'),
        to_1_ratio, to_1_mrrs['ours']['full_kg'], to_1_mrrs['-rs']['full_kg'], rel_change(to_1_mrrs, '-rs', 'full_kg'),
        to_1_mrrs['-ad']['full_kg'], rel_change(to_1_mrrs, '-ad', 'full_kg')))
    logging.info('--------------------------')
    logging.info('Performance w.r.t. seen/unseen queries')
    logging.info('\tSeen\t\t\t\tUnseen\t\t')
    logging.info('%\tOurs\t-RS\t-AD\t%\tOurs\t-RS\t-AD')
    logging.info('{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})\t{:.1f}\t{:.1f}\t{:.1f} ({:d})\t{:.1f} ({:d})'.format(
        seen_ratio, seen_mrrs['ours']['full_kg'], seen_mrrs['-rs']['full_kg'], rel_change(seen_mrrs, '-rs', 'full_kg'),
        seen_mrrs['-ad']['full_kg'], rel_change(seen_mrrs, '-ad', 'full_kg'),
        unseen_ratio, unseen_mrrs['ours']['full_kg'], unseen_mrrs['-rs']['full_kg'],
        rel_change(unseen_mrrs, '-rs', 'full_kg'), unseen_mrrs['-ad']['full_kg'],
        rel_change(unseen_mrrs, '-ad', 'full_kg')))


def export_to_embedding_projector(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_to_embedding_projector()


def export_reward_shaping_parameters(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_reward_shaping_parameters()


def export_fuzzy_facts(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_fuzzy_facts()


def export_error_cases(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.batch_size = args.dev_batch_size
    lf.eval()
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    lf.load_checkpoint(get_checkpoint_path(args))
    logging.info('Dev set performance:')
    pred_scores = lf.forward(dev_data, verbose=False)
    src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
    src.eval.export_error_cases(dev_data, pred_scores, lf.kg.dev_objects, os.path.join(lf.model_dir, 'error_cases.pkl'))


def compute_fact_scores(lf):
    data_dir = args.data_dir
    train_path = os.path.join(data_dir, 'train.triples')
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(train_path, entity_index_path, relation_index_path)
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    test_data = data_utils.load_triples(test_path, entity_index_path, relation_index_path)
    lf.eval()
    lf.load_checkpoint(get_checkpoint_path(args))
    train_scores = lf.forward_fact(train_data)
    dev_scores = lf.forward_fact(dev_data)
    test_scores = lf.forward_fact(test_data)

    logging.info('Train set average fact score: {}'.format(float(train_scores.mean())))
    logging.info('Dev set average fact score: {}'.format(float(dev_scores.mean())))
    logging.info('Test set average fact score: {}'.format(float(test_scores.mean())))


def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path


def load_configs(config_path):
    with open(config_path) as f:
        logging.info('loading configuration file {}'.format(config_path))
        for line in f:
            if not '=' in line:
                continue
            arg_name, arg_value = line.strip().split('=')
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                logging.info('{} = {}'.format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == 'True':
                        setattr(args, arg_name, True)
                    elif arg_value == 'False':
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError('Unrecognized boolean value description: {}'.format(arg_value))
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError('Unrecognized attribute type: {}: {}'.format(arg_name, type(arg_value2)))
            else:
                raise ValueError('Unrecognized argument: {}'.format(arg_name))
    return args


def run_experiment(args):
    if args.test:
        if 'NELL' in args.data_dir:
            dataset = os.path.basename(args.data_dir)
            args.distmult_state_dict_path = data_utils.change_to_test_model_path(dataset, args.distmult_state_dict_path)
            args.complex_state_dict_path = data_utils.change_to_test_model_path(dataset, args.complex_state_dict_path)
            args.conve_state_dict_path = data_utils.change_to_test_model_path(dataset, args.conve_state_dict_path)
        args.data_dir += '.test'

    if args.process_data:

        # Process knowledge graph data

        process_data()
    else:
        with torch.set_grad_enabled(args.train or args.search_random_seed or args.grid_search):
            initialize_model_directory(args)
            lf = construct_model(args)
            lf.cuda()

            if args.train:
                train(lf)
            elif args.inference:
                inference(lf)
            elif args.eval_by_relation_type:
                inference(lf)
            elif args.eval_by_seen_queries:
                inference(lf)
            elif args.export_to_embedding_projector:
                export_to_embedding_projector(lf)
            elif args.export_reward_shaping_parameters:
                export_reward_shaping_parameters(lf)
            elif args.compute_fact_scores:
                compute_fact_scores(lf)
            elif args.export_fuzzy_facts:
                export_fuzzy_facts(lf)
            elif args.export_error_cases:
                export_error_cases(lf)


def set_logger(args):
    """
    Write logs to checkpoint and console
    """
    dataset = args.data_dir
    dataset = dataset.split('/')[-1]
    if args.train:
        log_name = dataset + '-' + args.model + '-' + str(args.rl_module) + \
                   '-num_rollout_steps' + str(args.num_rollout_steps) + \
                   '-num_rollouts' + str(args.num_rollouts) + \
                   '-gamma' + str(args.gamma) + '-beamsize_high' + str(args.beam_size_high) + \
                   '-beamsize-low' + str(args.beam_size_low) + '-beamsize' + str(args.beam_size) + \
                   '-ff_dropout_rate' + str(args.ff_dropout_rate) + \
                   '-r_dropout_rate' + str(args.relation_dropout_rate) + \
                   '-te_dropout_rate' + str(args.tailentity_dropout_rate) + \
                   '-train_bs' + str(args.train_batch_size) + \
                   '-dev_bs' + str(args.dev_batch_size) + \
                   '-add_tail_size' + str(args.additional_tailentity_size)
        log_name += '-addtailentity'

    elif args.inference:
        log_name = dataset + '-' + args.model + '-' + 'inference'
    else:
        log_name = dataset + '-' + args.model + '-' + 'analysis'

    log_file = os.path.join(args.log_dir, 'test/' + log_name + '.log')
    print("Log file name => {}".format(log_file))

    # logging module
    logger = logging.getLogger()
    # clear default handlers for avioding the mutiple output infos
    logger.handlers = []
    # logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    ## log to file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)

if __name__ == '__main__':
    set_logger(args)
    run_experiment(args)
