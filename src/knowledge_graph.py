"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Knowledge Graph Environment.
"""

import collections
import os
import pickle

import logging
import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)

from src.data_utils import load_index
from src.data_utils import NO_OP_ENTITY_ID, NO_OP_RELATION_ID
from src.data_utils import DUMMY_ENTITY_ID, DUMMY_RELATION_ID
from src.data_utils import START_RELATION_ID
import src.utils.ops as ops
from src.utils.ops import int_var_cuda, var_cuda
import numpy as np


class KnowledgeGraph(nn.Module):
    """
    The discrete knowledge graph is stored with an adjacency list.
    """

    def __init__(self, args):
        super(KnowledgeGraph, self).__init__()
        self.entity2id, self.id2entity = {}, {}
        self.relation2id, self.id2relation = {}, {}
        self.type2id, self.id2type = {}, {}
        self.entity2typeid = {}
        self.adj_list = None
        self.bandwidth = args.bandwidth
        self.args = args
        self.rl_module = args.rl_module

        self.action_space = None
        self.action_space_buckets = None
        self.unique_r_space = None

        self.relation_space = None
        self.tailentity_space = None

        self.train_subjects = None
        self.train_objects = None
        self.dev_subjects = None
        self.dev_objects = None
        self.all_subjects = None
        self.all_objects = None
        self.train_subject_vectors = None
        self.train_object_vectors = None
        self.dev_subject_vectors = None
        self.dev_object_vectors = None
        self.all_subject_vectors = None
        self.all_object_vectors = None

        self.max_degree_e = 0
        self.max_degree_er = 0

        logging.info('** Create {} knowledge graph **'.format(args.model))

        self.load_graph_data(args.data_dir)
        self.load_all_answers(args.data_dir)

        # Define NN Modules
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate
        self.num_graph_convolution_layers = args.num_graph_convolution_layers
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_img_embeddings = None
        self.relation_img_embeddings = None
        self.EDropout = None
        self.RDropout = None

        self.define_modules()
        self.initialize_modules()

    def update_params(self, loss, step_size=0.5, first_order=False):
        grads = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, self.parameters()),
            create_graph=not first_order)
        return parameters_to_vector(filter(lambda p: p.requires_grad, self.parameters())) - parameters_to_vector(grads) * step_size

    def load_graph_data(self, data_dir):
        # Load indices
        self.entity2id, self.id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
        logging.info('Sanity check: {} entities loaded'.format(len(self.entity2id)))
        self.type2id, self.id2type = load_index(os.path.join(data_dir, 'type2id.txt'))
        logging.info('Sanity check: {} types loaded'.format(len(self.type2id)))
        with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'rb') as f:
            self.entity2typeid = pickle.load(f)
        self.relation2id, self.id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
        logging.info('Sanity check: {} relations loaded'.format(len(self.relation2id)))

        # Load graph structures
        if self.args.model.startswith('point'):
            # Base graph structure used for training and test
            adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
            with open(adj_list_path, 'rb') as f:
                self.adj_list = pickle.load(f)
            if self.rl_module == 'original':
                self.vectorize_action_space(data_dir)
            elif self.rl_module == 'hrl':
                self.vectorize_relation_entity_space(data_dir)

    def vectorize_relation_entity_space(self, data_dir):
        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    try:
                        e, score = line.strip().split(':')
                        e_id = self.entity2id[e.strip()]
                        score = float(score)
                        pgrk_scores[e_id] = score
                    except:
                        continue
            return pgrk_scores

        page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))

        def get_relation_space(entity):
            relation_space = []
            if entity in self.adj_list:
                relation_space = list(self.adj_list[entity])
                if len(relation_space) + 1 >= self.bandwidth:
                    # base graph pruning
                    sorted_option_space = sorted(relation_space, key=lambda x: page_rank_scores[x[1]],
                                                 reverse=True)
                    relation_space = sorted_option_space[:self.bandwith]
            # relation_space.insert(0, NO_OP_RELATION_ID)
            return relation_space

        def get_tailentity_space(entity, relation):
            tail_entity_space = list(self.adj_list[entity][relation])
            # add entity itself here => NO STOP here
            tail_entity_space.insert(0, entity)
            return tail_entity_space

        def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size):
            bucket_size = len(unique_r_space_list)
            unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
            for i, u_r_s in enumerate(unique_r_space_list):
                for j, r in enumerate(u_r_s):
                    unique_r_space[i, j] = r
            return int_var_cuda(unique_r_space)

        def vectorize_relation_space(relation_space_list, relation_space_size):
            bucket_size = len(relation_space_list)
            r_space = torch.zeros(bucket_size, relation_space_size) + self.dummy_r
            r_mask = torch.zeros(bucket_size, relation_space_size)
            for i, relation_space in enumerate(relation_space_list):
                for j, r in enumerate(relation_space):
                    r_space[i, j] = r
                    r_mask[i, j] = 1
            return int_var_cuda(r_space), var_cuda(r_mask)

        def vectorize_tailentity_space(tailentity_space_list, tailentity_space_size):
            bucket_size = len(tailentity_space_list)
            assert (tailentity_space_size == self.max_degree_er + 1)
            te_space = torch.zeros(bucket_size, self.num_relations, tailentity_space_size) + self.dummy_e
            te_mask = torch.zeros(bucket_size, self.num_relations, tailentity_space_size)

            for i, r_tailentity in enumerate(tailentity_space_list):
                for r in r_tailentity.keys():
                    for k, te in enumerate(r_tailentity[r]):
                        te_space[i, r, k] = te
                        te_mask[i, r, k] = 1

            return int_var_cuda(te_space), var_cuda(te_mask)

        # sanity check
        num_facts = 0
        out_degrees = collections.defaultdict(int)
        for e1 in self.adj_list:
            for r in self.adj_list[e1]:
                num_facts += len(self.adj_list[e1][r])
                out_degrees[e1] += len(self.adj_list[e1][r])
                if self.max_degree_er < len(self.adj_list[e1][r]):
                    self.max_degree_er = len(self.adj_list[e1][r])

        self.max_degree_e = max([len(self.adj_list[i]) for i in self.adj_list.keys()])
        logging.info("Sanity check: maximum out degree of head entity (maximum relation number of entity): {}".
                     format(self.max_degree_e))
        logging.info("Sanity check: maximum out degree of head entity and relation (maximum relation number "
                     "of entity): {}".format(self.max_degree_er))
        logging.info("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
        logging.info('Sanity check: {} facts in knowledge graph'.format(num_facts))

        relation_space_list = []
        tailentity_space_list = []
        max_num_relations = 0
        max_num_tailentity = 0
        for e1 in range(self.num_entities):
            # get relation(option) space
            relation_space = get_relation_space(e1)
            relation_space_list.append(relation_space)
            t = len(relation_space)
            if len(relation_space) > max_num_relations:
                max_num_relations = len(relation_space)

            # get entity(action) space
            # r2e: dictionary, key: relation idx, value: list of entity idx with same head entity
            r2e = collections.defaultdict()
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    entity_space = get_tailentity_space(e1, r)
                    r2e[r] = entity_space
                    if len(entity_space) > max_num_tailentity:
                        max_num_tailentity = len(entity_space)
            tailentity_space_list.append(r2e)

        self.relation_space = vectorize_relation_space(relation_space_list, max_num_relations)
        self.tailentity_space = vectorize_tailentity_space(tailentity_space_list, max_num_tailentity)
        print("Finish vectorizing relation entity space")

    def vectorize_action_space(self, data_dir):
        """
        Pre-process and numericalize the knowledge graph structure.
        """

        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    e, score = line.strip().split(':')
                    e_id = self.entity2id[e.strip()]
                    score = float(score)
                    pgrk_scores[e_id] = score
            return pgrk_scores

        # Sanity check
        # compute num_fact and out_degree (list)
        num_facts = 0
        out_degrees = collections.defaultdict(int)
        for e1 in self.adj_list:
            for r in self.adj_list[e1]:
                num_facts += len(self.adj_list[e1][r])
                out_degrees[e1] += len(self.adj_list[e1][r])
        logging.info("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
        logging.info('Sanity check: {} facts in knowledge graph'.format(num_facts))

        # load page rank scores
        page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))

        def get_action_space(e1):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2))
                if len(action_space) + 1 >= self.bandwidth:
                    # Base graph pruning
                    sorted_action_space = \
                        sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    action_space = sorted_action_space[:self.bandwidth]
            action_space.insert(0, (NO_OP_RELATION_ID, e1))
            return action_space

        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)

            r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
            e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
            action_mask = torch.zeros(bucket_size, action_space_size)

            for i, action_space in enumerate(action_space_list):
                for j, (r, e) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    action_mask[i, j] = 1
            return (int_var_cuda(r_space), int_var_cuda(e_space)), var_cuda(action_mask)

        def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size, volatile):
            bucket_size = len(unique_r_space_list)
            unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
            for i, u_r_s in enumerate(unique_r_space_list):
                for j, r in enumerate(u_r_s):
                    unique_r_space[i, j] = r
            return int_var_cuda(unique_r_space)

        action_space_list = []
        max_num_actions = 0
        for e1 in range(self.num_entities):
            action_space = get_action_space(e1)
            action_space_list.append(action_space)
            if len(action_space) > max_num_actions:
                max_num_actions = len(action_space)
        logging.info('Vectorizing action spaces...')
        self.action_space = vectorize_action_space(action_space_list, max_num_actions)
        logging.info('Finish vectorizing action spaces')

        if self.args.model.startswith('rule'):
            unique_r_space_list = []
            max_num_unique_rs = 0
            for e1 in sorted(self.adj_list.keys()):
                unique_r_space = get_unique_r_space(e1)
                unique_r_space_list.append(unique_r_space)
                if len(unique_r_space) > max_num_unique_rs:
                    max_num_unique_rs = len(unique_r_space)
            self.unique_r_space = vectorize_unique_r_space(unique_r_space_list, max_num_unique_rs)

    def load_all_answers(self, data_dir, add_reversed_edges=False):
        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        # store subjects for all (rel, object) queries and
        # objects for all (subject, rel) queries
        train_subjects, train_objects = {}, {}
        dev_subjects, dev_objects = {}, {}
        all_subjects, all_objects = {}, {}
        # include dummy examples
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)
        for file_name in ['raw.kb', 'train.triples', 'dev.triples', 'test.triples']:
            if 'NELL' in self.args.data_dir and self.args.test and file_name == 'train.triples':
                continue
            with open(os.path.join(data_dir, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1, e2, r = self.triple2ids((e1, e2, r))
                    if file_name in ['raw.kb', 'train.triples']:
                        add_subject(e1, e2, r, train_subjects)
                        add_object(e1, e2, r, train_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), train_objects)
                    if file_name in ['raw.kb', 'train.triples', 'dev.triples']:
                        add_subject(e1, e2, r, dev_subjects)
                        add_object(e1, e2, r, dev_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                    add_subject(e1, e2, r, all_subjects)
                    add_object(e1, e2, r, all_objects)
                    if add_reversed_edges:
                        add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                        add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
        self.train_subjects = train_subjects
        self.train_objects = train_objects
        self.dev_subjects = dev_subjects
        self.dev_objects = dev_objects
        self.all_subjects = all_subjects
        self.all_objects = all_objects

        # change the answer set into a variable
        def answers_to_var(d_l):
            d_v = collections.defaultdict(collections.defaultdict)
            for x in d_l:
                for y in d_l[x]:
                    v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
                    d_v[x][y] = int_var_cuda(v)
            return d_v

        self.train_subject_vectors = answers_to_var(train_subjects)
        self.train_object_vectors = answers_to_var(train_objects)
        self.dev_subject_vectors = answers_to_var(dev_subjects)
        self.dev_object_vectors = answers_to_var(dev_objects)
        self.all_subject_vectors = answers_to_var(all_subjects)
        self.all_object_vectors = answers_to_var(all_objects)

    def load_fuzzy_facts(self):
        # extend current adjacency list with fuzzy facts
        dev_path = os.path.join(self.args.data_dir, 'dev.triples')
        test_path = os.path.join(self.args.data_dir, 'test.triples')
        with open(dev_path) as f:
            dev_triples = [l.strip() for l in f.readlines()]
        with open(test_path) as f:
            test_triples = [l.strip() for l in f.readlines()]
        removed_triples = set(dev_triples + test_triples)
        theta = 0.5
        fuzzy_fact_path = os.path.join(self.args.data_dir, 'train.fuzzy.triples')
        count = 0
        with open(fuzzy_fact_path) as f:
            for line in f:
                e1, e2, r, score = line.strip().split()
                score = float(score)
                if score < theta:
                    continue
                logging.info(line)
                if '{}\t{}\t{}'.format(e1, e2, r) in removed_triples:
                    continue
                e1_id = self.entity2id[e1]
                e2_id = self.entity2id[e2]
                r_id = self.relation2id[r]
                if not r_id in self.adj_list[e1_id]:
                    self.adj_list[e1_id][r_id] = set()
                if not e2_id in self.adj_list[e1_id][r_id]:
                    self.adj_list[e1_id][r_id].add(e2_id)
                    count += 1
                    if count > 0 and count % 1000 == 0:
                        logging.info('{} fuzzy facts added'.format(count))

        self.vectorize_action_space(self.args.data_dir)

    def get_inv_relation_id(self, r_id):
        return r_id + 1

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def get_entity_embeddings(self, e):
        return self.EDropout(self.entity_embeddings(e))

    def get_all_relation_embeddings(self):
        return self.RDropout(self.relation_embeddings.weight)

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_all_entity_img_embeddings(self):
        return self.EDropout(self.entity_img_embeddings.weight)

    def get_entity_img_embeddings(self, e):
        return self.EDropout(self.entity_img_embeddings(e))

    def get_relation_img_embeddings(self, r):
        return self.RDropout(self.relation_img_embeddings(r))

    def id2triples(self, triple):
        e1, e2, r = triple
        return self.id2entity[e1], self.id2entity[e2], self.id2relation[r]

    def triple2ids(self, triple):
        e1, e2, r = triple
        return self.entity2id[e1], self.entity2id[e2], self.relation2id[r]

    def define_modules(self):
        if not self.args.relation_only:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            if self.args.model == 'complex':
                self.entity_img_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        if self.args.model == 'complex':
            self.relation_img_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def initialize_modules(self):
        if not self.args.relation_only:
            nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def self_edge(self):
        return NO_OP_RELATION_ID

    @property
    def self_e(self):
        return NO_OP_ENTITY_ID

    @property
    def dummy_r(self):
        return DUMMY_RELATION_ID

    @property
    def dummy_e(self):
        return DUMMY_ENTITY_ID

    @property
    def dummy_start_r(self):
        return START_RELATION_ID
