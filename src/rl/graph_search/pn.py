"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Graph Search Policy Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)
import numpy as np

import src.utils.ops as ops
from src.utils.ops import var_cuda, zeros_var_cuda
from src.utils.ops import int_var_cuda, var_cuda


class GraphSearchPolicy(nn.Module):
    def __init__(self, args, fn=None, fn_kg=None):
        super(GraphSearchPolicy, self).__init__()
        self.model = args.model
        self.relation_only = args.relation_only

        self.history_dim = args.history_dim
        self.history_num_layers = args.history_num_layers
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.leakyrelu_slope = 0.2

        if self.relation_only:
            self.action_dim = args.relation_dim
        else:
            self.action_dim = args.entity_dim + args.relation_dim
        self.ff_dropout_rate = args.ff_dropout_rate
        self.rnn_dropout_rate = args.rnn_dropout_rate
        self.action_dropout_rate = args.action_dropout_rate

        self.xavier_initialization = args.xavier_initialization

        self.relation_only_in_path = args.relation_only_in_path
        self.path = None
        self.relation_path = None

        self.rl_module = args.rl_module
        self.hrl_ff_dropout = args.hrl_ff_dropout

        self.data_dir = args.data_dir

        # Set policy network modules
        self.define_modules()
        self.initialize_modules()

        self.fn = fn
        self.fn_kg = fn_kg
        self.additional_tailentity_size = args.additional_tailentity_size

    def transit_high(self, e, obs, kg, use_relation_space_bucketing=True,
                     merge_aspace_batching_outcome=False):
        """
        Compute the next action distribution based on
            (a) the current node (entity) in KG and the query relation
            (b) action history representation
        :param e: agent location (node) at step t.
        :param obs: agent observation at step t.
            e_s: source node
            q: query relation
            e_t: target node
            last_step: If set, the agent is carrying out the last step.
            last_r: label of edge traversed in the previous step
            seen_nodes: notes seen on the paths
        :param kg: Knowledge graph environment.
        :param use_relation_space_bucketing: If set, group the action space of different nodes
            into buckets by their sizes.
        :param merge_aspace_batch_outcome: If set, merge the transition probability distribution
            generated of different action space bucket into a single batch.
        :return
            With aspace batching and without merging the outcomes:
                db_outcomes: (Dynamic Batch) (action_space, action_dist)
                    action_space: (Batch) padded possible action indices
                    action_dist: (Batch) distribution over actions.
                inv_offset: Indices to set the dynamic batching output back to the original order.
                entropy: (Batch) entropy of action distribution.
            Else:
                action_dist: (Batch) distribution over actions.
                entropy: (Batch) entropy of action distribution.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Representation of the current state (current node and other observations)
        Q = kg.get_relation_embeddings(q)
        # self.path[0]: 2 tuple
        H = self.path[-1][0][-1, :, :]
        E = kg.get_entity_embeddings(e)
        E_s = kg.get_entity_embeddings(e_s)
        if Q.size(0) < H.size(0):
            Q = ops.tile_along_beam(Q, beam_size=H.size(0) // Q.size(0), dim=0)
        relation_space = self.get_relation_space(e, obs, kg)  # return: (r_space, r_mask)

        if self.relation_only:
            X = torch.cat([H, Q], dim=-1)
        elif self.relation_only_in_path:
            X = torch.cat([E, H, E_s, Q], dim=-1)
        else:
            assert (E.size(0) == H.size(0) == Q.size(0))
            X = torch.cat([E, H, Q], dim=-1)

        # MLP
        # For high level policy ==> option / relation
        X = self.W1(X)
        X = F.relu(X)
        if self.hrl_ff_dropout:
            X = self.W1Dropout(X)
        X = self.W2(X)
        if self.hrl_ff_dropout:
            X = self.W2Dropout(X)

        # High level policy
        def policy_nn_fun_high(X, relation_space, use_max_logit = False):
            r_space, r_mask = relation_space
            R = kg.get_relation_embeddings(r_space)

            if use_max_logit:
                logit = torch.squeeze(R @ torch.unsqueeze(X, 2), 2)  # => size([2560, 200, 1])
                max_logit, _ = torch.max(logit, 1, keepdim=True)
                max_logit.detach_()

                option_dist = F.softmax(logit - max_logit - (1 - r_mask) * ops.HUGE_INT, dim=-1)

            else:
                option_dist = F.softmax(
                    torch.squeeze(R @ torch.unsqueeze(X, 2), 2) - (1 - r_mask) * ops.HUGE_INT, dim=-1)
            return option_dist, ops.entropy(option_dist)

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = ops.pad_and_cat(db_r_space, padding_value=kg.dummy_r)[
                inv_offset]  # size([mini_batch_size, max_action_dim])
            e_space = ops.pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
            action_mask = ops.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
            action_space = ((r_space, e_space), action_mask)
            return action_space

        relation_dist, relation_entropy = policy_nn_fun_high(X, relation_space)
        db_outcomes_high = [(relation_space, relation_dist)]
        inv_offset_high = None

        return db_outcomes_high, inv_offset_high, relation_entropy

    def transit_low(self, e, next_r, obs, kg, r_space,
                    use_action_space_bucketing=True, merge_aspace_batching_outcome=False):
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Representation of the current state (current node and other observations)
        Q = kg.get_relation_embeddings(q)
        # self.path[0]: 2 tuple
        H = self.path[-1][0][-1, :, :]
        if H.size(0) < Q.size(0):
            H = ops.tile_along_beam(H, Q.size(0) // H.size(0))
        elif H.size(0) > Q.size(0):
            k = H.size(0) // Q.size(0)
            H = torch.stack([H[k * i] for i in range(Q.size(0))])
        E = kg.get_entity_embeddings(e)
        R = kg.get_relation_embeddings(next_r)

        if self.relation_only:
            X = torch.cat([H, Q], dim=-1)
        elif self.relation_only_in_path:
            E_s = kg.get_entity_embeddings(e_s)
            X = torch.cat([E, H, E_s, Q], dim=-1)

        elif self.rl_module == 'hrl':
            # different from original paper, we also use relation embedding from high level policy selection
            E_s = kg.get_entity_embeddings(e_s)
            if R.size(0) > Q.size(0):
                H = ops.tile_along_beam(H, beam_size=R.size(0) // H.size(0), dim=0)
                Q = ops.tile_along_beam(Q, beam_size=R.size(0) // Q.size(0), dim=0)
            X = torch.cat([E, Q, R, H], dim=-1)

        # MLP
        # For high level policy ==> option / relation
        X = self.W3(X)
        X = F.relu(X)
        if self.hrl_ff_dropout:
            X = self.W3Dropout(X)
        X = self.W4(X)
        if self.hrl_ff_dropout:
            X = self.W4Dropout(X)

        # Low level policy
        def policy_nn_fun_low(X, tailentity_space):
            te_space, te_mask = tailentity_space
            E = kg.get_entity_embeddings(te_space)

            action_dist = F.softmax(
                torch.squeeze(E @ torch.unsqueeze(X, -1), -1) - (1 - te_mask) * ops.HUGE_INT, dim=-1)
            return action_dist, ops.entropy(action_dist)

        tail_anticipated = self.fn.forward(e, next_r, self.fn_kg)
        _, tail_selected = torch.topk(tail_anticipated, k=self.additional_tailentity_size, dim=-1)
        mask_selected = ops.var_cuda(torch.ones(tail_selected.size()))
        tailentity_dist, tailentity_entropy = policy_nn_fun_low(X, (tail_selected, mask_selected))
        tailentity_space = (tail_selected, mask_selected)
        db_outcomes_low = [(tailentity_space, tailentity_dist)]
        inv_offset_low = None
        return db_outcomes_low, inv_offset_low, tailentity_entropy

    def update_params(self, loss, step_size=0.5, first_order=False):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=not first_order)
        return parameters_to_vector(self.parameters()) - parameters_to_vector(grads) * step_size

    def initialize_path(self, init_action, kg):
        # [batch_size, action_dim]
        if self.relation_only_in_path:
            init_action_embedding = kg.get_relation_embeddings(init_action[0])
        else:
            # (r, e), recently traversed edge (relation) and target entity
            init_action_embedding = self.get_action_embedding(init_action, kg)

        init_action_embedding.unsqueeze_(1)
        init_h = zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim])
        init_c = zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim])
        self.path = [self.path_encoder(init_action_embedding, (init_h, init_c))[1]]  # get hidden state

    def update_path(self, action, kg, option_offset=None, action_offset=None):
        """
        Once an action was selected, update the action history.
        :param action (r, e): (Variable:batch) indices of the most recent action
            - r is the most recently traversed edge;
            - e is the destination entity.
        :param offset: (Variable:batch) if None, adjust path history with the given offset, used for search
        :param KG: Knowledge graph environment.
        """

        def offset_path_history(p, offset):
            if option_offset is not None:
                for i, x in enumerate(p):
                    if type(x) is tuple:
                        new_tuple = tuple([_x[:, offset, :] for _x in x])
                        p[i] = new_tuple
                    else:
                        p[i] = x[action_offset, :]
            else:
                for i, x in enumerate(p):
                    if type(x) is tuple:
                        new_tuple = tuple([_x[:, action_offset, :] for _x in x])
                        p[i] = new_tuple
                    else:
                        p[i] = x[action_offset, :]

        # update action history
        if self.relation_only_in_path:
            relation_embedding = kg.get_relation_embeddings(action[0])
        else:
            action_embedding = self.get_action_embedding(action, kg)
            relation_embedding = kg.get_relation_embeddings(action[0])

        if action_offset is not None:
            offset_path_history(self.path, action_offset)

        # path hidden update
        last_hidden = self.path[-1]
        if last_hidden[0].size(1) != action_embedding.size(0):
            k_tmp = action_embedding.size(0) // last_hidden[0].size(1)
            new_hidden_list = []
            for h in last_hidden:
                new_hidden_list.append(ops.tile_along_beam(h, k_tmp, dim=1))
            last_hidden = tuple(new_hidden_list)

        self.path.append(self.path_encoder(action_embedding.unsqueeze(1), last_hidden)[1])

    def get_relation_space(self, e, obs, kg):
        # for relation_space
        r_space, r_mask = kg.relation_space[0][e], kg.relation_space[1][e]

        relation_space = int_var_cuda(r_space), var_cuda(r_mask)
        return relation_space

    def get_tailentity_space(self, e, r, obs, kg, r_space):
        te_space = torch.zeros(e.size(0), kg.tailentity_space[0].size(-1))
        te_mask = torch.zeros(e.size(0), kg.tailentity_space[0].size(-1))

        for idx, (e_, r_) in enumerate(zip(e, r)):
            te_space[idx] = kg.tailentity_space[0][e_, r_]
            te_mask[idx] = kg.tailentity_space[1][e_, r_]

        te_space = ops.pad_and_cat(te_space, padding_value=0, padding_dim=0).view(e.size(0), -1)
        te_mask = ops.pad_and_cat(te_mask, padding_value=0, padding_dim=0).view(e.size(0), -1)

        tailentity_space = int_var_cuda(te_space), var_cuda(te_mask)
        return tailentity_space

    def validate_action_mask(self, action_mask):
        action_mask_min = action_mask.min()
        action_mask_max = action_mask.max()
        assert (action_mask_min == 0 or action_mask_min == 1)
        assert (action_mask_max == 0 or action_mask_max == 1)

    def get_action_embedding(self, action, kg):
        """
        Return (batch) action embedding which is the concatenation of the embeddings of
        the traversed edge and the target node.

        :param action (r, e):
            (Variable:batch) indices of the most recent action
                - r is the most recently traversed edge
                - e is the destination entity.
        :param kg: Knowledge graph environment.
        """
        r, e = action
        relation_embedding = kg.get_relation_embeddings(r)
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = kg.get_entity_embeddings(e)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def define_modules(self):
        if self.relation_only:
            input_dim = self.history_dim + self.relation_dim
        elif self.relation_only_in_path:
            input_dim = self.history_dim + self.entity_dim * 2 + self.relation_dim
        else:
            input_dim = self.history_dim + self.entity_dim + self.relation_dim
        if self.rl_module == 'original':
            self.W1 = nn.Linear(input_dim, self.action_dim)
            self.W2 = nn.Linear(self.action_dim, self.action_dim)
            self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
            self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)
        elif self.rl_module == 'hrl':
            input_dim_high = self.history_dim + self.entity_dim + self.relation_dim
            input_dim_low = self.history_dim + self.entity_dim + self.relation_dim * 2

            self.W1 = nn.Linear(input_dim_high, self.relation_dim)
            self.W2 = nn.Linear(self.relation_dim, self.relation_dim)
            self.W3 = nn.Linear(input_dim_low, self.entity_dim)
            self.W4 = nn.Linear(self.entity_dim, self.entity_dim)

            self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
            self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)
            self.W3Dropout = nn.Dropout(p=self.ff_dropout_rate)
            self.W4Dropout = nn.Dropout(p=self.ff_dropout_rate)

            self.leakyrelu = nn.LeakyReLU(self.leakyrelu_slope)

        if self.relation_only_in_path:
            self.path_encoder = nn.LSTM(input_size=self.relation_dim,
                                        hidden_size=self.history_dim,
                                        num_layers=self.history_num_layers,
                                        batch_first=True)
        else:
            self.path_encoder = nn.LSTM(input_size=self.action_dim,
                                        hidden_size=self.history_dim,
                                        num_layers=self.history_num_layers,
                                        batch_first=True)

    def initialize_modules(self):
        if self.xavier_initialization:
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            for name, param in self.path_encoder.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
