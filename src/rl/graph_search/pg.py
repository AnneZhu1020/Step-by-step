"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Policy gradient (REINFORCE algorithm) training and inference.
"""

import torch
import logging

from src.learn_framework import LFramework
import src.rl.graph_search.beam_search as search

import src.utils.ops as ops
from src.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda


class PolicyGradient(LFramework):
    def __init__(self, args, kg, pn):
        super(PolicyGradient, self).__init__(args, kg, pn)

        # Training hyperparameters
        self.relation_only = args.relation_only
        self.use_action_space_bucketing = args.use_action_space_bucketing
        logging.info("use_action_space_bucketing: {}".format(self.use_action_space_bucketing))
        self.use_relation_space_bucketing = args.use_relation_space_bucketing
        logging.info("use_relation_space_bucketing: ".format(self.use_relation_space_bucketing))
        self.num_rollouts = args.num_rollouts  # default: 20
        self.num_rollout_steps = args.num_rollout_steps  # default: 3
        self.baseline = args.baseline
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate
        self.action_dropout_anneal_factor = args.action_dropout_anneal_factor
        self.action_dropout_anneal_interval = args.action_dropout_anneal_interval

        self.relation_dropout_rate = args.relation_dropout_rate
        self.tailentity_dropout_rate = args.tailentity_dropout_rate

        # Inference hyperparameters
        self.beam_size = args.beam_size
        self.beam_size_high = args.beam_size_high
        self.beam_size_low = args.beam_size_low

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0
        self.fast_lr = 0.2

    def reward_fun(self, e1, r, e2, pred_e2):
        return (pred_e2 == e2).float()

    def loss_hrl(self, mini_batch):

        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r

        e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)

        output = self.rollout_hrl(e1, r, e2, num_steps=self.num_rollout_steps)

        # output: dict(7)
        # pred_e2, log_option_probs, option_entropy, log_action_probs, action_entropy, path_trace, path_components
        pred_e2 = output['pred_e2']
        log_option_probs = output['log_option_probs']
        option_entropy = output['option_entropy']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']
        path_trace = output['path_trace']
        path_trace_list = [list(x) for x in output['path_trace']]  # (r, e) => [r, e], turn set into list, 为了取值方便
        relation_selected = [x[0] for x in path_trace_list]  # get all relations
        tailentity_selected = [x[1] for x in path_trace_list]  # get all tailentities
        last_relation = relation_selected[-1]

        reward_high, reward_low = self.reward_fun(e1, r, e2, pred_e2)

        if self.baseline != 'n/a':  # default: n/a
            reward_high = stablize_reward(reward_high)

        cum_discounted_rewards_high = [0] * self.num_rollout_steps
        cum_discounted_rewards_high[-1] = reward_high
        cum_discounted_rewards_low = [0] * self.num_rollout_steps
        cum_discounted_rewards_low[-1] = reward_low

        R_low, R_high = 0, 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R_high = self.gamma * R_high + cum_discounted_rewards_high[i]
            cum_discounted_rewards_high[i] = R_high
            R_low = self.gamma * R_low + cum_discounted_rewards_low[i]
            cum_discounted_rewards_low[i] = R_low

        # compute policy gradient loss
        pg_loss_high, pt_loss_high = 0, 0
        pg_loss_low, pt_loss_low = 0, 0

        for i in range(self.num_rollout_steps):
            log_option_prob = log_option_probs[i]
            pg_loss_high += -cum_discounted_rewards_high[i] * log_option_prob
            pt_loss_high += -cum_discounted_rewards_high[i] * torch.exp(log_option_prob)
            log_action_prob = log_action_probs[i]
            pg_loss_low += -cum_discounted_rewards_low[i] * log_action_prob
            pt_loss_low += -cum_discounted_rewards_low[i] * torch.exp(log_action_prob)

        # high level entropy regularization
        entropy_high = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss_high = (pg_loss_high - entropy_high * self.beta).mean()
        pt_loss_high = (pt_loss_high - entropy_high * self.beta).mean()

        # low level entropy regularization
        entropy_low = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss_low = (pg_loss_low - entropy_low * self.beta).mean()
        pt_loss_low = (pt_loss_low - entropy_low * self.beta).mean()

        # dict(8+1)
        loss_dict = {'model_loss_high': pg_loss_high,
                     'print_loss_high': float(pt_loss_high),
                     'reward_high': reward_high,
                     'entropy_high': float(entropy_high.mean()),
                     'model_loss_low': pg_loss_low,
                     'print_loss_low': float(pt_loss_low),
                     'reward_low': reward_low,
                     'entropy_low': float(entropy_low.mean())}

        if self.run_analysis:
            fn = torch.zeros(reward_low.size())
            for i in range(len(reward_low)):
                if not reward_low[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        return loss_dict

    def sample_relation(self, db_outcomes, inv_offset=None):
        """
        sample an option(relation) based on current high level policy
        :param num_samples: select num_samples relation
        :param db_outcomes: [((r_space, r_mask), relation_dist)]:
                r_space
                r_mask
                relation_dist
        :param inv_offset: Indexes for restoring original order in a batch.
        :return: next_relation
        :return: relation_prob: probability of the sampled relation
        """

        def apply_relation_dropout_mask(r_dist, r_mask):
            if self.relation_dropout_rate > 0:
                rand = torch.rand(r_dist.size())
                relation_keep_mask = var_cuda(rand > self.relation_dropout_rate).float()
                sample_relation_dist = r_dist * relation_keep_mask + ops.EPSILON * (1 - relation_keep_mask) * r_mask
                return sample_relation_dist

            else:
                return r_dist

        def sample(r_space, r_mask, r_dist):
            sample_relation_dist = apply_relation_dropout_mask(r_dist, r_mask)
            next_r_idx = torch.multinomial(sample_relation_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, next_r_idx)
            r_prob = ops.batch_lookup(r_dist, next_r_idx)
            return next_r, r_prob

        r_space, r_mask = db_outcomes[0][0]  # size([2560, 33])
        r_dist = db_outcomes[0][1]
        # TODO HERE
        if inv_offset is not None:
            pass
        else:
            next_r, next_r_prob = sample(r_space, r_mask, r_dist)

        return next_r, next_r_prob, r_space, r_mask, r_dist

    def sample_tailentity(self, db_outcomes, next_r, r_prob, tailentity_entropy, inv_offset=None):
        """
        sample an action(tailentity) based on current low level policy
        :param db_outcomes: [((te_space, te_mask), tailentity_dist)]:
                te_space: size([2560, 33, 116])
                te_mask: size([2560, 33, 116])
                tailentity_dist
        :param inv_offset: Indexes for restoring original order in a batch.
        :return: next_tailentity
        :return: tailentity_prob: probability of the sampled relation
        """

        def apply_tailentity_dropout_mask(te_dist, te_mask):
            if self.tailentity_dropout_rate > 0:
                rand = torch.rand(te_dist.size())
                tailentity_keep_mask = var_cuda(rand > self.tailentity_dropout_rate).float()
                sample_tail_dist = te_dist * tailentity_keep_mask + ops.EPSILON * (1 - tailentity_keep_mask) * te_mask
                return sample_tail_dist
            else:
                return te_dist

        def sample(te_space, te_mask, te_dist, next_r, r_prob, tailentity_entropy):
            sample_tailentity_dist = apply_tailentity_dropout_mask(te_dist, te_mask)  # size([2560*8, 116])

            idx = torch.multinomial(sample_tailentity_dist, 1, replacement=True)
            next_te = ops.batch_lookup(te_space, idx)
            te_prob = ops.batch_lookup(te_dist, idx)
            return next_te, te_prob, next_r, r_prob, tailentity_entropy

        te_space, te_mask = db_outcomes[0][0]
        te_dist = db_outcomes[0][1]

        # TODO HERE
        if inv_offset is not None:
            pass
        else:
            next_te, next_te_prob, next_r, r_prob, tailentity_entropy = sample(te_space, te_mask, te_dist, next_r,
                                                                               r_prob, tailentity_entropy)

        return next_r.view(-1), r_prob, next_te.view(-1), next_te_prob, tailentity_entropy, te_space, te_mask, te_dist

    def rollout_hrl(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps, default 3
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)  # dummy_start_r => 1

        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)  # dummy_e => 0
        path_components = []

        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg)

        # high level
        log_option_probs = []
        option_entropy = []

        # low level
        log_action_probs = []
        action_entropy = []

        # every step, run both high level policy and low level policy => transit function
        for t in range(num_steps):  # num_steps: 3
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
            # 1) high level option transit

            db_outcomes_high, inv_offset_high, relation_entropy = pn.transit_high(
                e, obs, kg, use_relation_space_bucketing=self.use_relation_space_bucketing)
            next_r, r_prob, r_space, r_mask, r_dist = self.sample_relation(db_outcomes_high, inv_offset=inv_offset_high)
            log_option_probs.append(ops.safe_log(r_prob))
            option_entropy.append(relation_entropy)

            # 2) low level action transits
            r_space = db_outcomes_high[0][0][0]

            neighbor_embeddings_entity_list = []

            db_outcomes_low, inv_offset_low, tailentity_entropy = \
                pn.transit_low(e, next_r, obs, kg, r_space, use_action_space_bucketing=self.use_action_space_bucketing)
            next_r, r_prob, next_te, next_te_prob, tailentity_entropy, te_space, te_mask, te_dist = \
                self.sample_tailentity(db_outcomes_low, next_r, r_prob, tailentity_entropy, inv_offset_low)

            log_action_probs.append(ops.safe_log(next_te_prob))
            action_entropy.append(tailentity_entropy)

            # 3) update path and seen_nodes with selected next_r, next_te
            pn.update_path((next_r, next_te), kg)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append((next_r, next_te))

        pred_e2 = path_trace[-1][1]
        self.record_path_trace(path_trace)

        return {
            'pred_e2': pred_e2,
            'log_option_probs': log_option_probs,
            'option_entropy': option_entropy,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components
        }

    def predict(self, mini_batch, verbose=False):
        kg, pn = self.kg, self.mdl
        e1, e2, r = self.format_batch(mini_batch)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size, self.beam_size_high,
            self.beam_size_low, self.rl_module, self.use_action_space_bucketing, self.use_relation_space_bucketing)

        pred_e2s = beam_search_output[
            'pred_e2s']
        pred_e2_scores = beam_search_output[
            'pred_e2_scores']
        if verbose:
            # print inference paths
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            for i in range(len(e1)):
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                    logging.info('beam {}: score = {} \n<PATH> {}'.format(
                        j, float(pred_e2_scores[i][j]), ops.format_path(search_trace, kg)))
        with torch.no_grad():
            pred_scores = zeros_var_cuda(
                [len(e1), kg.num_entities])  # size([self.mini_batch_size, 137]) e.g., size([32, 137])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores

    def record_path_trace(self, path_trace):
        path_length = len(path_trace)
        flattened_path_trace = [x for t in path_trace for x in t]
        path_trace_mat = torch.cat(flattened_path_trace).reshape(-1, path_length)
        path_trace_mat = path_trace_mat.data.cpu().numpy()

        for i in range(path_trace_mat.shape[0]):
            path_recorder = self.path_types
            for j in range(path_trace_mat.shape[1]):
                e = path_trace_mat[i, j]
                if not e in path_recorder:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] = 1
                        self.num_path_types += 1
                    else:
                        path_recorder[e] = {}
                else:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] += 1
                path_recorder = path_recorder[e]