"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Beam search on the graph.
"""

import torch

import src.utils.ops as ops
from src.utils.ops import unique_max, var_cuda, zeros_var_cuda, int_var_cuda, int_fill_var_cuda, var_to_numpy

def adjust_search_trace(search_trace, action_offset):
    for i, (r, e) in enumerate(search_trace):
        new_r = r[action_offset]
        new_e = e[action_offset]
        search_trace[i] = (new_r, new_e)

def beam_search(pn, e_s, q, e_t, kg, num_steps, beam_size, beam_size_high, beam_size_low, rl_module='original',
                use_action_space_bucketing=False, use_relation_space_bucketing=False, return_path_components=False):
    """
    Beam search from source.

    :param use_relation_space_bucketing:
    :param use_action_space_bucketing:
    :param beam_size_low:
    :param beam_size_high:
    :param rl_module:
    :param pn: Policy network.
    :param e_s: (Variable:batch) source entity indices.
    :param q: (Variable:batch) query relation indices.
    :param e_t: (Variable:batch) target entity indices.
    :param kg: Knowledge graph environment.
    :param num_steps: Number of search steps.
    :param beam_size: Beam size used in search.
    :param return_path_components: If set, return all path components at the end of search.
    """
    assert (num_steps >= 1)
    batch_size = len(e_s)

    # Initialization
    r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
    seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
    init_action = (r_s, e_s)
    # path encoder
    pn.initialize_path(init_action, kg)
    if kg.args.save_beam_search_paths:
        search_trace = [(r_s, e_s)]

    # Run beam search for num_steps
    # [batch_size*k], k=1
    log_option_prob = zeros_var_cuda(batch_size)
    log_tailentity_prob = zeros_var_cuda(batch_size * beam_size_high)
    log_action_prob = zeros_var_cuda([batch_size, 1])
    if return_path_components:
        log_option_probs = []
        log_tailentity_probs = []
        log_action_probs = []

    action = init_action
    q_init = q
    for t in range(num_steps):
        last_r, e = action

        if e.size(0) != q_init.size(0):
            k_temp = e.size(0) // q_init.size(0)
            # => [batch_size * k]
            q = ops.tile_along_beam(q_init, k_temp)
            e_s = ops.tile_along_beam(e_s, k_temp)
            e_t = ops.tile_along_beam(e_t, k_temp)
        assert (e.size(0) == q.size(0) == last_r.size(0))

        obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]

        # 1. sample relation based on high level policy
        db_outcomes_high, inv_offset_high, relation_entropy = pn.transit_high(
            e, obs, kg, use_relation_space_bucketing)
        r_space, r_mask = db_outcomes_high[0][0]
        r_dist = db_outcomes_high[0][1]

        log_option_prob = log_option_prob.view(-1, 1)
        log_option_dist = log_option_prob + ops.safe_log(r_dist)

        if t == num_steps:
            # top k answer unique
            next_r_list, next_r_idxs = [], []
            log_option_prob_list = []
            option_offset_list = []

            full_size_high = len(log_option_dist)
            assert (full_size_high % batch_size == 0)
            last_k_high = int(full_size_high / batch_size)

            r_space_size = r_space.size()[1]  # 33
            size_tmp = r_space.size(0) // beam_size_high
            beam_option_space_size = log_option_dist.size()[1]
            k_high = min(beam_size_high, beam_option_space_size)

            for i in range(log_option_dist.size(0)):
                log_option_dist_b = log_option_dist[i]
                r_space_b = r_space[i]  # 8448
                unique_r_space_b = var_cuda(
                    torch.unique(r_space_b.data.cpu()))
                unique_log_option_dist, unique_idx_high = unique_max(unique_r_space_b, r_space_b,
                                                                     log_option_dist_b)
                k_prime_high = min(len(unique_r_space_b), k_high)
                top_unique_log_option_dist, top_unique_high_idx2 = torch.topk(unique_log_option_dist,
                                                                              k_prime_high)
                top_unique_high_idx = unique_idx_high[top_unique_high_idx2]
                top_r_idx = top_unique_high_idx % r_space_size
                top_unique_beam_offset_high = top_unique_high_idx // r_space_size
                top_r = r_space_b[top_unique_high_idx]
                next_r_list.append(top_r.unsqueeze(0))
                next_r_idxs.append(top_r_idx.unsqueeze(0))
                log_option_prob_list.append(top_unique_log_option_dist.unsqueeze(0))
                top_unique_batch_offset_high = i * 1
                top_unique_option_offset = top_unique_batch_offset_high + top_unique_beam_offset_high
                option_offset_list.append(top_unique_option_offset.unsqueeze(0))
            next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
            log_option_prob = ops.pad_and_cat(log_option_prob_list, padding_value=-ops.HUGE_INT)
            option_offset = ops.pad_and_cat(option_offset_list, padding_value=-1).view(-1)

        else:
            # top k option
            full_size_high = len(log_option_dist)
            assert (full_size_high % batch_size == 0)
            last_k_high = int(full_size_high / batch_size)
            r_space_size = r_space.size()[1]  # 33
            beam_option_space_size = log_option_dist.size()[1]
            k_high = min(beam_size_high, beam_option_space_size)

            log_option_prob, next_r_idx = torch.topk(log_option_dist,
                                                     k_high)
            next_r = ops.batch_lookup(r_space, next_r_idx).view(-1)
            next_r_dist = ops.batch_lookup(r_dist, next_r_idx).view(-1)

            option_beam_offset = next_r_idx // r_space_size  # [batch_size, k_high]
            option_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k_high).unsqueeze(
                1)  # [batch_size, 1]
            if option_batch_offset.size(0) < option_beam_offset.size(0):
                option_batch_offset = ops.tile_along_beam(option_batch_offset, beam_size=option_beam_offset.size(
                    0) // option_batch_offset.size(0))
            option_offset = (option_batch_offset + option_beam_offset).view(-1)

        if next_r.size(0) != e.size(0):
            k = next_r.size(0) // e.size(0)
            e_ = ops.tile_along_beam(e, k)
            # => [batch_size * k]
            q_ = ops.tile_along_beam(q, k)
            e_s_ = ops.tile_along_beam(e_s, k)
            e_t_ = ops.tile_along_beam(e_t, k)

            obs = [e_s_, q_, e_t_, t == (num_steps - 1), last_r, seen_nodes]
        else:
            e_ = e

        # 2. sample tailentity based on low level policy
        db_outcomes_low, inv_offset_low, tailentity_entropy = \
            pn.transit_low(e_, next_r, obs, kg, r_space, use_action_space_bucketing)
        te_space, te_mask = db_outcomes_low[0][0]
        te_dist = db_outcomes_low[0][1]

        if t == 0:
            log_tailentity_prob = zeros_var_cuda(te_dist.size(0))

        log_tailentity_prob = log_tailentity_prob.view(-1, 1)
        if log_tailentity_prob.size(0) < te_dist.size(0):
            log_tailentity_prob = ops.tile_along_beam(log_tailentity_prob,
                                                      beam_size=te_dist.size(0) // log_tailentity_prob.size(0))

        log_tailentity_dist = log_tailentity_prob + ops.safe_log(te_dist)

        if t == num_steps:
            # top k answer unique
            next_te_list, next_te_idxs = [], []
            log_tailentity_prob_list = []
            action_offset_list = []

            full_size_low = len(log_tailentity_dist)
            assert (full_size_low % batch_size == 0)
            last_k_low = int(full_size_low / batch_size)

            action_space_size = te_space.size()[1]  # 116

            beam_action_space_size = log_tailentity_dist.size()[1]  # 116?
            assert (beam_action_space_size % action_space_size == 0)

            k_low = min(beam_size_low, beam_action_space_size)

            for i in range(log_tailentity_dist.size(0)):
                log_tailentity_dist_b = log_tailentity_dist[i]
                te_space_b = te_space[i]
                unique_te_space_b = var_cuda(torch.unique(te_space_b.data.cpu()))
                unique_log_tailentity_dist, unique_idx_low = unique_max(unique_te_space_b, te_space_b,
                                                                        log_tailentity_dist_b)

                k_prime = min(len(unique_te_space_b), k_low)
                top_unique_log_tailentity_dist, top_unique_low_idx2 = torch.topk(unique_log_tailentity_dist,
                                                                                 k_prime)
                top_unique_low_idx = unique_idx_low[top_unique_low_idx2]
                top_te_idx = top_unique_low_idx % action_space_size
                top_unique_beam_offset_low = top_unique_low_idx // action_space_size

                top_te = te_space_b[top_unique_low_idx]
                next_te_list.append(top_te.unsqueeze(0))
                next_te_idxs.append(top_te_idx.unsqueeze(0))
                log_tailentity_prob_list.append(top_unique_log_tailentity_dist.unsqueeze(0))

                top_unique_batch_offset_low = i // beam_size_high
                top_unique_action_offset = top_unique_batch_offset_low + top_unique_beam_offset_low
                action_offset_list.append(top_unique_action_offset.unsqueeze(0))
            next_te = ops.pad_and_cat(next_te_list, padding_value=kg.dummy_e).view(-1)

            log_tailentity_prob = ops.pad_and_cat(log_tailentity_prob_list, padding_value=-ops.HUGE_INT)
            action_offset = ops.pad_and_cat(action_offset_list, padding_value=-1).view(-1)

        else:
            # top k action
            full_size_low = len(log_tailentity_dist)
            assert (full_size_low % batch_size == 0)
            last_k_low = int(full_size_low / batch_size)

            action_space_size = te_space.size()[1]
            beam_action_space_size = log_tailentity_dist.size()[1]
            assert (beam_action_space_size % action_space_size == 0)

            k_low = min(beam_size_low, beam_action_space_size)

            log_tailentity_prob, next_te_idx = torch.topk(log_tailentity_dist,
                                                          k_low)

            next_te = ops.batch_lookup(te_space, next_te_idx).view(-1)
            next_te_dist = ops.batch_lookup(te_dist, next_te_idx).view(-1)

            action_beam_offset = next_te_idx // action_space_size
            action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k_high)
            if action_batch_offset.size(0) < action_beam_offset.size(0):
                action_batch_offset = ops.tile_along_beam(action_batch_offset, beam_size=action_beam_offset.size(
                    0) // action_batch_offset.size(0))
            action_batch_offset = action_batch_offset.view(-1).unsqueeze(1)
            action_offset = (action_batch_offset + action_beam_offset).view(-1)

        log_option_prob, log_tailentity_prob = log_option_prob.view(-1), log_tailentity_prob.view(-1)
        if log_option_prob.size(0) < log_tailentity_prob.size(0):
            log_option_prob = ops.tile_along_beam(log_option_prob,
                                                  beam_size=log_tailentity_prob.size(0) // log_option_prob.size(0))
        elif log_option_prob.size(0) > log_tailentity_prob.size(0):
            log_tailentity_prob = ops.tile_along_beam(log_tailentity_prob,
                                                      beam_size=log_option_prob.size(0) // log_tailentity_prob.size(0))

        if next_te.size(0) > next_r.size(0):
            next_r = ops.tile_along_beam(next_r, beam_size=next_te.size(0) // next_r.size(0))

        log_action_prob = (log_option_prob + log_tailentity_prob).view(batch_size, -1)

        next_te_list, next_r_list = [], []
        log_option_prob_list, log_tailentity_prob_list, log_action_prob_list = [], [], []
        option_offset_list, action_offset_list = [], []
        next_te = next_te.view(batch_size, -1)
        next_r = next_r.view(batch_size, -1)
        log_option_prob = log_option_prob.view(batch_size, -1)
        log_tailentity_prob = log_tailentity_prob.view(batch_size, -1)

        if option_offset.size(0) < action_offset.size(0):
            option_offset = ops.tile_along_beam(option_offset,
                                                beam_size=action_offset.size(0) // option_offset.size(0))
        option_offset = option_offset.view(batch_size, -1)
        action_offset = action_offset.view(batch_size, -1)

        # option1: first unqiue max, then top k
        # last step
        if t == num_steps - 1:
            for i in range(batch_size):
                log_action_prob_b = log_action_prob[i]
                next_te_b = next_te[i]
                next_r_b = next_r[i]
                log_option_prob_b = log_option_prob[i]
                log_tailentity_prob_b = log_tailentity_prob[i]
                option_offset_b = option_offset[i]
                action_offset_b = action_offset[i]

                unique_te_b = var_cuda(torch.unique(next_te_b.data.cpu()))
                unique_log_action_dist, unique_idx = unique_max(unique_te_b, next_te_b, log_action_prob_b)
                k_prime = min(len(unique_te_b), beam_size)
                top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
                top_unique_idx = unique_idx[top_unique_idx2]
                next_te_list.append(next_te_b[top_unique_idx])
                next_r_list.append(next_r_b[top_unique_idx])
                log_action_prob_list.append(log_action_prob_b[top_unique_idx])
                log_option_prob_list.append(log_option_prob_b[top_unique_idx])
                log_tailentity_prob_list.append(log_tailentity_prob_b[top_unique_idx])
                option_offset_list.append(option_offset_b[top_unique_idx])
                action_offset_list.append(action_offset_b[top_unique_idx])

            next_te = ops.pad_and_cat(next_te_list, padding_value=kg.dummy_e, padding_dim=0).view(-1)
            next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r, padding_dim=0).view(-1)
            log_option_prob = ops.pad_and_cat(log_option_prob_list, padding_value=-ops.HUGE_INT, padding_dim=0)
            log_tailentity_prob = ops.pad_and_cat(log_tailentity_prob_list, padding_value=-ops.HUGE_INT,
                                                  padding_dim=0)
            log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=-ops.HUGE_INT, padding_dim=0)
            option_offset = ops.pad_and_cat(option_offset_list, padding_value=-1, padding_dim=0).view(-1)
            action_offset = ops.pad_and_cat(action_offset_list, padding_value=-1, padding_dim=0).view(-1)

        # option2: select topk directly, without unique max
        else:
            # (bs, k1 * k2) ==topk==> (bs, k)
            k = min(log_action_prob.size(-1), beam_size)
            log_action_prob, action_idx = torch.topk(log_action_prob, k)
            next_r = next_r.view(batch_size, -1)
            next_r = next_r[torch.arange(next_r.size(0))[:, None], action_idx].view(-1)
            next_te = next_te.view(batch_size, -1)
            next_te = next_te[torch.arange(next_te.size(0))[:, None], action_idx].view(-1)
            log_option_prob = log_option_prob.view(batch_size, -1)
            log_option_prob = log_option_prob[torch.arange(log_option_prob.size(0))[:, None], action_idx].view(-1)
            log_tailentity_prob = log_tailentity_prob.view(batch_size, -1)
            log_tailentity_prob = log_tailentity_prob[
                torch.arange(log_tailentity_prob.size(0))[:, None], action_idx].view(-1)

            option_offset = option_offset.view(batch_size, -1)
            option_offset = option_offset[torch.arange(option_offset.size(0))[:, None], action_idx].view(-1)
            action_offset = action_offset.view(batch_size, -1)
            action_offset = action_offset[torch.arange(action_offset.size(0))[:, None], action_idx].view(-1)

        if return_path_components:
            ops.rearrange_vector_list(log_action_probs, action_offset)
            log_action_probs.append(log_action_prob)

        action = (next_r, next_te)
        pn.update_path(action, kg, option_offset, action_offset)

        seen_nodes = torch.cat([seen_nodes[action_offset], action[1].unsqueeze(1)], dim=1)

        if kg.args.save_beam_search_paths:
            adjust_search_trace(search_trace, action_offset)
            search_trace.append(action)

    output_beam_size = int(action[0].size()[0] / batch_size)
    # [batch_size*beam_size] => [batch_size, beam_size]
    beam_search_output = dict()
    beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
    beam_search_output['pred_e2_scores'] = log_action_prob.view(batch_size, -1)
    if kg.args.save_beam_search_paths:
        beam_search_output['search_traces'] = search_trace

    if return_path_components:
        path_width = 10
        path_components_list = []
        for i in range(batch_size):
            p_c = []
            for k, log_action_prob in enumerate(log_action_probs):
                top_k_edge_labels = []
                for j in range(min(output_beam_size, path_width)):
                    ind = i * output_beam_size + j
                    r = kg.id2relation[int(search_trace[k + 1][0][ind])]
                    e = kg.id2entity[int(search_trace[k + 1][1][ind])]
                    if r.endswith('_inv'):
                        edge_label = ' <-{}- {} {}'.format(r[:-4], e, float(log_action_probs[k][ind]))
                    else:
                        edge_label = ' -{}-> {} {}'.format(r, e, float(log_action_probs[k][ind]))
                    top_k_edge_labels.append(edge_label)
                top_k_action_prob = log_action_prob[:path_width]
                e_name = kg.id2entity[int(search_trace[1][0][i * output_beam_size])] if k == 0 else ''
                p_c.append((e_name, top_k_edge_labels, var_to_numpy(top_k_action_prob)))
            path_components_list.append(p_c)
        beam_search_output['path_components_list'] = path_components_list

    return beam_search_output
