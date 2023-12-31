# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import ctypes
import numpy as np
import torch
from cpp_sampler import libsampler
from cpp_sampler import sampler_clib
from common.util import name_query_dict

from collections import defaultdict
from tqdm import tqdm


def is_all_relation(query_structure):
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            return False
    return True


def has_negation(query_structure):
    for ele in query_structure[-1]:
        if ele == 'n':
            return True
    return False


def build_query_tree(query_structure, fn_qt_create):
    if is_all_relation(query_structure):
        assert len(query_structure) == 2
        if query_structure[0] == 'e':
            prev_node = fn_qt_create(libsampler.entity)
        else:
            prev_node = build_query_tree(query_structure[0], fn_qt_create)
        for i, c in enumerate(query_structure[-1]):
            if c == 'r':
                cur_op = libsampler.relation
            else:
                assert c == 'n'
                cur_op = libsampler.negation
            cur_root = fn_qt_create(libsampler.entity_set)
            cur_root.add_child(cur_op, prev_node)
            prev_node = cur_root
        return cur_root
    else:
        last_qt = query_structure[-1]
        node_type = libsampler.intersect
        if len(last_qt) == 1 and last_qt[0] == 'u':
            node_type = libsampler.union
            query_structure = query_structure[:-1]
        sub_root = fn_qt_create(node_type)
        for c in query_structure:
            ch_node = build_query_tree(c, fn_qt_create)
            sub_root.add_child(libsampler.no_op, ch_node)
        return sub_root


class OnlineSampler(object):
    """
            初始化在线采样器对象。

            参数：
            - kg: 知识图对象，表示知识图的数据结构。
            - query_names: 查询名称列表，表示要执行的查询的名称列表。
            - negative_sample_size: 负样本的数量。
            - sample_mode: 采样模式，包括多个参数，例如关系带宽、最大保留数量等。
            - normalized_structure_prob: 查询结构的概率分布。
            - sampler_type: 采样类型，可以是'naive'、'sqrt'、'nosearch'等。
            - share_negative: 是否共享负样本。
            - same_in_batch: 是否在同一批次中执行相同的查询。
            - weighted_answer_sampling: 是否对答案采样进行加权。
            - weighted_negative_sampling: 是否对负样本采样进行加权。
            - nprefetch: 预取批次的数量。
            - num_threads: 采样线程的数量。
            """
    def __init__(self, kg, query_names, negative_sample_size, 
                 sample_mode, normalized_structure_prob, sampler_type='naive', 
                 share_negative=False, same_in_batch=False,
                 weighted_answer_sampling=False, weighted_negative_sampling=False,
                 nprefetch=10, num_threads=8):
        self.kg = kg
        kg_dtype = kg.dtype
        fn_qt_create = libsampler.create_qt32 if kg_dtype == 'uint32' else libsampler.create_qt64
        query_structures = [name_query_dict[task] for task in query_names]
        self.query_structures = query_structures
        self.normalized_structure_prob = normalized_structure_prob
        assert len(normalized_structure_prob) == len(query_structures)
        self.negative_sample_size = negative_sample_size
        self.share_negative = share_negative
        self.same_in_batch = same_in_batch
        self.nprefetch = nprefetch
        if len(sample_mode) == 5:
            self.rel_bandwidth, self.max_to_keep, self.weighted_style, self.structure_weighted_style, self.max_n_partial_answers = sample_mode
            self.weighted_ans_sample = False
            self.weighted_neg_sample = False
        else:
            self.rel_bandwidth, self.max_to_keep, self.weighted_style, self.structure_weighted_style, self.max_n_partial_answers, self.weighted_ans_sample, self.weighted_neg_sample = sample_mode
        if self.rel_bandwidth <= 0:
            self.rel_bandwidth = kg.num_ent
        if self.max_to_keep <= 0:
            self.max_to_keep = kg.num_ent
        if self.max_n_partial_answers <= 0:
            self.max_n_partial_answers = kg.num_ent
        if self.structure_weighted_style == 'wstruct':
            assert self.normalized_structure_prob is not None

        list_qt = []
        list_qt_nargs = []
        for qs in query_structures:
            if qs[0] == '<':  # inverse query
                assert is_all_relation(qs[1]) and not has_negation(qs[1])
                qt = build_query_tree(qs[1], fn_qt_create)
                qt.is_inverse = True
            else:
                qt = build_query_tree(qs, fn_qt_create)
            list_qt.append(qt)
            list_qt_nargs.append(qt.get_num_args())
        self.list_qt = list_qt
        self.list_qt_nargs = list_qt_nargs
        self.max_num_args = max(list_qt_nargs)
        no_search_list = []
        
        if sampler_type == 'naive':
            sampler_cls = sampler_clib.naive_sampler(kg_dtype)
        elif sampler_type.startswith('sqrt'):
            sampler_cls = sampler_clib.rejection_sampler(kg_dtype)
            if '-' in sampler_type:
                no_search_list = [int(x) for x in sampler_type.split('-')[1].split('.')]
        elif sampler_type == 'nosearch':
            sampler_cls = sampler_clib.no_search_sampler(kg_dtype)
        elif sampler_type == 'edge':
            sampler_cls = sampler_clib.edge_sampler(kg_dtype)
            list_qt = query_names
        else:
            raise ValueError("Unknown sampler %s" % sampler_type)
        self.sampler_type = sampler_type
        self.sampler = sampler_cls(kg, list_qt, normalized_structure_prob, self.share_negative, self.same_in_batch,
                                    self.weighted_ans_sample, self.weighted_neg_sample,
                                    negative_sample_size, self.rel_bandwidth, self.max_to_keep, self.max_n_partial_answers, num_threads, no_search_list)

    def print_queries(self):
        """
                打印查询结构信息，用于调试和查看查询的结构。
        """
        self.sampler.print_queries()

    def sample_entities(self, weighted, num):
        """
                采样实体。

                参数：
                - weighted: 是否带权重进行采样。
                - num: 要采样的实体数量。

                返回：
                - 采样的实体列表。
        """
        entities = torch.LongTensor(num)
        self.sampler.sample_batch_entities(weighted, num, entities.numpy())
        return entities

    def set_seed(self, seed):
        self.sampler.set_seed(seed)

    def batch_generator(self, batch_size):
        self.sampler.prefetch(batch_size, self.nprefetch)
        uniform_weigths = torch.ones(batch_size)
        list_buffer = []
        for i in range(2):
            t_pos_ans = torch.LongTensor(batch_size) #(512)
            if self.share_negative:
                t_neg_ans = torch.LongTensor(512, self.negative_sample_size) #(1,128)
                t_is_neg_mat = torch.FloatTensor(batch_size, self.negative_sample_size)
                # t_neg_ans = torch.LongTensor(batch_size, self.negative_sample_size)
                # t_is_neg_mat = torch.FloatTensor(1, 2)
            else:
                t_neg_ans = torch.LongTensor(batch_size, self.negative_sample_size)#(512,128)
                t_is_neg_mat = torch.FloatTensor(1, 2) #(1,2)
            t_weights = torch.FloatTensor(batch_size)
            t_arg_buffer = torch.LongTensor(batch_size, self.max_num_args)
            list_buffer.append((t_pos_ans, t_neg_ans, t_is_neg_mat, t_weights, t_arg_buffer))

        buf_idx = 0
        pos_ans, neg_ans, is_neg_mat, weights, arg_buffer = list_buffer[buf_idx]
        q_type = self.sampler.next_batch(pos_ans.numpy(), neg_ans.numpy(), weights.numpy(), is_neg_mat.numpy(),
                                            arg_buffer.numpy())
        while True:            
            next_buf_idx = 1 - buf_idx
            next_pos_ans, next_neg_ans, next_is_neg_mat, next_weights, next_arg_buffer = list_buffer[next_buf_idx]
            next_q_type = self.sampler.next_batch(next_pos_ans.numpy(), next_neg_ans.numpy(), 
                                                  next_weights.numpy(), next_is_neg_mat.numpy(),
                                                  next_arg_buffer.numpy())            
            if self.weighted_style == 'u':
                weights = uniform_weigths
            pos_ans, neg_ans, is_neg_mat, weights, arg_buffer = list_buffer[buf_idx]                
            q_args = arg_buffer[:, :self.list_qt_nargs[q_type]]
            q_structs = [self.query_structures[q_type]] * batch_size
            if self.sampler_type == 'edge':
                is_neg_mat = None
            if self.share_negative:
                yield pos_ans, neg_ans, is_neg_mat, weights, q_args, q_structs
            else:
                yield pos_ans, neg_ans, None, weights, q_args, q_structs
            # yield pos_ans, neg_ans, is_neg_mat if self.share_negative else None, weights, q_args, q_structs
            #self.share_negative=True时，返回pos_ans, neg_ans，is_neg_mat，weights, q_args, q_structs，否则返回pos_ans, neg_ans，None，weights, q_args, q_structs
            q_type = next_q_type
            buf_idx = 1 - buf_idx


def has_negation(st):
    if isinstance(st, tuple):
        for c in st:
            if has_negation(c):
                return True
    else:
        assert isinstance(st, str)
        return st == 'n'
    return False


def print_qt(qt, g, idx):
    import graphviz
    node_type = str(qt.node_type).split('.')[-1]
    root_idx = str(idx)
    color = '#CCCCFF' if qt.sqrt_middle else '#FFFFFF'
    g.node(root_idx, node_type, fillcolor=color)
    idx += 1
    ch_list = []
    qt.get_children(ch_list)
    for ch in ch_list:
        ch_idx = idx
        idx = print_qt(ch, g, ch_idx)
        l = str(ch.parent_edge).split('.')[-1]
        if l == 'no_op':
            l = ''
        s = 'solid'
        if l == 'negation':
            s = 'dashed'
        g.edge(root_idx, str(ch_idx), label=l, style=s)
    return idx


if __name__ == '__main__':
    import time
    db_name = 'FB15k'
    data_folder = os.path.join(os.path.expanduser('~'), 'data/knowledge_graphs/%s' % db_name)
    with open(os.path.join(data_folder, 'stats.txt'), 'r') as f:
        num_ent = f.readline().strip().split()[-1]
        num_rel = f.readline().strip().split()[-1]
        num_ent, num_rel = int(num_ent), int(num_rel)

    kg = libsampler.KG32(num_ent, num_rel)
    kg.load(data_folder + '/train_bidir.bin')
    print('num ent', kg.num_ent)
    print('num rel', kg.num_rel)
    print('num edges', kg.num_edges)

    sampler_type = 'naive'
    query_structures = ["1p"]

    negative_sample_size = 256
    sample_mode = (0, 0, 'u', 'u', 0, True, False)
    sampler = OnlineSampler(kg, query_structures, negative_sample_size, sample_mode, [1.0 / len(query_structures)] * len(query_structures),
                        sampler_type=sampler_type,
                        share_negative=True,
                        same_in_batch=True,
                        num_threads=1)
    batch_gen = sampler.batch_generator(10)
    idx = 0
    for pos_ans, neg_ans, is_neg_mat, weights, q_args, q_structs in tqdm(batch_gen):
        idx += 1
        # if idx > 10:
        #     break
    # for i, qt in enumerate(sampler.list_qt):
    #     g = graphviz.Digraph()
    #     g.node_attr['style']='filled'

    #     print_qt(qt, g, 0)
    #     g.render('graph-%d' % i)
    # log_file = open('%s/%s-%d.txt' % (db_name, sampler_type, bd), 'w')

    # samplers = [None] * len(all_query_structures)
    # for i in range(len(all_query_structures)):
    #     query_structures = all_query_structures[i:i+1]
    #     samplers[i] = OnlineSampler(kg, query_structures, negative_sample_size, sample_mode, [1.0 / len(query_structures)] * len(query_structures), 
    #                         sampler_type=sampler_type,
    #                         num_threads=8)
    #     sampler = samplers[i]
    #     batch_gen = sampler.batch_generator(1024)
    #     t = time.time() 
    #     idx = 0
    #     for pos_ans, weights, q_args, q_structs, neg_ans in batch_gen:
    #         idx += 1
    #         if idx > 10:
    #             break
    #     log_file.write('%d %.4f\n' % (i, (time.time() - t) / 10))
    # log_file.close()
