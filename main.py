#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from models import KGReasoning
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from tensorboardX import SummaryWriter
import pickle
from collections import defaultdict
from util import flatten_query, parse_time, set_global_seed, eval_tuple

####################################################################################################
from cpp_sampler.online_sampler import OnlineSampler
import torch.distributed as dist
from cpp_sampler.sampler_clib import KGMem
import torch.multiprocessing as mp
import math
from evaluation.dataloader import MultihopTestDataset, Test1pDataset, Test1pBatchDataset
import collections
from collections import namedtuple
QueryData = namedtuple('QueryData', ['data', 'buffer', 'writer_buffer'])
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='4'

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys()) # ['1p', '2p', '3p', '4p', '5p']

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    
    parser.add_argument('--do_train', action='store_true', help="do train", default=True)
    parser.add_argument('--do_valid', action='store_true', help="do valid", default=True)
    parser.add_argument('--do_test', action='store_true', help="do test", default=True)

    parser.add_argument('--data_path', type=str, default='./data/FB15k-237-long_chain', help="KG data path")
    parser.add_argument('--eval_path', type=str, default=None, help="KG eval data path")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=400, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=0.375, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=512, type=int, help="batch size of queries")
    parser.add_argument('--drop', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=1, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default='./logs/FB15k-237/gqe_baseline_test', type=str, help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=300000, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int, help="no need to set manually, will configure automatically")
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('--geo', default='logic', type=str, choices=['vec', 'box', 'beta', 'cone', 'logic'], help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')
    parser.add_argument('--print_on_screen', action='store_true', default=True)
    
    parser.add_argument('--tasks', default='1p.2p.3p.4p.5p', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str, help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')
    parser.add_argument('-cenr', '--center_reg', default=0.02, type=float, help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
    parser.add_argument('-logicm', '--logic_mode', default="(luk,1,1,0,1600,2)", type=str, help='(tnorm,bounded,use_att,use_gtrans,hidden_dim,num_layer)')
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    parser.add_argument('--model_mode', default="baseline", type=str, choices=['baseline', 'temp'], help='the type of model')
    parser.add_argument('--faithful', default="no_faithful", type=str, choices=['faithful', 'no_faithful'], help='faithful or not')
    parser.add_argument('--neighbor_ent_type_samples', type=int, default=32, help='number of sampled entity type neighbors')
    parser.add_argument('--neighbor_rel_type_samples', type=int, default=64, help='number of sampled relation type neighbors')

###############################################################################################################################################
    parser.add_argument('--training_tasks', default=None, type=str,
                        help="training tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--online_sample', action='store_true', default=False)
    parser.add_argument('--online_sample_mode', default="(500,-1,w,wstruct,80,True,True)", type=str,
                        help='(0,0,w/u,wstruct/u,0,True,True) or (relation_bandwidth,max_num_of_intermediate,w/u,wstruct/u,max_num_of_partial_answer,weighted_ans,weighted_neg)')
    parser.add_argument('--share_negative', action='store_true', default=False)
    parser.add_argument('--kg_dtype', default='uint32', type=str, choices=['uint32', 'uint64'], help='data type of kg')
    parser.add_argument('--sampler_type', type=str, default='naive',
                        help="type of sampler, choose from [naive, sqrt, nosearch, mix_0.x]")
    parser.add_argument('--eval_link_pred', action='store_true', default=False)
    parser.add_argument('--filter_test', action='store_true', default=False)
    parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size for eval')
    parser.add_argument('--train_online_mode', default="(single,500,er,False,before)", type=str,
                        help='(mix/single,sync_steps,er/e/r/n trained on cpu,async flag, before/after)')
    parser.add_argument('--optim_mode', default="(fast,adagrad,cpu,True,5)", type=str,
                        help='(fast/aggr,adagrad/rmsprop,cpu/gpu,True/False,queue_size)')
    parser.add_argument('--online_weighted_structure_prob', default="(70331,141131,438875)", type=str,
                        help='(same,0,w/u,wstruct/u)')
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    return parser.parse_args(args)

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H：%M：%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def evaluate(model, args, dataloader, query_name_dict, mode, step, writer):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)


    metrics = model.test_step(model, args, dataloader, query_name_dict)
    # negative_sample, queries, queries_unflatten, query_structures
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode+" "+query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]), metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average'%mode, step, average_metrics)

    return all_metrics
        
# def load_data(args, tasks):
#     '''
#     Load queries and remove queries not in tasks
#     '''
#     logging.info("loading data")
#     train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
#     train_answers = pickle.load(open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))
#     valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
#     valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
#     valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
#     test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
#     test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
#     test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))
#
#     if args.faithful == 'faithful':
#         # Train on all splits to evaluate reasoning faithfulness (entailment)
#         for queries in [valid_queries, test_queries]:
#             for query_structure in queries:
#                 train_queries[query_structure] |= queries[query_structure]
#
#         for answers in [valid_hard_answers, valid_easy_answers, test_hard_answers, test_easy_answers]:
#             for query in answers:
#                 train_answers.setdefault(query, set())
#                 train_answers[query] |= answers[query]
#
#     # remove tasks not in args.tasks
#     for name in all_tasks:
#         if 'u' in name:
#             name, evaluate_union = name.split('-')
#         else:
#             evaluate_union = args.evaluate_union
#         if name not in tasks or evaluate_union != args.evaluate_union:
#             query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
#             if query_structure in train_queries:
#                 del train_queries[query_structure]
#             if query_structure in valid_queries:
#                 del valid_queries[query_structure]
#             if query_structure in test_queries:
#                 del test_queries[query_structure]
#
#     return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers

# def setup_train_mode(args):
#     #tasks=['1p','2p','3p','2i','3i','ip','pi','2u','up']
#     tasks = args.tasks.split('.')
#     if args.training_tasks is None:
#         args.training_tasks = args.tasks
#     #training_tasks=['1p','2p','3p','2i','3i']
#     training_tasks = args.training_tasks.split('.')
#
#     #online_sample=True   online_sample_mode='(500,0,w,wstruct,120)'
#     if args.online_sample:
#         if eval_tuple(args.online_sample_mode)[3] == 'wstruct':
#             #normalized_structure_prob=[2,2,2,1,1]
#             normalized_structure_prob = np.array(eval_tuple(args.online_weighted_structure_prob)).astype(np.float32)
#             #normalized_structure_prob=[0.25,0.25,0.25,0.125,0.125]
#             normalized_structure_prob /= np.sum(normalized_structure_prob)
#             normalized_structure_prob = normalized_structure_prob.tolist()
#             assert len(normalized_structure_prob) == len(training_tasks)
#         else:
#             normalized_structure_prob = [1/len(training_tasks)] * len(training_tasks)
#         args.normalized_structure_prob = normalized_structure_prob
#         #train_online_mode=('single',3000,'e',True,'before')
#         train_online_mode = eval_tuple(args.train_online_mode)
#         train_dataset_mode, sync_steps, sparse_embeddings, async_optim, merge_mode = eval_tuple(args.train_online_mode)
#         #optim_mode=('aggr','adam','cpu',False,5)
#         update_mode, optimizer_name, optimizer_device, squeeze_flag, queue_size = eval_tuple(args.optim_mode)
#         assert train_dataset_mode in ['single'], "mix has been deprecated"
#         assert update_mode in ['aggr'], "fast has been deprecated"
#         assert optimizer_name in ['adagrad', 'rmsprop', 'adam']
#         args.sync_steps = sync_steps
#         args.async_optim = async_optim
#         args.merge_mode = merge_mode
#         args.sparse_embeddings = sparse_embeddings
#         args.sparse_device = optimizer_device
#         args.train_dataset_mode = train_dataset_mode

def load_1p_eval_data(args, phase):
    logging.info("loading %s data for link pred" % phase)
    all_data = torch.load(os.path.join(args.eval_path, "%s.pt" % phase))
    if 'head_neg' in all_data:  # bi-directional
        logging.info('evaluating bi-directional 1p')
        fwd_data = {'head': all_data['head'],
                    'relation': all_data['relation'] * 2,
                    'tail': all_data['tail']}
        if 'tail_neg' in all_data:
            fwd_data['tail_neg'] = all_data['tail_neg']
        backwd_data = {'head': all_data['tail'],
                    'relation': all_data['relation'] * 2 + 1,
                    'tail': all_data['head']}
        if 'head_neg' in backwd_data:
            backwd_data['tail_neg'] = all_data['head_neg']
        merged_dict = {}
        for key in fwd_data:
            merged_dict[key] = np.concatenate([fwd_data[key], backwd_data[key]])
    else:
        logging.info('evaluating uni-directional 1p')
        fwd_data = {'head': all_data['head'],
                    'relation': all_data['relation'],
                    'tail': all_data['tail']}
        if 'tail_neg' in all_data:
            fwd_data['tail_neg'] = all_data['tail_neg']
        merged_dict = fwd_data
    if args.eval_batch_size > 1:
        test_dataset = Test1pBatchDataset(merged_dict, args.nentity, args.nrelation)
    else:
        test_dataset = Test1pDataset(merged_dict, args.nentity, args.nrelation)

    logging.info("%s info:" % phase)
    logging.info("num queries: %s" % len(test_dataset))
    buf = mp.Queue()
    writer_buffer = mp.Queue()
    return QueryData(test_dataset, buf, writer_buffer)

def load_eval_data(args, phase):
    tasks = args.tasks.split('.')
    logging.info("loading %s data" % phase)
    if args.eval_path is not None:
        all_data = pickle.load(open(os.path.join(args.eval_path, "all-%s-data.pkl" % phase), 'rb'))

        # remove tasks not in args.tasks
        query_structures_to_remove = []
        for name in all_tasks:
            if not args.filter_test:
                continue
            if 'u' in name:
                name, evaluate_union = name.split('-')
            else:
                evaluate_union = args.evaluate_union
            if name not in tasks or evaluate_union != args.evaluate_union:
                query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
                query_structures_to_remove.append(query_structure)
        if len(query_structures_to_remove) != 0:
            all_data = [data for data in all_data if data[0] not in query_structures_to_remove]
    else:
        print('no %s data found' % phase)
        all_data = []
    test_dataset = MultihopTestDataset(all_data, args.nentity, args.nrelation)
    logging.info("%s info:" % phase)
    logging.info("num queries: %s" % len(test_dataset))
    buf = mp.Queue()
    writer_buffer = mp.Queue()
    return QueryData(test_dataset, buf, writer_buffer)

def async_aggr(args, result_buffer, writer_buffer, mode):
    all_result_dicts = collections.defaultdict(list)
    num_received = 0
    num_finished = 0
    num_workers = 1

    while True:
        if num_finished == num_workers and num_received == 0:  # no more write jobs
            writer_buffer.put((None, None, -1))
            return
        result_dict, step = result_buffer.get()
        if result_dict is None:
            num_finished += 1
            assert num_finished <= num_workers
            continue
        for query_structure in result_dict:
            all_result_dicts[query_structure].extend([result_dict[query_structure]])
        num_received += 1
        if num_received == num_workers:
            metrics = collections.defaultdict(lambda: collections.defaultdict(int))
            for query_structure in all_result_dicts:
                num_queries = sum([result_dict['num_queries'] for result_dict in all_result_dicts[query_structure]])
                for metric in all_result_dicts[query_structure][0].keys():
                    if metric in ['num_hard_answer', 'num_queries']:
                        continue
                    metrics[query_structure][metric] = sum([result_dict[metric] for result_dict in all_result_dicts[query_structure]]) / num_queries
                metrics[query_structure]['num_queries'] = num_queries
            average_metrics = collections.defaultdict(float)
            all_metrics = collections.defaultdict(float)

            num_query_structures = 0
            num_queries = 0
            for query_structure in metrics:
                qname = query_name_dict[query_structure] if query_structure in query_name_dict else str(query_structure)
                log_metrics(mode+" "+qname, step, metrics[query_structure])
                for metric in metrics[query_structure]:
                    all_metrics["_".join([qname, metric])] = metrics[query_structure][metric]
                    if metric != 'num_queries':
                        average_metrics["_".join([metric, "qs"])] += metrics[query_structure][metric]
                        average_metrics["_".join([metric, "q"])] += metrics[query_structure][metric] * metrics[query_structure]['num_queries']
                num_queries += metrics[query_structure]['num_queries']
                num_query_structures += 1

            for metric in average_metrics:
                if '_qs' in metric:
                    average_metrics[metric] /= num_query_structures
                else:
                    average_metrics[metric] /= num_queries
                all_metrics["_".join(["average", metric])] = average_metrics[metric]
            for metric in all_metrics:
                print(metric, all_metrics[metric])
            log_metrics('%s average'%mode, step, average_metrics)

            writer_buffer.put((dict(metrics), dict(average_metrics), step))
            all_result_dicts = collections.defaultdict(list)
            num_received = 0

def main(args):
    set_global_seed(args.seed)
    # setup_train_mode(args)
    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta' or args.geo == 'cone' or args.geo == 'logic', "only BetaE supports modeling union using De Morgan's Laws"

    cur_time = parse_time()
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix

    print ("overwritting args.save_path")
    args.save_path = os.path.join(args.save_path, args.tasks, args.geo)
    if args.geo in ['box']:
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.box_mode)
    elif args.geo in ['vec']:
        tmp_str = "g-{}".format(args.gamma)
    elif args.geo == 'beta':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.beta_mode)
    elif args.geo == 'cone':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.center_reg)
    elif args.geo == 'logic':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.logic_mode)

    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path
    else:
        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print ("logging to", args.save_path)
    if not args.do_train: # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    set_logger(args)

    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
        ntype = int(entrel[2].split(' ')[-1])
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('-------------------------------'*3)
    logging.info('Geo: %s' % args.geo)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)

    # train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data(args, tasks)

    entity2type, relation2type = build_kg(args.data_path, args.neighbor_ent_type_samples,
                                          args.neighbor_rel_type_samples)
    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        ntype=ntype,
        entity2type=entity2type,
        relation2type=relation2type,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        geo=args.geo,
        use_cuda = args.cuda,
        box_mode=eval_tuple(args.box_mode),
        beta_mode = eval_tuple(args.beta_mode),
        center_reg=args.center_reg,
        logic_mode=eval_tuple(args.logic_mode),
        model_mode = args.model_mode,
        test_batch_size=args.test_batch_size,
        query_name_dict = query_name_dict,
        drop=args.drop,
        neighbor_ent_type_samples=args.neighbor_ent_type_samples
    )


    logging.info("Training info:")
    if args.do_train:
        # for query_structure in train_queries:
        #     logging.info(query_name_dict[query_structure]+": "+str(len(train_queries[query_structure])))
        # train_path_queries = defaultdict(set)
        # train_other_queries = defaultdict(set)
        # path_list = ['1p', '2p', '3p', '4p', '5p']
        # for query_structure in train_queries:
        #     if query_name_dict[query_structure] in path_list:
        #         train_path_queries[query_structure] = train_queries[query_structure]
        #     else:
        #         train_other_queries[query_structure] = train_queries[query_structure]
        # train_path_queries = flatten_query(train_path_queries)
        # train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
        #                             TrainDataset(train_path_queries, nentity, nrelation, args.negative_sample_size, train_answers),
        #                             batch_size=args.batch_size,
        #                             shuffle=True,
        #                             num_workers=args.cpu_num,
        #                             collate_fn=TrainDataset.collate_fn
        #                         ))
#         positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure

# ############################################################################################################################
        kg_mem = KGMem(dtype=args.kg_dtype)
        kg_mem.load(os.path.join(args.data_path, 'train_bidir.bin'))
        kg = kg_mem.create_kg()
        training_tasks = args.training_tasks.split('.')
        init_sampler_type = 'nosearch' if args.sampler_type.startswith('mix') else args.sampler_type

        train_sampler = OnlineSampler(kg, training_tasks, args.negative_sample_size, eval_tuple(args.online_sample_mode), [0.25, 0.25, 0.25, 0.125, 0.125],
                                      sampler_type=args.sampler_type,
                                      share_negative=args.share_negative,
                                      same_in_batch=True,
                                      num_threads=args.cpu_num)
        train_sampler.set_seed(1)
        train_path_iterator = train_sampler.batch_generator(args.batch_size)
#pos_ans, neg_ans, is_neg_mat, weights, q_args, q_structs

        train_other_iterator = None
        # if len(train_other_queries) > 0:
        #     train_other_queries = flatten_query(train_other_queries)
        #     train_other_iterator = None
        #     '''
        #
        #     train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
        #                                 TrainDataset(train_other_queries, nentity, nrelation, args.negative_sample_size, train_answers),
        #                                 batch_size=args.batch_size,
        #                                 shuffle=True,
        #                                 num_workers=args.cpu_num,
        #                                 collate_fn=TrainDataset.collate_fn
        #                             ))
        #     '''
        # else:
        #     train_other_iterator = None
    
    logging.info("Validation info:")
######################################################################################################
    eval_dict = {}
    aggr_procs = []
    for phase in ['valid', 'test']:
        if getattr(args, 'do_%s' % phase, False):
            if args.eval_link_pred:  # load ogb benchmark 1p dataset
                d = load_1p_eval_data(args, phase)
            else:
                d = load_eval_data(args, phase)
            result_aggregator = mp.Process(target=async_aggr, args=(args, d.buffer, d.writer_buffer, 'phase'))
            result_aggregator.start()
            aggr_procs.append(result_aggregator)
            eval_dict[phase] = d
####################################得到eval_dict{}#########################################

    local_eval_dict = {}
    for phase in eval_dict:
        q_data = eval_dict[phase]
        nq_per_proc = math.ceil(len(q_data.data) / 1)
        local_eval_dict[phase] = QueryData(q_data.data.subset(0 * nq_per_proc, nq_per_proc), q_data.buffer,q_data.writer_buffer)
####################################得到local_eval_dict{}##############################################################################

    data_loaders = {}
    for key in local_eval_dict:
        data_loaders[key] = DataLoader(
            eval_dict[key].data,
            batch_size=args.eval_batch_size,
            num_workers=1,
            collate_fn=eval_dict[key].data.collate_fn
        )
###################################得到data_loaders{}##################################################
#negative_sample, queries, queries_unflatten, query_structures,     easy_answers, hard_answers




    # if args.do_valid:
        # for query_structure in valid_queries:
        #     logging.info(query_name_dict[query_structure]+": "+str(len(valid_queries[query_structure])))
        #
        # valid_queries = flatten_query(valid_queries)
        # valid_dataloader = DataLoader(
        #     TestDataset(
        #         valid_queries,
        #         args.nentity,
        #         args.nrelation,
        #     ),
        #     batch_size=args.test_batch_size,
        #     num_workers=args.cpu_num,
        #     collate_fn=TestDataset.collate_fn
        # )



    # logging.info("Test info:")
    # if args.do_test:
    #     for query_structure in test_queries:
    #         logging.info(query_name_dict[query_structure]+": "+str(len(test_queries[query_structure])))
    #     test_queries = flatten_query(test_queries)
    #     test_dataloader = DataLoader(
    #         TestDataset(
    #             test_queries,
    #             args.nentity,
    #             args.nrelation,
    #         ),
    #         batch_size=args.test_batch_size,
    #         num_workers=args.cpu_num,
    #         collate_fn=TestDataset.collate_fn
    #     )





    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        model = model.cuda()
    
    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate
        )
        warm_up_steps = args.max_steps // 2

    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0

    step = init_step 
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    elif args.geo == 'cone':
        logging.info('cone mode = %s' % args.center_reg)
    elif args.geo == 'logic':
        logging.info('logic mode = %s (tnorm,bounded,use_att,use_gtrans,hidden_dim,num_layer)' % args.logic_mode)
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('Evaluate unions using: %s' % args.evaluate_union)
    
    if args.do_train:
        training_logs = []
        # #Training Loop
        for step in range(init_step, args.max_steps):
            if step == 2*args.max_steps//3:
                args.valid_steps *= 4

            log = model.train_step(model, optimizer, train_path_iterator, args, step)
            for metric in log:
                writer.add_scalar('path_'+metric, log[metric], step)
            if train_other_iterator is not None:
                log = model.train_step(model, optimizer, train_other_iterator, args, step)
                for metric in log:
                    writer.add_scalar('other_'+metric, log[metric], step)
                log = model.train_step(model, optimizer, train_path_iterator, args, step)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, args)

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    # for phase in eval_dict:
                    # valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args, valid_dataloader, query_name_dict, 'Valid', step, writer)
                    valid_all_metrics = evaluate(model, args, data_loaders['valid'], query_name_dict, 'Valid', step, writer)
                #negative_sample, queries, queries_unflatten, query_structures
                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    # for phase in eval_dict:
                    test_all_metrics = evaluate(model, args, data_loaders['test'], query_name_dict, 'Test', step, writer)
                    # test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args)
        
    try:
        print(step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        # for phase in eval_dict:
        test_all_metrics = evaluate(model, args, data_loaders['test'],query_name_dict, 'Test', step, writer)
        # test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)

    logging.info("Training finished!!")

def build_kg(data_path, neighbor_ent_type_samples, neighbor_rel_type_samples):
    entity_type_mapping = np.load(data_path + '/entity_type.npy', allow_pickle=True)
    entity2types = []
    for i in range(len(entity_type_mapping)):
        sampled_types = np.random.choice(entity_type_mapping[i], size=neighbor_ent_type_samples,
                                             replace=len(entity_type_mapping[i]) < neighbor_ent_type_samples)
        entity2types.append(sampled_types)

    relation_type_mapping = np.load(data_path + '/relation_type.npy', allow_pickle=True)
    relation2types = []
    for i in range(len(relation_type_mapping)):
        sampled_types = np.random.choice(relation_type_mapping[i], size=neighbor_rel_type_samples,
                                         replace=len(relation_type_mapping[i]) < neighbor_rel_type_samples)
        relation2types.append(sampled_types)
    return entity2types, relation2types

if __name__ == '__main__':
    main(parse_args())