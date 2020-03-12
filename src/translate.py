#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""python translate.py -model /save/checkpoints/USPTO/MIT_mixed_augm/USPTO_model_step_20000.pt -src /data/USPTO/MIT_mixed_augm/src-test10.txt"""
from __future__ import unicode_literals
import sys
sys.path.append("../..")
sys.path.insert(0, '../OpenNMT-py')
import IPython
from IPython import embed

import IPython
from IPython import embed
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
import pdb
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

class TranslatorClass:
    def __init__(self, 
                 opts,
                 model,
                 n_best=1,
                 beam_size=5,
                 random_sampling_topk=1,
                 seed=1,
                 verbose=False,
                 max_length=300,
                 batch_size=128):
        self.opts = opts
        self.model = model
        self.n_best = n_best
        self.beam_size = beam_size
        self.random_sampling_topk = random_sampling_topk
        self.verbose = verbose
        self.max_length = max_length
        self.seed = seed
        self.batch_size = batch_size

        vars(opts)['models'] = [self.model]
        vars(opts)['n_best'] = self.n_best
        vars(opts)['gpu'] = 0
        vars(opts)['replace_unk'] = True
        vars(opts)['beam_size'] = self.beam_size
        vars(opts)['random_sampling_topk'] = self.random_sampling_topk
        vars(opts)['verbose'] = self.verbose
        vars(opts)['max_length'] = self.max_length
        vars(opts)['seed'] = self.seed
        vars(opts)['batch_size'] = self.batch_size
        
        ArgumentParser.validate_translate_opts(opts)
        self.translator = build_translator(opts, report_score=False)
    
    def translate(self, src, tgt=None):
        """:param src (str): path
                or src is a list of sequences
        """
        if type(src)==str:
            vars(self.opts)['src'] = src
            vars(self.opts)['tgt'] = tgt
            src_shards = split_corpus(self.opts.src, self.opts.shard_size)
            tgt_shards = split_corpus(self.opts.tgt, self.opts.shard_size) \
                if self.opts.tgt is not None else repeat(None)
        else:
            src_shards = split_list(src, self.opts.shard_size)
            tgt_shards = split_list(tgt, self.opts.shard_size) \
                if tgt is not None else repeat(None)
        
        shard_pairs = zip(src_shards, tgt_shards)
        global_scores, global_sequences = [], []
        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            output = self.translator.translate(
                    src=src_shard,
                    tgt=tgt_shard,
                    src_dir=self.opts.src_dir,
                    batch_size=self.opts.batch_size,
                    # batch_type=self.opts.batch_type,
                    attn_debug=self.opts.attn_debug
                    )
            scores, sequences = output[0], output[1]
            global_scores.extend(scores)
            global_sequences.extend(sequences)
        return global_scores, global_sequences

def _get_parser():
    parser = ArgumentParser(description='translate.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser

def translate(model,
              src,
              tgt,
              n_best=1,
              beam_size=5,
              random_sampling_topk=1,
              seed=1):
    # init translator
    parser = _get_parser()
    opts = parser.parse_args()
    trans = TranslatorClass(opts,
                            model,
                            n_best=n_best,
                            beam_size=beam_size,
                            random_sampling_topk=random_sampling_topk,
                            seed=seed)
    # translate
    scores, preds = trans.translate(src, tgt)
    return scores, preds

def unroll(x, n):
    """turn a list of lists of length n into a single list
        :param x list of lists
        :n length of each list. note n <= sub list
    """
    unrolled_x = []
    for i in range(len(x)):
        for j in range(n):
            unrolled_x.append(x[i][j])
    return unrolled_x

def split_list(x, n):
    """x is a list of lists where each sublist of length one
    return a list of lists of size n"""
    new_x = []
    for i in range(len(x)):
        if i % n == 0:
            new_x.append(unroll(x[i:i+n], 1))
    return new_x
 