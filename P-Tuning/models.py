#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   models.py
@Author  :   xiatianci(xiatianci@baidu.com)
@Time    :   2023/07/03 14:34:37
@Desc    :   创建模型
"""

from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM

def create_model(args):
    MODEL_CLASS, _ = get_model_and_tokenizer_class(args)
    model = MODEL_CLASS.from_pretrained(args.model_name)
    if not args.use_lm_finetune:
        model = model.half()
    return model


def get_model_and_tokenizer_class(args):
    if 'gpt' in args.model_name:
        return GPT2LMHeadModel, AutoTokenizer
    elif 'bert' in args.model_name:
        return AutoModelForMaskedLM, AutoTokenizer
    elif 'megatron' in args.model_name:
        return None, AutoTokenizer
    else:
        raise NotImplementedError('This model type `{}` is not implemented'.format(args.model_name))

def get_embedding_layer(args, model):
    if 'roberta' in args.model_name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'bert' in args.model_name:
        embeddings = model.bert.get_input_embeddings()
    elif 'gpt' in args.model_name:
        embeddings = model.base_model.get_input_embeddings()
    elif 'megatron' in args.model_name:
        embeddings = model.decoder.embed_tokens
    else:
        raise NotImplementedError()
    return embeddings