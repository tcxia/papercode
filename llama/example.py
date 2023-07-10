#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   example.py
@Author  :   xiatianci(xiatianci@baidu.com)
@Time    :   2023/07/10 18:12:58
@Desc    :   实例
"""


from typing import Tuple
import os
import sys
import time
import json
import fire
from pathlib import Path

import torch

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import LLaMA, ModelArgs, Transformer, Tokenizer

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    word_size = int(os.environ.get("WORD_SIZE", -1))

    initialize_model_parallel(word_size)
    torch.cuda.set_device(local_rank)

    torch.manual_seed(1)

    return local_rank, word_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    word_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    assert word_size == len(checkpoints), f"Loading a checkpoint for MP={len(checkpoints)} but word size is {word_size}"

    ckpt_path = checkpoints[local_rank]
    print("Loading")

    checkpoints = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )

    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoints, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, word_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, word_size, max_seq_len, max_batch_size
    )

    prompts = [
        "I believe the meaning of life is",
        "Simple put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
    ]

    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)