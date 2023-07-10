#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   tokenizer.py
@Author  :   xiatianci(xiatianci@baidu.com)
@Time    :   2023/07/10 16:07:52
@Desc    :   tokenizer BPE
"""

from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os

logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

        assert self.sp_model.vocab_size() == self.sp_model.GetPieceSize()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.Encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.Decode(t)

