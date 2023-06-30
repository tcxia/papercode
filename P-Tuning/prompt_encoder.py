#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   prompt_encoder.py
@Author  :   xiatianci(xiatianci@baidu.com)
@Time    :   2023/06/30 17:00:30
@Desc    :   prompt的编码模式
"""

import torch
import torch.nn as nn

class PromptEncoder(nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args) -> None:
        super().__init__()
        self.device = device
        self.spell_lenght = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args

        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0] # first cloze
            + [1] * self.cloze_length[1] # second cloze
            + [1] * self.cloze_length[2] # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)
        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)

        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds