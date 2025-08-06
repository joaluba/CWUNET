#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Mel-spectrogram loss modules."""
import os
import torch
import torch.nn.functional as F
import helpers as hlp
import trainer


class EmbeddingLossCosine(torch.nn.Module):
    def __init__(self,checkpointpath,device="cuda"):
        super(EmbeddingLossCosine, self).__init__()

        self.criterion_emb=torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(device)
         # load training configuration
        config_train=hlp.load_config(os.path.join(os.path.dirname(checkpointpath),"train_config.yaml"))
        # load model architecture
        model=trainer.load_chosen_model(config_train,config_train["modeltype"]).to(device)
        # load weights from checkpoint
        train_results=torch.load(os.path.join(checkpointpath),map_location=device,weights_only=True)
        model.load_state_dict(train_results["model_state_dict"])
        self.reverb_encoder=model.conditioning_network

    def forward(self, x1, x2):
        
        # (B,N) -> (B,C,N) 
        if len(x1.shape)<3:
            x1=x1.unsqueeze(1)
        if len(x2.shape)<3:
            x2=x2.unsqueeze(1)
            
        x1_emb=self.reverb_encoder(x1)
        x2_emb=self.reverb_encoder(x2)

        loss = torch.mean(self.criterion_emb(x1_emb,x2_emb), 0)
        return loss

class EmbeddingLossEuclidean(torch.nn.Module):
    def __init__(self,checkpointpath,device="cuda"):
        super(EmbeddingLossEuclidean, self).__init__()

        self.criterion_emb=torch.nn.PairwiseDistance().to(device)
         # load training configuration
        config_train=hlp.load_config(os.path.join(os.path.dirname(checkpointpath),"train_config.yaml"))
        # load model architecture
        model=trainer.load_chosen_model(config_train,config_train["modeltype"]).to(device)
        # load weights from checkpoint
        train_results=torch.load(os.path.join(checkpointpath),map_location=device, weights_only=True)
        model.load_state_dict(train_results["model_state_dict"])
        self.reverb_encoder=model.conditioning_network

    def forward(self, x1, x2):
        
        # (B,N) -> (B,C,N) 
        if len(x1.shape)<3:
            x1=x1.unsqueeze(1)
        if len(x2.shape)<3:
            x2=x2.unsqueeze(1)
            
        x1_emb=self.reverb_encoder(x1)
        x2_emb=self.reverb_encoder(x2)

        loss = torch.mean(self.criterion_emb(x1_emb,x2_emb), 0)
        return loss

