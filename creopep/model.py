import torch.nn as nn
import copy, math
import torch
import train
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoConfig
from bertmodel import make_bert, make_bert_without_emb
from utils import ContraLoss
args = train.get_args()

def load_pretrained_model():
    model_checkpoint = args.PLM
    config = AutoConfig.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_config(config)
    
    return model

class ConoEncoder(nn.Module):
    def __init__(self, encoder):
        super(ConoEncoder, self).__init__()
        
        self.encoder = encoder
        self.trainable_encoder = make_bert_without_emb()

        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        
    def forward(self, x, mask):
        feat = self.encoder(x, attention_mask=mask)
        feat = list(feat.values())[0]
        
        feat = self.trainable_encoder(feat, mask)

        return feat

class MSABlock(nn.Module):
    def __init__(self, in_dim, out_dim, vocab_size):
        super(MSABlock, self).__init__()
        self.embedding = nn.Embedding(vocab_size, in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.init()
    
    def init(self):
        for layer in self.mlp.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        # nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mlp(x)
        return x

class ConoModel(nn.Module):
    def __init__(self, encoder, msa_block, decoder):
        super(ConoModel, self).__init__()
        self.encoder = encoder
        self.msa_block = msa_block
        self.feature_combine = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        self.decoder = decoder

    def forward(self, input_ids, msa, attn_idx=None):
        encoder_output = self.encoder.forward(input_ids, attn_idx)
        msa_output = self.msa_block(msa)
        encoder_output = encoder_output.view(input_ids.shape[0], 54, -1).unsqueeze(1)
        
        output = torch.cat([encoder_output*5, msa_output], dim=1)
        output = self.feature_combine(output)
        output = output.squeeze(1)
        logits = self.decoder(output)
        
        return logits
