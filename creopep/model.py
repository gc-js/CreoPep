import torch.nn as nn
import copy, math
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoConfig

from bertmodel import make_bert, make_bert_without_emb
from utils import ContraLoss
    
def load_pretrained_model():
    # model_checkpoint = "/home/ubuntu/work/zq/conoMLM/prot_bert/prot_bert"
    model_checkpoint = "/home/ubuntu/work/gecheng/conoGen_final/FinalCono/MLM/prot_bert_finetuned_model_mlm_best"
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
        
        
    def forward(self, x, mask):  # x:(128,54)  mask:(128,54)
        feat = self.encoder(x, attention_mask=mask)  # (128,54,128)
        feat = list(feat.values())[0] # (128,54,128)
        
        feat = self.trainable_encoder(feat, mask) # (128,54,128)

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

    def forward(self, x):  # x: (128,3,54)
        x = self.embedding(x) # x: (128,3,54,128)
        x = self.mlp(x) # x: (128,3,54,128)
        return x

class ConoModel(nn.Module):
    def __init__(self, encoder, msa_block, decoder):
        super(ConoModel, self).__init__()
        self.encoder = encoder
        self.msa_block = msa_block
        self.feature_combine = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        self.decoder = decoder

    def forward(self, input_ids, msa, attn_idx=None):
        # 仅使用 input_ids 作为输入，获取编码器输出
        encoder_output = self.encoder.forward(input_ids, attn_idx) # (128,54,128)
        msa_output = self.msa_block(msa) # (128,3,54,128)
        # msa_output = torch.mean(msa_output, dim=1)
        encoder_output = encoder_output.view(input_ids.shape[0], 54, -1).unsqueeze(1) # (128,1,54,128)
        
        output = torch.cat([encoder_output*5, msa_output], dim=1) # (128,4,54,128)
        output = self.feature_combine(output) # (128,1,54,128)
        output = output.squeeze(1) # (128,54,128)
        # 解码器对编码器的输出进行解码
        logits = self.decoder(output) # (128,54,85)
        
        return logits

class ContraModel(nn.Module):
    def __init__(self, cono_encoder):
        super(ContraModel, self).__init__()
        
        self.contra_loss = ContraLoss()

        self.encoder1 = cono_encoder
        self.encoder2 = make_bert(404, 6, 128)

        # contrastive decoder
        self.lstm = nn.LSTM(16, 16, batch_first=True)
        self.contra_decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        
        # classifier
        self.pre_classifer = nn.LSTM(128, 64, batch_first=True)
        self.classifer = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 6),
            nn.Softmax(dim=-1)
        )

        self.init()

    def init(self):
        
        for layer in self.contra_decoder.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.classifer.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.pre_classifer.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.lstm.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def compute_class_loss(self, feat1, feat2, labels):
        _, cls_feat1= self.pre_classifer(feat1)
        _, cls_feat2 = self.pre_classifer(feat2)
        cls_feat1 = torch.cat([cls_feat1[0], cls_feat1[1]], dim=-1).squeeze(0)
        cls_feat2 = torch.cat([cls_feat2[0], cls_feat2[1]], dim=-1).squeeze(0)

        cls1_dis = self.classifer(cls_feat1)
        cls2_dis = self.classifer(cls_feat2)
        cls1_loss = F.cross_entropy(cls1_dis, labels.to('cuda:0'))
        cls2_loss = F.cross_entropy(cls2_dis, labels.to('cuda:0'))
        
        return cls1_loss, cls2_loss

    def compute_contrastive_loss(self, feat1, feat2):
    
        contra_feat1 = self.contra_decoder(feat1)
        contra_feat2 = self.contra_decoder(feat2)
        
        _, feat1 = self.lstm(contra_feat1)
        _, feat2 = self.lstm(contra_feat2)
        feat1 = torch.cat([feat1[0], feat1[1]], dim=-1).squeeze(0)
        feat2 = torch.cat([feat2[0], feat2[1]], dim=-1).squeeze(0)

        ctr_loss = self.contra_loss(feat1, feat2)
    
        return ctr_loss
    
    def forward(self, x1, x2, labels=None):
        loss = dict()

        idx1, attn1 = x1
        idx2, attn2 = x2
        feat1 = self.encoder1(idx1.to('cuda:0'), attn1.to('cuda:0'))
        feat2 = self.encoder2(idx2.to('cuda:0'), attn2.to('cuda:0'))
        
        cls1_loss, cls2_loss = self.compute_class_loss(feat1, feat2, labels)

        ctr_loss = self.compute_contrastive_loss(feat1, feat2)

        loss['cls1_loss'] = cls1_loss
        loss['cls2_loss'] = cls2_loss
        loss['ctr_loss'] = ctr_loss

        return loss