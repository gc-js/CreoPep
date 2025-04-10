import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from creopep.utils import setup_seed, CrossEntropyLossWithMask, show_parameters, create_vocab
from creopep.dataset_mlm import  get_paded_token_idx, make_mask, add_tokens_to_vocab
from creopep.model import  load_pretrained_model, ConoModel, MSABlock, ConoEncoder

def eval_one_epoch(loader, cono_model, loss_fct, vocab_mlm, device):
    """Evaluate one epoch on validation set"""
    cono_model.eval()
    batch_loss = []
    with torch.no_grad():
        for i, (data, label, msa, attn) in enumerate(loader):
            logits = cono_model(data.to(device), msa.to(device), attn.to(device))
            logits = logits.view(-1, len(vocab_mlm))
            label_reshape = label.view(-1).to(device)
            
            pos_mask = (data == 4).view(-1).to(device)
            label_mask = torch.zeros_like(attn)
            label_mask[:, 1:3] = 1
            label_mask = label_mask.view(-1).to(device)
            
            seq_mask = torch.ones_like(attn)
            seq_mask[:, 0:3] = 0
            seq_mask = seq_mask.view(-1).to(device)

            loss = loss_fct(logits, label_reshape, (pos_mask, label_mask, seq_mask))
            batch_loss.append(loss.item())
        
        total_loss = np.mean(batch_loss)
        return total_loss

if __name__ == '__main__':
    T_step = [27]
    all_ep_loss_train = []
    all_ep_loss_val = []
    for T in T_step:
        epochs = 100
        setup_seed(42)
        device = torch.device('cuda:0')
        
        # load vocab
        vocab_mlm = create_vocab()
        vocab_mlm = add_tokens_to_vocab(vocab_mlm)

        # load pretrained model
        model = load_pretrained_model()
        model.resize_token_embeddings(len(vocab_mlm))

        # cono_encoder 
        bert_part = model.bert
        cono_encoder = ConoEncoder(bert_part).to(device)

        # msa_block
        msa_block = MSABlock(in_dim=128, out_dim=128, vocab_size=len(vocab_mlm)).to(device)
        cono_decoder = model.cls.to(device)

        # integrate model
        cono_model = ConoModel(cono_encoder, msa_block, cono_decoder).to(device)
        
        # freeze encoder
        for param in cono_model.encoder.parameters():
            param.requires_grad = False
        for param in cono_model.encoder.trainable_encoder.parameters():
            param.requires_grad = True
        for param in cono_model.decoder.parameters():
            param.requires_grad = True
        
        show_parameters(cono_model, show_trainable=True)

        # train settings
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, cono_model.parameters()), lr=5e-5)
        loss_fct = CrossEntropyLossWithMask()

        # prepare data
        padded_seq, idx_seq, idx_msa, attn_idx = get_paded_token_idx(vocab_mlm)
        labels = torch.tensor(idx_seq)
        idx_msa = torch.tensor(idx_msa)
        attn_idx = torch.tensor(attn_idx)

        ep_loss_train = []
        ep_loss_val = []
        best_loss = 1e5
        for ep in range(epochs):
            print(f'[INFO] Start training encoder model on {ep} epoch')
            cono_model.train()

            time_loss = []
            time_val_loss = []
            time_step = range(1, T+1)
            for t in time_step:
                padded_seq_copy = deepcopy(padded_seq)
                labels_copy = deepcopy(labels)
                train_loader, test_loader = make_mask(padded_seq_copy, start=0, end=53, time=t, 
                                                    vocab_mlm=vocab_mlm, labels=labels_copy, idx_msa=idx_msa, attn_idx=attn_idx)
                
                for i, (train_data, label, msa, attn) in enumerate(train_loader):
                    batch_loss = []

                    opt.zero_grad()
                    logits = cono_model(train_data, msa, attn)
                    logits = logits.view(-1, len(vocab_mlm))
                    label_reshape = label.view(-1)
                    pos_mask = (train_data==4)
                    pos_mask = pos_mask.view(-1)
                    label_mask = torch.zeros_like(attn)
                    label_mask[:, 1:3] = 1
                    label_mask = label_mask.view(-1)
                    seq_mask = torch.ones_like(attn)
                    seq_mask[:, 0:3] = 0
                    seq_mask = seq_mask.view(-1)

                    loss = loss_fct(logits, label_reshape, (pos_mask, label_mask, seq_mask))
                    loss.backward()
                    opt.step()
                    batch_loss.append(loss.item())
                    time_loss.append(np.mean(batch_loss))
                    
                valid_loss = eval_one_epoch(test_loader, cono_model, loss_fct, vocab_mlm, device)
                time_val_loss.append(valid_loss)
            ep_loss_train.append(np.mean(time_loss))
            ep_loss_val.append(np.mean(time_val_loss))
            print(f'[INFO] Epoch {ep} loss: {ep_loss_train[-1]}, valid loss: {ep_loss_val[-1]}')
            valid_loss = ep_loss_val[-1]

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(cono_model.state_dict(), f'./models/model-param-{T}.pt')
                torch.save(cono_model, f'./models/model-{T}.pt')
        all_ep_loss_train.append(ep_loss_train)
        all_ep_loss_val.append(ep_loss_val)
        
    # Plotting the loss curves for each T_step
    plt.figure(figsize=(10, 8))
    for i, T in enumerate(T_step):
        plt.plot(all_ep_loss_val[i],'-o', label=f"T={T}", markersize=3)

    plt.xticks(np.arange(0, epochs, step=10))
    plt.yticks(np.arange(0, 10, step=1))
    plt.ylim((0, 10))
    plt.savefig(f"./imgs/Loss_curves.png")



