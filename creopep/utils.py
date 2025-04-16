import torch.nn as nn
import copy, math
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import random
from transformers import set_seed
from vocab import PepVocab

def create_vocab():
    vocab_mlm = PepVocab()
    vocab_mlm.vocab_from_txt('./data/vocab.txt')
    # vocab_mlm.token_to_idx['-'] = 23
    return vocab_mlm

def show_parameters(model: nn.Module, show_all=False, show_trainable=True):

    mlp_pa = {name:param.requires_grad for name, param in model.named_parameters()}
    
    if show_all:
        print('All parameters:')
        print(mlp_pa)

    if show_trainable:
        print('Trainable parameters:')
        print(list(filter(lambda x: x[1], list(mlp_pa.items()))))

class ContraLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ContraLoss, self).__init__(*args, **kwargs)
        
        self.temp = 0.07

    def contrastive_loss(self, proj1, proj2):
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        dot = torch.matmul(proj1, proj2.T) / self.temp
        dot_max, _ = torch.max(dot, dim=1, keepdim=True)
        dot = dot - dot_max.detach()

        exp_dot = torch.exp(dot)
        log_prob = torch.diag(dot, 0) - torch.log(exp_dot.sum(1))
        cont_loss = -log_prob.mean()
        return cont_loss
    
    def forward(self, x, y, label=None):
        return self.contrastive_loss(x, y)




def show_parameters(model: nn.Module, show_all=False, show_trainable=True):

    mlp_pa = {name:param.requires_grad for name, param in model.named_parameters()}
    
    if show_all:
        print('All parameters:')
        print(mlp_pa)

    if show_trainable:
        print('Trainable parameters:')
        print(list(filter(lambda x: x[1], list(mlp_pa.items()))))

def extract_args(text):
    str_list = []
    substr = ""
    for s in text:
        if s in ('(', ')', '=', ',', ' ', '\n', "'"):
            if substr != '':
                str_list.append(substr)
                substr = ''
        else:
            substr += s

def eval_one_epoch(loader, cono_encoder):
    cono_encoder.eval()
    batch_loss = []
    for i, data in enumerate(tqdm(loader)):
        
        loss = cono_encoder.contra_forward(data)
        batch_loss.append(loss.item())
        print(f'[INFO] Test batch {i} loss: {loss.item()}')

    total_loss = np.mean(batch_loss)    
    print(f'[INFO] Total loss: {total_loss}')    
    return total_loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)

class CrossEntropyLossWithMask(torch.nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLossWithMask, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y_pred, y_true, mask):
        (pos_mask, label_mask, seq_mask) = mask
        loss = self.criterion(y_pred, y_true) # (6912)
        
        pos_loss = (loss * pos_mask).sum() / torch.sum(pos_mask)
        label_loss = (loss * label_mask).sum() / torch.sum(label_mask)
        seq_loss = (loss * seq_mask).sum() / torch.sum(seq_mask)
        
        loss = pos_loss + label_loss/2 + seq_loss/3

        return loss


def mask(x, start, end, time):
    ske_pos = np.where(np.array(x)=='C')[0] - start
    lables_pos = np.array([1, 2]) - start
    ske_pos = list(filter(lambda x: end-start >= x >= 0, ske_pos))
    lables_pos = list(filter(lambda x: x >= 0, lables_pos))
    weight = np.ones(end - start+1)
    rand = np.random.rand()
    if rand < 0.5:
        weight[lables_pos] = 100000
    else:
        weight[lables_pos] = 1
    mask_pos = np.random.choice(range(start, end+1), time, p=weight/np.sum(weight), replace=False)
    for idx in mask_pos:
        x[idx]  = '[MASK]'
    return x
