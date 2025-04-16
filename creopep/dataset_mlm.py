import pandas as pd
from copy import deepcopy

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from vocab import PepVocab
from utils import mask, create_vocab

addtition_tokens = ['<K16>', '<α1β1γδ>', '<Ca22>', '<AChBP>', '<K13>', '<α1BAR>', '<α1β1ε>', '<α1AAR>', '<GluN3A>', '<α4β2>',
                     '<GluN2B>', '<α75HT3>', '<Na14>', '<α7>', '<GluN2C>', '<NET>', '<NavBh>', '<α6β3β4>', '<Na11>', '<Ca13>', 
                     '<Ca12>', '<Na16>', '<α6α3β2>', '<GluN2A>', '<GluN2D>', '<K17>', '<α1β1δε>', '<GABA>', '<α9>', '<K12>', 
                     '<Kshaker>', '<α3β4>', '<Na18>', '<α3β2>', '<α6α3β2β3>', '<α1β1δ>', '<α6α3β4β3>', '<α2β2>','<α6β4>', '<α2β4>',
                     '<Na13>', '<Na12>', '<Na15>', '<α4β4>', '<α7α6β2>', '<α1β1γ>', '<NaTTXR>', '<K11>', '<Ca23>', 
                     '<α9α10>','<α6α3β4>', '<NaTTXS>', '<Na17>','<high>','<low>']

def add_tokens_to_vocab(vocab_mlm: PepVocab):
    vocab_mlm.add_special_token(addtition_tokens)
    return vocab_mlm

def split_seq(seq, vocab, get_seq=False):
    '''
    note: the function is suitable for the sequences with the format of "label|label|sequence|msa1|msa2|msa3"
    '''
    start = '[CLS]'
    end = '[SEP]'
    pad = '[PAD]'
    cls_label = seq.split('|')[0]
    act_label = seq.split('|')[1]

    if get_seq == True:
        add = lambda x: [start] + [cls_label] + [act_label] + x + [end]
        pep_seq = seq.split('|')[2]
        # return [start] + [cls_label] + [act_label] + vocab.split_seq(pep_seq) + [end]
        return add(vocab.split_seq(pep_seq))
    
    else:
        add = lambda x: [start] + [pad] + [pad] + x + [end]
        msa1_seq = seq.split('|')[3]
        msa2_seq = seq.split('|')[4]
        msa3_seq = seq.split('|')[5]

        # return [vocab.split_seq(msa1_seq)]  + [vocab.split_seq(msa2_seq)]  + [vocab.split_seq(msa3_seq)]
        return [add(vocab.split_seq(msa1_seq))]  + [add(vocab.split_seq(msa2_seq))]  + [add(vocab.split_seq(msa3_seq))]

def get_paded_token_idx(vocab_mlm):
    cono_path = './data/conoData_C5.csv'
    seq = pd.read_csv(cono_path)['Sequences']
    
    splited_seq = list(seq.apply(split_seq, args=(vocab_mlm,True, )))
    splited_msa = list(seq.apply(split_seq, args=(vocab_mlm, False, )))
    
    vocab_mlm.set_get_attn(is_get=True)
    padded_seq = vocab_mlm.truncate_pad(splited_seq, num_steps=54, padding_token='[PAD]')
    attn_idx = vocab_mlm.get_attention_mask_mat()

    vocab_mlm.set_get_attn(is_get=False)
    padded_msa = vocab_mlm.truncate_pad(splited_msa, num_steps=54, padding_token='[PAD]')
    
    idx_seq = vocab_mlm.__getitem__(padded_seq) # [b, 54]  start, cls_label, act_label, sequence, end
    
    idx_msa = vocab_mlm.__getitem__(padded_msa) # [b, 3, 50]

    return padded_seq, idx_seq, idx_msa, attn_idx

def get_paded_token_idx_gen(vocab_mlm, seq):
    
    splited_seq = split_seq(seq[0], vocab_mlm, True)
    splited_msa = split_seq(seq[0], vocab_mlm, False)
    
    vocab_mlm.set_get_attn(is_get=True)
    padded_seq = vocab_mlm.truncate_pad(splited_seq, num_steps=54, padding_token='[PAD]')
    attn_idx = vocab_mlm.get_attention_mask_mat()

    vocab_mlm.set_get_attn(is_get=False)
    padded_msa = vocab_mlm.truncate_pad(splited_msa, num_steps=54, padding_token='[PAD]')
    
    idx_seq = vocab_mlm.__getitem__(padded_seq) # [b, 54]  start, cls_label, act_label, sequence, end
    
    idx_msa = vocab_mlm.__getitem__(padded_msa) # [b, 3, 50]

    return padded_seq, idx_seq, idx_msa, attn_idx


def get_paded_token_idx_gen(vocab_mlm, seq, new_seq):
    if new_seq == None:
        splited_seq = split_seq(seq[0], vocab_mlm, True)
        splited_msa = split_seq(seq[0], vocab_mlm, False)
        
        vocab_mlm.set_get_attn(is_get=True)
        padded_seq = vocab_mlm.truncate_pad(splited_seq, num_steps=54, padding_token='[PAD]')
        attn_idx = vocab_mlm.get_attention_mask_mat()
        vocab_mlm.set_get_attn(is_get=False)

        padded_msa = vocab_mlm.truncate_pad(splited_msa, num_steps=54, padding_token='[PAD]')
        
        idx_seq = vocab_mlm.__getitem__(padded_seq)  # [b, 54]  start, cls_label, act_label, sequence, end
        idx_msa = vocab_mlm.__getitem__(padded_msa)  # [b, 3, 50]
    else:
        splited_seq = split_seq(seq[0], vocab_mlm, True)
        splited_msa = split_seq(seq[0], vocab_mlm, False)
        vocab_mlm.set_get_attn(is_get=True)
        padded_seq = vocab_mlm.truncate_pad(splited_seq, num_steps=54, padding_token='[PAD]')
        attn_idx = vocab_mlm.get_attention_mask_mat()
        vocab_mlm.set_get_attn(is_get=False)
        padded_msa = vocab_mlm.truncate_pad(splited_msa, num_steps=54, padding_token='[PAD]')
        idx_msa = vocab_mlm.__getitem__(padded_msa)  # [b, 3, 50]

        idx_seq = vocab_mlm.__getitem__(new_seq)
    return padded_seq, idx_seq, idx_msa, attn_idx



def make_mask(seq_ser, start, end, time, vocab_mlm, labels, idx_msa, attn_idx):
    seq_ser = pd.Series(seq_ser)
    masked_seq = seq_ser.apply(mask, args=(start, end, time))
    masked_idx = vocab_mlm.__getitem__(list(masked_seq))
    masked_idx = torch.tensor(masked_idx)
    device = torch.device('cuda:0')
    data_arrays = (masked_idx.to(device), labels.to(device), idx_msa.to(device), attn_idx.to(device)) 
    dataset = TensorDataset(*data_arrays)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    
    return train_loader, test_loader

if __name__ == '__main__':
    # from add_args import parse_args
    import numpy as np
    # args = parse_args()

    vocab_mlm = create_vocab()
    vocab_mlm = add_tokens_to_vocab(vocab_mlm)
    padded_seq, idx_seq, idx_msa, attn_idx = get_paded_token_idx(vocab_mlm)
    labels = torch.tensor(idx_seq)
    idx_msa = torch.tensor(idx_msa)
    attn_idx = torch.tensor(attn_idx)

    # time_step = args.mask_time_step
    for t in np.arange(1, 50):
        padded_seq_copy = deepcopy(padded_seq)        
        train_loader, test_loader = make_mask(padded_seq_copy, start=0, end=49, time=t, 
                                              vocab_mlm=vocab_mlm, labels=labels, idx_msa=idx_msa, attn_idx=attn_idx)
        for i, (masked_idx, label, msa, attn) in enumerate(train_loader):
            print(f"the {i}th batch is that masked_idx is {masked_idx.shape}, labels is {label.shape}, idx_msa is {msa.shape}")
        print(f"the {t}th time step is done")
        
        
    
