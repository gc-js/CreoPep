import pandas as pd
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from creopep.vocab import PepVocab
from creopep.utils import mask, create_vocab

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
    note: the function is suitable for the sequences with the format of "label1|label2|sequence|msa1|msa2|msa3"
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

        return [add(vocab.split_seq(msa1_seq))]  + [add(vocab.split_seq(msa2_seq))]  + [add(vocab.split_seq(msa3_seq))]

def get_paded_token_idx(vocab_mlm, train_data_path):
    data_path = train_data_path
    seq = pd.read_csv(data_path)['Sequences']
    
    splited_seq = list(seq.apply(split_seq, args=(vocab_mlm,True, )))
    splited_msa = list(seq.apply(split_seq, args=(vocab_mlm, False, )))
    
    vocab_mlm.set_get_attn(is_get=True)
    padded_seq = vocab_mlm.truncate_pad(splited_seq, num_steps=54, padding_token='[PAD]')
    attn_idx = vocab_mlm.get_attention_mask_mat()

    vocab_mlm.set_get_attn(is_get=False)
    padded_msa = vocab_mlm.truncate_pad(splited_msa, num_steps=54, padding_token='[PAD]')
    
    idx_seq = vocab_mlm[padded_seq]
    
    idx_msa = vocab_mlm[padded_msa]

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
        
        idx_seq = vocab_mlm[padded_seq]
        idx_msa = vocab_mlm[padded_msa]
        
    else:
        splited_seq = split_seq(seq[0], vocab_mlm, True)
        splited_msa = split_seq(seq[0], vocab_mlm, False)
        vocab_mlm.set_get_attn(is_get=True)
        padded_seq = vocab_mlm.truncate_pad(splited_seq, num_steps=54, padding_token='[PAD]')
        attn_idx = vocab_mlm.get_attention_mask_mat()
        vocab_mlm.set_get_attn(is_get=False)
        padded_msa = vocab_mlm.truncate_pad(splited_msa, num_steps=54, padding_token='[PAD]')
        idx_msa = vocab_mlm[padded_msa]
        idx_seq = vocab_mlm[new_seq]
        
    return padded_seq, idx_seq, idx_msa, attn_idx

def make_mask(seq_ser, start, end, time, vocab_mlm, labels, idx_msa, attn_idx, test_size, batch_size, seed, device):
    
    seq_ser = pd.Series(seq_ser)
    masked_seq = seq_ser.apply(mask, args=(start, end, time))
    masked_idx = vocab_mlm[list(masked_seq)]
    masked_idx = torch.tensor(masked_idx)
    
    data_arrays = (masked_idx.to(device), labels.to(device), idx_msa.to(device), attn_idx.to(device)) 
    dataset = TensorDataset(*data_arrays)
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=seed, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader