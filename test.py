from creopep.vocab import PepVocab
from creopep.dataset_mlm import *
from train import get_args

def test_vocab():
    data1 = ['PACCTHPACHVNHPELC']
    data2 = ['PACCTHPACHVNHPELC', 'PACCTHPACHVNHPELC']
    data3 = [['PACCTHPACHVNHPELC', 'PACCTHPACHVNHPELC', 'PACCTHPACHVNHPELC'],
             ['PACCTHPACHVNHPELC', 'PACCTHPACHVNHPELC', 'PACCTHPACHVNHPELC']]
    vocab = PepVocab()
    data1_idx = vocab.split_seq(data1)
    data2_idx = vocab.split_seq(data2)
    data3_idx = vocab.split_seq(data3)
    
    data1_idx = vocab.truncate_pad(data1_idx, 30)
    data2_idx = vocab.truncate_pad(data2_idx, 10)
    data3_idx = vocab.truncate_pad(data3_idx, 10)
    
    data1_idx = vocab[data1_idx]
    data2_idx = vocab[data2_idx]
    data3_idx = vocab[data3_idx]
    
    print(data1_idx)
    print(data2_idx)
    print(data3_idx)
    
def test_data():
    args = get_args()
    print(args)
    vocab_mlm = create_vocab(args)
    vocab_mlm = add_tokens_to_vocab(vocab_mlm)
    padded_seq, idx_seq, idx_msa, attn_idx = get_paded_token_idx(vocab_mlm, args.train_data)
    labels = torch.tensor(idx_seq)
    idx_msa = torch.tensor(idx_msa)
    attn_idx = torch.tensor(attn_idx)
    for t in np.arange(1, 50):
        padded_seq_copy = deepcopy(padded_seq)        
        train_loader, test_loader = make_mask(padded_seq_copy, start=0, end=49, time=t, 
                                              vocab_mlm=vocab_mlm, labels=labels, idx_msa=idx_msa, attn_idx=attn_idx,
                                              test_size=args.test_size, batch_size=args.batch_size, seed=args.seed, device='cpu')
        for i, (masked_idx, label, msa, attn) in enumerate(train_loader):
            print(f"the {i}th batch is that masked_idx is {masked_idx.shape}, labels is {label.shape}, idx_msa is {msa.shape}")
        print(f"the {t}th time step is done")
    
    
if __name__ == '__main__':
    # test_vocab()
    test_data()