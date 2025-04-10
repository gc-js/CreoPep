import torch
import random
import pandas as pd
from creopep.utils import create_vocab, setup_seed
from creopep.dataset_mlm import  get_paded_token_idx_gen, add_tokens_to_vocab
import argparse

parser = argparse.ArgumentParser('Unconstrained_generation', add_help=False)

def temperature_sampling(logits, temperature):
    logits = logits / temperature
    probabilities = torch.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probabilities, 1)
    return sampled_token

def CreoPep(X1, X2, τ, g_num, start, end, model_name, seed, output):
    if seed =='random':
        seed = random.randint(0,100000)
        setup_seed(seed)
    else:
        setup_seed(int(seed))

    device = torch.device("cuda:0") 
    vocab_mlm = create_vocab()
    vocab_mlm = add_tokens_to_vocab(vocab_mlm)
    save_path = model_name
    train_seqs = pd.read_csv('./data/C0_seq.csv') # Avoid generating peptides that duplicate those in the training set.
    train_seq = train_seqs['Seq'].tolist()
    model = torch.load(save_path, weights_only=False)
    model = model.to(device)

    msa_data = pd.read_csv('./data/conoData_C0.csv') 
    msa = msa_data['Sequences'].tolist()
    msa = [x for x in msa if x.startswith(f"{X1}|{X2}")]
    msa = random.choice(msa)
    X4 = msa.split("|")[3]
    X5 = msa.split("|")[4]
    X6 = msa.split("|")[5]

    IDs = []
    generated_seqs = []
    generated_seqs_FINAL = []
    cls_probability_all = []
    act_probability_all = []

    count = 0
    gen_num = int(g_num)
    NON_AA = ["B","O","U","Z","X",'<K16>', '<α1β1γδ>', '<Ca22>', '<AChBP>', '<K13>', '<α1BAR>', '<α1β1ε>', '<α1AAR>', '<GluN3A>', '<α4β2>',
                    '<GluN2B>', '<α75HT3>', '<Na14>', '<α7>', '<GluN2C>', '<NET>', '<NavBh>', '<α6β3β4>', '<Na11>', '<Ca13>', 
                    '<Ca12>', '<Na16>', '<α6α3β2>', '<GluN2A>', '<GluN2D>', '<K17>', '<α1β1δε>', '<GABA>', '<α9>', '<K12>', 
                    '<Kshaker>', '<α3β4>', '<Na18>', '<α3β2>', '<α6α3β2β3>', '<α1β1δ>', '<α6α3β4β3>', '<α2β2>','<α6β4>', '<α2β4>',
                    '<Na13>', '<Na12>', '<Na15>', '<α4β4>', '<α7α6β2>', '<α1β1γ>', '<NaTTXR>', '<K11>', '<Ca23>', 
                    '<α9α10>','<α6α3β4>', '<NaTTXS>', '<Na17>','<high>','<low>','[UNK]','[SEP]','[PAD]','[CLS]','[MASK]']             
    while count < gen_num:
        new_seq = None

        gen_len = random.randint(int(start), int(end))
        X3 = "X" * gen_len
        seq = [f"{X1}|{X2}|{X3}|{X4}|{X5}|{X6}"]
        vocab_mlm.token_to_idx["X"] = 4

        padded_seq, _, _, _ = get_paded_token_idx_gen(vocab_mlm, seq, new_seq)
        input_text = ["[MASK]" if i=="X" else i for i in padded_seq]

        gen_length = len(input_text)
        length = gen_length - sum(1 for x in input_text if x != '[MASK]')

        for i in range(length):
            _, idx_seq, idx_msa, attn_idx = get_paded_token_idx_gen(vocab_mlm, seq, new_seq)
            idx_seq = torch.tensor(idx_seq).unsqueeze(0).to(device)
            idx_msa = torch.tensor(idx_msa).unsqueeze(0).to(device)
            attn_idx = torch.tensor(attn_idx).to(device)

            mask_positions = [j for j in range(gen_length) if input_text[j] == "[MASK]"]
            mask_position = torch.tensor([mask_positions[torch.randint(len(mask_positions), (1,))]])
            
            logits = model(idx_seq,idx_msa, attn_idx) 
            mask_logits = logits[0, mask_position.item(), :] 

            predicted_token_id = temperature_sampling(mask_logits, τ)

            predicted_token = vocab_mlm.to_tokens(int(predicted_token_id))
            input_text[mask_position.item()] = predicted_token
            padded_seq[mask_position.item()] = predicted_token.strip()
            new_seq = padded_seq

        generated_seq = input_text
    
        generated_seq[1] = "[MASK]"
        input_ids = vocab_mlm.__getitem__(generated_seq)
        logits = model(torch.tensor([input_ids]).to(device), idx_msa)
        cls_mask_logits = logits[0, 1, :]
        cls_probability, cls_mask_probs = torch.topk((torch.softmax(cls_mask_logits, dim=-1)), k=85)

        generated_seq[2] = "[MASK]"
        input_ids = vocab_mlm.__getitem__(generated_seq)
        logits = model(torch.tensor([input_ids]).to(device), idx_msa)
        act_mask_logits = logits[0, 2, :]
        act_probability, act_mask_probs = torch.topk((torch.softmax(act_mask_logits, dim=-1)), k=2)

        cls_pos = vocab_mlm.to_tokens(list(cls_mask_probs))
        act_pos = vocab_mlm.to_tokens(list(act_mask_probs))
        
        if X1 in cls_pos and X2 in act_pos:
            cls_proba = cls_probability[cls_pos.index(X1)].item()
            act_proba = act_probability[act_pos.index(X2)].item()
            generated_seq = generated_seq[generated_seq.index('[MASK]') + 2:generated_seq.index('[SEP]')]
            if generated_seq.count('C') % 2 == 0 and len("".join(generated_seq)) == gen_len:
                generated_seqs.append("".join(generated_seq))
                if "".join(generated_seq) not in train_seq and "".join(generated_seq) not in generated_seqs[0:-1] and all(x not in NON_AA for x in generated_seq):
                    generated_seqs_FINAL.append("".join(generated_seq))
                    cls_probability_all.append(cls_proba)
                    act_probability_all.append(act_proba)
                    IDs.append(count+1)
                    out = pd.DataFrame({
                        'ID':IDs,
                        'Generated_seq': generated_seqs_FINAL,
                        'Subtype': X1,
                        'Subtype_probability': cls_probability_all, 
                        'Potency': X2, 
                        'Potency_probability': act_probability_all, 
                        'Random_seed': seed
                    })
                    out.to_csv(output, index=False, encoding='utf-8-sig')
                    count += 1

if __name__ == '__main__':
    parser.add_argument('-is', '--subtype', default='<α7>', type=str, help='subtype of action. For example, α7.')
    parser.add_argument('-ip', '--potency', default='<high>', type=str, help='required potency. For example, High.')   
    parser.add_argument('-t', '--temperature', default='1', type=int, help='τ: temperature factor controls the diversity of conotoxins generated. The higher the value, the higher the diversity.')
    parser.add_argument('-n', '--num', default='10', type=int, help='Number of generations')
    parser.add_argument('-b', '--begin', default='12', type=int, help='Min length for generating peptides')
    parser.add_argument('-e', '--end', default='16', type=int, help='Max length for generating peptides')
    parser.add_argument('-m', '--model', default='./models/model_final.pt', type=str, help='Model: model parameters trained at different stages of data augmentation.')
    parser.add_argument('-s', '--seed', default='random', help='Seed: enter an integer as the random seed to ensure reproducible results. The default is random.')
    parser.add_argument('-o', '--output', default='./test/output_unconstrained_generation.csv', help='output file')
    args = parser.parse_args()
    CreoPep(args.subtype, args.potency, args.temperature, args.num, args.begin, args.end, args.model, args.seed, args.output)
