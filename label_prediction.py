import torch
from creopep.utils import create_vocab, setup_seed
from creopep.dataset_mlm import  get_paded_token_idx_gen, add_tokens_to_vocab
import argparse
import pandas as pd
from tqdm import tqdm

def CreoPep(ctxs, X1, X2, output):
    with open(ctxs, 'r') as f:
        lines = f.readlines()
    Seq_all  = []
    subtype_all = []
    potency_all  = []
    topk_all = []
    subtype_probability_all = []
    potency_probability_all = []

    pbar = tqdm(lines, desc="Processing sequences")
    
    for X3 in lines:
        predicted_token_probability_all = []
        topk = []
        new_seq = None
        seq = [f"{X1}|{X2}|{X3}|||"]
        vocab_mlm.token_to_idx["X"] = 4
        padded_seq, _, idx_msa, _ = get_paded_token_idx_gen(vocab_mlm, seq, new_seq)
        idx_msa = torch.tensor(idx_msa).unsqueeze(0).to(device)
        mask_positions = [i for i, token in enumerate(padded_seq) if token == "X"]
        if not mask_positions:
            raise ValueError("Nothing found in the sequence to predict.")
        for mask_position in mask_positions:
            padded_seq[mask_position] = "[MASK]"
            input_ids = vocab_mlm.__getitem__(padded_seq)
            input_ids = torch.tensor([input_ids]).to(device)
            logits = model(input_ids, idx_msa)
            mask_logits = logits[0, mask_position, :]
            predicted_token_probability, predicted_token_id = torch.topk((torch.softmax(mask_logits, dim=-1)), k=5)
            topk.append(predicted_token_id)
            predicted_token = vocab_mlm.idx_to_token[predicted_token_id[0].item()]
            predicted_token_probability_all.append(predicted_token_probability[0].item())
            padded_seq[mask_position] = predicted_token

        cls_pos = vocab_mlm.to_tokens(list(topk[0]))
        if X1 != "X":
            Topk = X1
            Subtype = X1
            Potency = padded_seq[2]
            Potency_probability = predicted_token_probability_all[0]
        elif X2 != "X":
            Topk = cls_pos
            Subtype = padded_seq[1]
            Subtype_probability = predicted_token_probability_all[0]
            Potency = X2
        else:
            Topk = cls_pos
            Subtype = padded_seq[1]
            Subtype_probability = predicted_token_probability_all[0]
            Potency = padded_seq[2]
            Potency_probability = predicted_token_probability_all[1]
        Seq_all.append(X3.strip())
        subtype_all.append(Subtype)
        potency_all.append(Potency)
        topk_all.append(Topk)
        subtype_probability_all.append(Subtype_probability)
        potency_probability_all.append(Potency_probability)
        pbar.set_postfix({
            "Processed": len(Seq_all),
            "Current Seq": X3.strip()[:10] + "..." if len(X3) > 10 else X3.strip()
        })
        data = {'Sequence': Seq_all, 'Subtype': subtype_all, 'Subtype_probability': subtype_probability_all, 'Potency': potency_all, 'Potency_probability':potency_probability_all, 'Topk': topk_all}
        df = pd.DataFrame(data)
        df.to_csv(output, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Label Prediction', add_help=False)
    
    parser.add_argument('-i', '--ctxs', default='./test/ctx.txt', required=True, type=str, help='Conotoxins: conotoxins need to be predicted.')
    parser.add_argument('-is', '--subtype', default='X', type=str, help='Subtype: X if needs to be predicted.')
    parser.add_argument('-ip', '--potency', default='X', type=str, help='Potency: X if needs to be predicted.')
    parser.add_argument('-m', '--model', default='./models/model_final.pt', type=str, help='Model: model parameters trained at different stages of data augmentation.')
    parser.add_argument('-o', '--output', default='./test/output_label_prediction.csv', help='output file')
    args = parser.parse_args()
    setup_seed(4)
    device = torch.device("cuda:0")
    vocab_mlm = create_vocab()
    vocab_mlm = add_tokens_to_vocab(vocab_mlm)
    save_path = args.model
    model = torch.load(save_path, weights_only=False)
    model = model.to(device)
    CreoPep(args.ctxs, args.subtype, args.potency, args.output)
