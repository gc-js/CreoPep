import os
import pandas as pd

data = pd.read_csv("./data/conoData5.csv")
output_dir = "./data/data_process_s1/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

seq = data["Seq"].tolist()
target = data["Target"].tolist()
active = data["Potency"].tolist()
same_value = []

for i in range(len(seq)):
    temp = [] 
    for j in range(len(seq)):
        if i != j and target[i] == target[j] and active[i] == active[j]:
            temp.append(seq[j])
    same_value.append((seq[i], target[i], active[i], temp))

for idx, seq_pair in enumerate(same_value):
    fasta_path = os.path.join(output_dir, f"seq{idx+1}_{seq_pair[1]}_{seq_pair[2]}.fasta")
    
    with open(fasta_path, "w") as fasta_file:
        fasta_file.write(f">1\n")
        fasta_file.write(f"{seq_pair[0]}\n")
        for match_idx, match in enumerate(seq_pair[3], start=2):
            fasta_file.write(f">{match_idx}\n")
            fasta_file.write(f"{match}\n")

