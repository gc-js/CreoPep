import os
import pandas as pd
import random
import csv

def process_stage1():
    """Process data stage 1: Create FASTA files from input CSV"""
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

def process_stage2():
    """Process data stage 2: Select random sequences from FASTA files"""
    input_path = "./data/data_process_s1"
    output_path = "./data/data_process_s2"
    dirs = os.listdir(input_path)
    
    os.makedirs(output_path, exist_ok=True)

    for dir in dirs:
        dir_name = dir.split('.')[0]
        mafft_input = os.path.join(input_path, dir)

        with open(mafft_input, 'r') as file:
            lines = file.readlines()

        sequences = []
        current_sequence = ""

        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = ""
            else:
                current_sequence += line

        if current_sequence:
            sequences.append(current_sequence)

        first_sequence = sequences[0]
        num_sequences = len(sequences) - 1

        selected_sequences = [first_sequence]
        
        if num_sequences >= 3:
            selected_sequences.extend(random.sample(sequences[1:], 3))
        else:
            selected_sequences.extend(sequences[1:] if sequences[1:] else [])
            while len(selected_sequences) < 4:
                selected_sequences.append("" * len(first_sequence))

        output_file_path = os.path.join(output_path, f"{dir_name}.fasta")
        with open(output_file_path, 'w') as output_file:
            for i, seq in enumerate(selected_sequences):
                output_file.write(f">seq{i + 1}\n{seq}\n")

def process_stage3():
    """Process data stage 3: Create final CSV from processed FASTA files"""
    input_path = "./data/data_process_s2"
    output_path = "./data/data_process_s3"
    dirs = os.listdir(input_path)

    csv_output_file = os.path.join(output_path, "conoData_final.csv")

    os.makedirs(output_path, exist_ok=True)

    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Sequences'])

        for dir in dirs:
            dir_target = dir.split('_')[1]
            dir_value = (dir.split('_')[2]).split('.')[0]
            fasta_input = os.path.join(input_path, dir)

            with open(fasta_input, 'r') as file:
                lines = file.readlines()

                sequences = []
                for i in range(1, len(lines), 2):
                    sequence = lines[i].strip()
                    
                    if len(sequences) == 0:
                        sequence = sequence.replace('-', '')
                    
                    sequences.append(sequence)

                result = f"<{dir_target}>|<{dir_value}>|{'|'.join(sequences)}"
                csv_writer.writerow([result])

def main():
    """Main function to execute all processing stages"""
    print("Starting stage 1 processing...")
    process_stage1()
    
    print("Starting stage 2 processing...")
    process_stage2()
    
    print("Starting stage 3 processing...")
    process_stage3()
    
    print("All processing completed!")

if __name__ == "__main__":
    main()
