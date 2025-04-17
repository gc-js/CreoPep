import os
import pandas as pd
import random
import csv
import argparse

def get_base_directory(csv_path):
    """Get the base directory from the input CSV path"""
    return os.path.dirname(os.path.abspath(csv_path))

def process_stage1(input_csv, output_dir):
    """Process data stage 1: Create FASTA files from input CSV"""
    data = pd.read_csv(input_csv)
    output_dir = os.path.join(output_dir, "data_process_s1")

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

def process_stage2(base_dir):
    """Process data stage 2: Select random sequences from FASTA files"""
    input_path = os.path.join(base_dir, "data_process_s1")
    output_path = os.path.join(base_dir, "data_process_s2")
    
    os.makedirs(output_path, exist_ok=True)

    dirs = [f for f in os.listdir(input_path) if f.endswith('.fasta')]
    
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

def process_stage3(base_dir, final_output_csv):
    """Process data stage 3: Create final CSV from processed FASTA files"""
    input_path = os.path.join(base_dir, "data_process_s2")
    output_path = os.path.join(base_dir, "data_process_s3")

    # Use the user-specified output CSV path
    csv_output_file = final_output_csv if final_output_csv else os.path.join(base_dir, "output.csv")

    with open(csv_output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Sequences'])

        dirs = [f for f in os.listdir(input_path) if f.endswith('.fasta')]
        
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
    parser = argparse.ArgumentParser(description='Process Data through three stages')
    parser.add_argument('-i', '--input', required=True, help='Path to input CSV file')
    parser.add_argument('-o', '--output', help='Path to output final CSV file (optional)')
    args = parser.parse_args()

    # Get base directory from input CSV path
    base_dir = get_base_directory(args.input)

    print("Starting stage 1 processing...")
    process_stage1(args.input, base_dir)
    
    print("Starting stage 2 processing...")
    process_stage2(base_dir)
    
    print("Starting stage 3 processing...")
    process_stage3(base_dir, args.output)

if __name__ == "__main__":
    main()
