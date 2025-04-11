import os
import random

input_path = "./data/data_process_s1"
output_path = "./data/data_process_s2"
dirs = os.listdir(input_path)

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
