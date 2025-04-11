import os
import csv

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
