from pyfoldx.structure import Structure
import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pdb_path', type=str, help='Path of wild type peptide pdb file', required=True)
parser.add_argument('--mutants', type=str, help='Path of mutants sequence file with column names: generated_seq (.csv)', required=True)
parser.add_argument('--task', type=str, help='Name of task', required=True)
parser.add_argument('--output', type=str, help='Path of output file', required=True)

args = parser.parse_args()    

st = Structure(f"{args.task}", args.pdb_path)

receptor_seq = st.getSequence(chain="A")
receptor_residues = [f"{receptor_seq[i]}A{i+1}" for i in range(len(receptor_seq))]
print(receptor_seq)
st = st.repair()

ori_Interaction_Energy = st.getInterfaceEnergy()['Interaction Energy'][('A', 'C')]
print(float(ori_Interaction_Energy))

ori_seq = st.getSequence(chain="C")
path = args.mutants
df = pd.read_csv(path)
mutate_seq = df['generated_seq'].tolist()

saved_seq = []
saved_value = []
for i in mutate_seq:
    start_time = time.time()
    diff = [j for j in range(len(ori_seq)) if ori_seq[j] != i[j]]
    mutate_list = [f"{ori_seq[j]}C{j+1}{i[j]};" for j in diff]

    current_structure = st

    for mutation in mutate_list:
        ddGsDf, mutated_structure, trajWT = current_structure.mutate(mutation, 1)
        current_structure = mutated_structure.getFrame(0)
    current_structure = current_structure.repair(fix_residues=receptor_residues)

    mutate_Interaction_Energy = current_structure.getInterfaceEnergy()['Interaction Energy'][('A', 'C')]

    value = float(mutate_Interaction_Energy)-float(ori_Interaction_Energy)

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Repair operation took {execution_time:.2f} seconds.")

    saved_seq.append(i)
    saved_value.append(value)
    df = pd.DataFrame({'Seq': saved_seq, 'Value': saved_value})
    df.to_csv(args.output, index=False)
