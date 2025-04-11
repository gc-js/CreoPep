from pyfoldx.structure import Structure
import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str, help='Path to the directory containing the data files', required=True)
parser.add_argument('--task', type=str, help='Path to the directory containing the data files', required=True)
args = parser.parse_args()

st = Structure(f"{args.task}", f"{args.path}/{args.task}.pdb")

receptor_seq = st.getSequence(chain="A")
receptor_residues = [f"{receptor_seq[i]}A{i+1}" for i in range(len(receptor_seq))]
print(receptor_seq)
st = st.repair()

ori_Interaction_Energy = st.getInterfaceEnergy()['Interaction Energy'][('A', 'C')]
print(float(ori_Interaction_Energy))

ori_seq = st.getSequence(chain="C")
path = f'{args.path}/output_{args.task}.csv'
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
    df.to_csv(f"{args.path}/foldx_output_{args.task}.csv", index=False)
