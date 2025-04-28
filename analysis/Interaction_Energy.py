from pyfoldx.structure import Structure
import pandas as pd
import time
import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str, help='Path to the directory containing the data files', required=True)
parser.add_argument('--task', type=str, help='task', required=True)
parser.add_argument('--num', type=str, help='num', required=True)
args = parser.parse_args()

case_name = os.path.basename(os.path.normpath(args.path))

results = {'case_name': case_name}

pdbs = [f for f in os.listdir(args.path) if f.endswith('.pdb')]
pdbs.sort()

for i, pdb in enumerate(pdbs[:5], 1):
    print(pdb)
    try:
        st = Structure(f"{args.task}", f"{args.path}/{pdb}")
        st = st.repair()
        ori_Interaction_Energy = st.getInterfaceEnergy()['Interaction Energy'][('A', 'C')]
        print(ori_Interaction_Energy)
        results[f'af{i}'] = float(ori_Interaction_Energy)
    except Exception as e:
        print(f"Error processing {pdb}: {e}")
        results[f'af{i}'] = None

for i in range(len(results), 5):
    results[f'af{i+1}'] = None

df = pd.DataFrame([results])

if not os.path.exists('results.csv'):
    df.to_csv('results.csv', index=False)
else:
    df.to_csv('results.csv', mode='a', header=False, index=False)

print(f"Results for {case_name} written to results.csv")
