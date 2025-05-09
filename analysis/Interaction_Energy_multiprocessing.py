from pyfoldx.structure import Structure
import pandas as pd
import time
import argparse
import os
from multiprocessing import Pool
import functools

def process_pdb(pdb, task, path):
    print(f"Processing {pdb}")
    try:
        st = Structure(task, f"{path}/{pdb}")
        st = st.repair()
        ori_Interaction_Energy = st.getInterfaceEnergy()['Interaction Energy'][('A', 'C')]
        print(f"Successfully processed {pdb}: {ori_Interaction_Energy}")
        return (pdb, float(ori_Interaction_Energy))
    except Exception as e:
        print(f"Error processing {pdb}: {e}")
        return (pdb, None)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to the directory containing the data files', required=True)
    parser.add_argument('--task', type=str, help='task', required=True)
    parser.add_argument('--num', type=str, help='num', required=True)
    args = parser.parse_args()

    case_name = os.path.basename(os.path.normpath(args.path))
    pdbs = [f for f in os.listdir(args.path) if f.endswith('.pdb')]
    pdbs.sort()
    pdbs = pdbs[:5]

    with Pool() as pool:
        process_func = functools.partial(process_pdb, task=args.task, path=args.path)
        results = pool.map(process_func, pdbs)

    results_dict = {'case_name': case_name}
    for i, (pdb, energy) in enumerate(results, 1):
        results_dict[f'af{i}'] = energy
    
    for i in range(len(results) + 1, 6):
        results_dict[f'af{i}'] = None

    df = pd.DataFrame([results_dict])
    if not os.path.exists('results.csv'):
        df.to_csv('results.csv', index=False)
    else:
        df.to_csv('results.csv', mode='a', header=False, index=False)

    print(f"Results for {case_name} written to results.csv")

if __name__ == '__main__':
    main()
