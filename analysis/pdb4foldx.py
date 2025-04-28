from Bio.PDB import PDBParser, PDBIO
import os
def modify_pdb(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line in lines:
            modified_line = line.replace(' B ', ' A ')
            f.write(modified_line)

def renumber_chains(pdb_file, output_file):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('structure', pdb_file)

    for model in structure:
        new_res_id = 1
        for chain in model:
            if chain.id in ['A', 'B']:
                for residue in chain:
                    residue.id = (residue.id[0], new_res_id, residue.id[2])
                    new_res_id += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)

if __name__ == "__main__":
    dir_path = "./cif"
    files = os.listdir(dir_path)
    for file in files:
        if file.split(".")[1] == "pdb":
            pdb_file = os.path.join(dir_path, file)
            output_file = os.path.join(dir_path, file)
            renumber_chains(pdb_file, output_file)
            modify_pdb(pdb_file, output_file)
