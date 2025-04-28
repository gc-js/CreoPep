from Bio.PDB import MMCIFParser, PDBIO
import os
import argparse

parser = argparse.ArgumentParser('foldx', add_help=False)
parser.add_argument('-base_path', default='./foldx_a9a10', required=True, type=str, help='base_path')
args = parser.parse_args()

cases = os.listdir(args.base_path)
for case in cases:
    files = os.listdir(os.path.join(args.base_path,case))
    for file in files:
        if file.split(".")[1] == "cif":
            pdb_name = file.split(".")[0]
            input_cif_filename = os.path.join(args.base_path, case, file)
            output_pdb_filename = os.path.join(args.base_path, case, pdb_name + ".pdb")
            parser = MMCIFParser()
            structure = parser.get_structure("Protein", input_cif_filename)

            io = PDBIO()
            io.set_structure(structure)
            io.save(output_pdb_filename)
