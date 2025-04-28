from Bio.PDB import MMCIFParser, PDBIO
import os

dir_path = r"./cif"
files = os.listdir(dir_path)

for file in files:
    if file.split(".")[1] == "cif":
        pdb_name = file.split(".")[0]
        input_cif_filename = os.path.join(dir_path, file)
        output_pdb_filename = os.path.join(dir_path, pdb_name + ".pdb")

        parser = MMCIFParser()
        structure = parser.get_structure("Protein", input_cif_filename)

        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb_filename)
