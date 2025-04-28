from Bio.PDB import MMCIFParser, PDBIO
import os

base_path = r"./foldx_a9a10"
cases = os.listdir(base_path)
for case in cases:
    files = os.listdir(os.path.join(base_path,case))
    for file in files:
        if file.split(".")[1] == "cif":
            pdb_name = file.split(".")[0]
            input_cif_filename = os.path.join(base_path, case, file)
            output_pdb_filename = os.path.join(base_path, case, pdb_name + ".pdb")
            parser = MMCIFParser()
            structure = parser.get_structure("Protein", input_cif_filename)

            io = PDBIO()
            io.set_structure(structure)
            io.save(output_pdb_filename)
