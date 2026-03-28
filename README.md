# A deep learning framework (CreoPep) for target-specific design and optimization of conotoxin peptides
<a href="https://www.nature.com/articles/s42004-025-01885-5"><img src="https://img.shields.io/badge/Paper-communications chemistry-orange" style="max-width: 100%;"></a>
<a href="https://huggingface.co/spaces/oucgc1996/CreoPep"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-red?label=Web" style="max-width: 100%;"></a>

## Overview

<img src="https://github.com/gc-js/CreoPep/blob/main/imgs/Fig2.png" alt="workflow" width="400"/>

## :desktop_computer: Web
We also host a trained version of the model on the HuggingFace Spaces, so you can **start your inference using just your browser**.

🤗[Label prediction](https://huggingface.co/spaces/oucgc1996/CTXGen_Label_Prediction)

🤗[Unconstrained generation](https://huggingface.co/spaces/oucgc1996/CTXGen_Unconstrained_generation)

🤗[Conditional generation](https://huggingface.co/spaces/oucgc1996/CTXGen_conditional_generation)

🤗[Optimization generation](https://huggingface.co/spaces/oucgc1996/CTXGen_optimization_generation)



## :gear: Installation

```shell
git clone git@github.com:gc-js/CreoPep.git
cd CreoPep
conda create -n CreoPep python=3.10.14
conda activate CreoPep
pip install -e .
```

You can download the trained models [here](https://zenodo.org/records/15192592) and save them to `models`.

```bash
bash download.sh
```

## :rocket: Get Started

#### :one: Label_Prediction

```bash
python label_prediction.py -i ./test/ctxs.txt -is X -ip X -m ./models/model_final.pt -o ./test/output_label_prediction.csv
```

- `-i`: conotoxins need to be predicted.


- `-is`: Subtype: X if needs to be predicted.

    optional: `<AChBP>`, `<Ca12>`, `<Ca13>`, `<Ca22>`, `<Ca23>`, `<GABA>`, `<GluN2A>`, `<GluN2B>`, `<GluN2C>`, `<GluN2D>`,
   `<GluN3A>`, `<K11>`, `<K12>`, `<K13>`, `<K16>`, `<K17>`, `<Kshaker>`, `<Na11>`, `<Na12>`, `<Na13>`, `<Na14>`, `<Na15>`,
   `<Na16>`, `<Na17>`, `<Na18>`, `<NaTTXR>`, `<NaTTXS>`, `<NavBh>`, `<NET>`, `<α1AAR>`, `<α1BAR>`, `<α1β1γ>`, `<α1β1γδ>`,
  `<α1β1δ>`, `<α1β1δε>`, `<α1β1ε>`, `<α2β2>`, `<α2β4>`, `<α3β2>`, `<α3β4>`, `<α4β2>`, `<α4β4>`, `<α6α3β2>`, `<α6α3β2β3>`,
  `<α6α3β4>`, `<α6α3β4β3>`, `<α6β3β4>`, `<α6β4>`, `<α7>`, `<α7α6β2>`, `<α75HT3>`, `<α9>`, `<α9α10>`

- `-ip`: Potency: X if needs to be predicted.

    optional: `<high>`, `<low>`


- `-m`: model parameters trained at different stages of data augmentation.

- `-o`: output file (.csv)

#### :two: Unconstrained Generation

```bash
python unconstrained_generation.py -t 1 -n 100 -b 12 -e 16 -m ./models/model_final.pt -s 666 -o ./test/output_unconstrained_generation.csv
```
- `-t`: temperature factor (τ) controls the diversity of conotoxins generated. The higher the value, the higher the diversity.
- `-n`: Number of generations: if it is not completed within 1200 seconds, it will automatically stop.
- `-b`: Min length for generating peptides.
- `-e`: Max length for generating peptides.
- `-m`: model parameters trained at different stages of data augmentation.
- `-s`: Seed: enter an integer as the random seed to ensure reproducible results. The default is random.
- `-o`: output file (.csv)

#### :three: Conditional Generation

```bash
python conditional_generation.py -is "<α7>" -ip "<high>" -t 1 -n 100 -b 12 -e 16 -m ./models/model_final.pt -s 666 -o ./test/output_conditional_generation.csv
```
- `-is`: subtype of action. For example, `<α7>`.
- `-ip`: required potency. For example, `<high>`.
- `-t`: temperature factor (τ) controls the diversity of conotoxins generated. The higher the value, the higher the diversity.
- `-n`: Number of generations.
- `-b`: Min length for generating peptides.
- `-e`: Max length for generating peptides.
- `-m`: model parameters trained at different stages of data augmentation.
- `-s`: Seed: enter an integer as the random seed to ensure reproducible results. The default is random.
- `-o`: output file (.csv)

#### :four: Optimization Generation

```bash
python optimization_generation.py -i GCCSDPRCAWRC -x GCCXXXXCAWRC -is "<α7>" -ip "<high>" -t 1 -n 100 -m ./models/model_final.pt -s 666 -o ./test/output_optimization_generation.csv
```
- `-i`: a conotoxin that needs to be optimized. For example, GCCSDPRCAWRC.
- `-x`: the positions that need to be optimized, replaced by X. For example, GCCXXXXCAWRC.
- `-is`: subtype of action. For example, `<α7>`.
- `-ip`: required potency. For example, `<high>`.
- `-t`: temperature factor (τ) controls the diversity of conotoxins generated. The higher the value, the higher the diversity.
- `-n`: Number of generations.
- `-m`: model parameters trained at different stages of data augmentation.
- `-s`: Seed: enter an integer as the random seed to ensure reproducible results. The default is random.
- `-o`: output file (.csv)

## :computer: Model Training

#### :one: Data processing

```bash
python ./analysis/data_processing.py -i ./data/conoData5.csv -o ./data/conoData_C5.csv
```
- `-i`: Input data, the raw training data with column names: Seq, Target, Potency (.csv)

- `-o`: Output data, the processed data for CreoPep training (.csv)

#### :two: Training
```bash
python train.py --train_data ./data/conoData_C5.csv --model_save_path ./models/best_model.pt --loss_save_path ./imgs/Loss_curves.png --PLM Rostlab/prot_bert --time_step 27 --epochs 100 --batch_size 128 --test_size 0.1 --lr 5e-5 --vocab ./data/vocab.txt --device cuda:0 --seed 42
```

- `--train_data`: Train data, the raw training data with column names: Sequences. (.csv)
- `--model_save_path`: Save path for model.
- `--loss_save_path`: Loss curves for model.
- `--PLM`: Protein language model.
- `--PLM_config`: PLM config. You can modify it [here](https://github.com/gc-js/CreoPep/blob/main/models/PLM_config.json).
- `--vocab`: Vocab file path. You can modify it [here](https://github.com/gc-js/CreoPep/blob/main/data/vocab.txt).
- `--time_step`: Time step to use.
- `--epochs`: Number of epochs for training.
- `--batch_size`: Batch size for training.
- `--test_size`: Proportion of test sets.
- `--lr`: Learning rate for training.
- `--device`: Device to use for training.
- `--seed`: Random seed.

#### :three: Data augmentation

- You can obtain Foldx for free upon registration [here](https://foldxsuite.crg.eu/).

- And, install pyfoldx [here](https://github.com/leandroradusky/pyfoldx).

- Then, running:

```bash
python ./data_augmentation/foldx.py --pdb_path ./data_augmentation/pdb/a7/a7.pdb --mutants ./data_augmentation/pdb/a7/output_a7.csv --task a7 --output ./data_augmentation/pdb/a7/foldx_a7_out.csv
```

- `--pdb_path`: Path of wild type peptide pdb file.
- `--mutants`: Path of mutants sequence file with column names: generated_seq (.csv).
- `--task`: Name of task.
- `--output`: Path of output file.

#### :four: Calculate ΔG

- Place the AlphaFold3-predicted complex structures in the base folder, e.g., [foldx_a9a10](https://github.com/gc-js/CreoPep/tree/main/analysis/foldx_a9a10).
- Convert the cif to pdb.

```bash
python ./analysis/cif2pdb.py -base_path ./analysis/foldx_a9a10
```

- Renumber the pdb to match the FoldX input format.

```bash
python ./analysis/pdb4foldx.py -base_path ./analysis/foldx_a9a10
```
  
- Run FoldX to calculate the ΔG of all complexes.

```bash
bash ./analysis/run.sh -d foldx_a9a10 -s Interaction_Energy.py -t ./analysis/foldx_a9a10
```

#### 💥 *Note*:

- Put FoldX executable into /usr/bin/

- Add the following line to your ~/.bashrc:
```bash
export FOLDX_BINARY=/usr/bin/foldx
export PATH=$PATH:/usr/bin/foldx
```

- Add permission.
```bash
sudo chmod 777 /usr/bin/foldx
```

## Reference

```bash
Ge, C., Tae, HS., Lu, L. et al. A deep learning framework (CreoPep) for target-specific
design and optimization of conotoxin peptides. Commun Chem (2026).
https://doi.org/10.1038/s42004-025-01885-5
```
