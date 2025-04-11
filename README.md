# CreoPep
This is the official repository for "CreoPep: A Universal Deep Learning Framework for Target-Specific Peptide Design and Optimization."

## ![Huggingface](https://img.shields.io/badge/Hugging%20Face-Spaces-brightgreen)
We also host a trained version of the model on the HuggingFace Spaces, so you can start your inference using just your browser.

🤗[Label prediction](https://huggingface.co/spaces/oucgc1996/CTXGen_Label_Prediction)

🤗[Unconstrained generation](https://huggingface.co/spaces/oucgc1996/CTXGen_Unconstrained_generation)

🤗[Conditional generation](https://huggingface.co/spaces/oucgc1996/CTXGen_conditional_generation)

🤗[Optimization generation](https://huggingface.co/spaces/oucgc1996/CTXGen_optimization_generation)



## :gear: Installation

```shell
git clone git@github.com:gc-js/CreoPep.git
cd CreoPep
conda create -n CreoPep python=3.10.14
python -m pip install -r requirements.txt
```

## :rocket: Get Started

You can download the trained models [here](https://zenodo.org/records/15192592) and save them to `models`.

#### :one: Label_Prediction

```bash
python label_prediction.py -i ./test/ctxs.txt -is X -ip X -m ./models/model_final.pt -o ./test/output_label_prediction.csv
```

- `-i`: conotoxins need to be predicted.


- `-is`: Subtype: X if needs to be predicted.

    optional: `<K16>`, `<α1β1γδ>`, `<Ca22>`, `<AChBP>`, `<K13>`, `<α1BAR>`, `<α1β1ε>`, `<α1AAR>`, `<GluN3A>`, `<α4β2>`,
`<GluN2B>`, `<α75HT3>`, `<Na14>`, `<α7>`, `<GluN2C>`, `<NET>`, `<NavBh>`, `<α6β3β4>`, `<Na11>`, `<Ca13>`,
`<Ca12>`, `<Na16>`, `<α6α3β2>`, `<GluN2A>`, `<GluN2D>`, `<K17>`, `<α1β1δε>`, `<GABA>`, `<α9>`, `<K12>`,
`<Kshaker>`, `<α3β4>`, `<Na18>`, `<α3β2>`, `<α6α3β2β3>`, `<α1β1δ>`, `<α6α3β4β3>`, `<α2β2>`, `<α6β4>`, `<α2β4>`,
`<Na13>`, `<Na12>`, `<Na15>`, `<α4β4>`, `<α7α6β2>`, `<α1β1γ>`, `<NaTTXR>`, `<K11>`, `<Ca23>`,
`<α9α10>`, `<α6α3β4>`, `<NaTTXS>`, `<Na17>`

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

```bash
python train.py
```

