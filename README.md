# CreoPep
This is the official repository for "CreoPep: A Universal Deep Learning Framework for Target-Specific Peptide Design and Optimization."

## ![Huggingface](https://img.shields.io/badge/Hugging%20Face-Spaces-brightgreen)
We also host a trained version of the model on the HuggingFace Spaces, so you can start your inference using just your browser.

🤗[Label_Prediction](https://huggingface.co/spaces/oucgc1996/CTXGen_Label_Prediction)

🤗[Unconstrained_generation](https://huggingface.co/spaces/oucgc1996/CTXGen_Unconstrained_generation)

🤗[Conditional_generation](https://huggingface.co/spaces/oucgc1996/CTXGen_conditional_generation)

🤗[Optimization_generation](https://huggingface.co/spaces/oucgc1996/CTXGen_optimization_generation)



## Installation

```shell
git clone git@github.com:gc-js/CreoPep.git
cd CreoPep
conda create -n CreoPep python=3.10.14
python -m pip install -r requirements.txt
```

## Get Started

#### Label_Prediction

```bash
python label_prediction.py -i ./test/ctxs.txt -is X -ip X -m ./models/mlm-model-27.pt
```

- `-i`: conotoxins need to be predicted.

    optional: `<K16>`, `<α1β1γδ>`, `<Ca22>`, `<AChBP>`, `<K13>`, `<α1BAR>`, `<α1β1ε>`, `<α1AAR>`, `<GluN3A>`, `<α4β2>`,
`<GluN2B>`, `<α75HT3>`, `<Na14>`, `<α7>`, `<GluN2C>`, `<NET>`, `<NavBh>`, `<α6β3β4>`, `<Na11>`, `<Ca13>`,
`<Ca12>`, `<Na16>`, `<α6α3β2>`, `<GluN2A>`, `<GluN2D>`, `<K17>`, `<α1β1δε>`, `<GABA>`, `<α9>`, `<K12>`,
`<Kshaker>`, `<α3β4>`, `<Na18>`, `<α3β2>`, `<α6α3β2β3>`, `<α1β1δ>`, `<α6α3β4β3>`, `<α2β2>`, `<α6β4>`, `<α2β4>`,
`<Na13>`, `<Na12>`, `<Na15>`, `<α4β4>`, `<α7α6β2>`, `<α1β1γ>`, `<NaTTXR>`, `<K11>`, `<Ca23>`,
`<α9α10>`, `<α6α3β4>`, `<NaTTXS>`, `<Na17>`

- `-is`: Subtype: X if needs to be predicted.

    optional: `<high>`, `<low>`

- `-ip`: Potency: X if needs to be predicted.
- `-m`: model parameters trained at different stages of data augmentation.

#### Unconstrained Generation

```bash
python label_prediction.py -i ./test/ctxs.txt -is X -ip X -m ./models/mlm-model-27.pt
```

- `-i`: conotoxins need to be predicted.

    optional: `<K16>`, `<α1β1γδ>`, `<Ca22>`, `<AChBP>`, `<K13>`, `<α1BAR>`, `<α1β1ε>`, `<α1AAR>`, `<GluN3A>`, `<α4β2>`,
`<GluN2B>`, `<α75HT3>`, `<Na14>`, `<α7>`, `<GluN2C>`, `<NET>`, `<NavBh>`, `<α6β3β4>`, `<Na11>`, `<Ca13>`,
`<Ca12>`, `<Na16>`, `<α6α3β2>`, `<GluN2A>`, `<GluN2D>`, `<K17>`, `<α1β1δε>`, `<GABA>`, `<α9>`, `<K12>`,
`<Kshaker>`, `<α3β4>`, `<Na18>`, `<α3β2>`, `<α6α3β2β3>`, `<α1β1δ>`, `<α6α3β4β3>`, `<α2β2>`, `<α6β4>`, `<α2β4>`,
`<Na13>`, `<Na12>`, `<Na15>`, `<α4β4>`, `<α7α6β2>`, `<α1β1γ>`, `<NaTTXR>`, `<K11>`, `<Ca23>`,
`<α9α10>`, `<α6α3β4>`, `<NaTTXS>`, `<Na17>`

- `-is`: Subtype: X if needs to be predicted.

    optional: `<high>`, `<low>`

- `-ip`: Potency: X if needs to be predicted.
- `-m`: model parameters trained at different stages of data augmentation. 

## Model Training

