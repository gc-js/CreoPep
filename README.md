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

```bash
docker container run --gpus "device=0" -m 28G --name teamname --rm -v $PWD/CellSeg_Test/:/workspace/inputs/ -v $PWD/teamname_seg/:/workspace/outputs/ teamname:latest /bin/bash -c "sh predict.sh"
```

- `--gpus`: specify the available GPU during inference
## Get Started

## Model Training

