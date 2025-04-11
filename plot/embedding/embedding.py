import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import umap.umap_ as umap
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModelForSequenceClassification,set_seed
reducer = umap.UMAP()
import warnings

warnings.filterwarnings('ignore')
model_checkpoint = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=320)

def setup_seed(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(4)
df = pd.read_csv('a7_rand_c5_high_low_embedding.csv')
seq = df["generated_seq"].tolist()
inputs = tokenizer(seq, return_tensors="pt",padding=True,max_length=50)

with torch.no_grad():
    logits = model(**inputs).logits

projection = umap.UMAP(n_components=2, n_neighbors=25, random_state=42).fit(logits)
train_proj_emb = projection.transform(logits)

colors = ['#ff8066','#00c9a7','#2c73d2','#926c00','#845ec2','#b0a8b9','#ffc75f','pink','#D0D0D0',"green","red"]

def plot_embedding():
    fig, ax = plt.subplots(figsize=(6, 6))
    # plt.scatter(train_proj_emb[0:1000,0], train_proj_emb[0:1000, 1], s=20, alpha=1, c=colors[8], edgecolors='grey')   # C1
    # plt.scatter(train_proj_emb[1000:2000,0], train_proj_emb[1000:2000, 1], s=20, alpha=1, c=colors[8], edgecolors='grey')  # C2
    # plt.scatter(train_proj_emb[2000:3000,0], train_proj_emb[2000:3000, 1], s=20, alpha=1, c=colors[8], edgecolors='grey')  # C3
    # plt.scatter(train_proj_emb[3000:4000,0], train_proj_emb[3000:4000, 1], s=20, alpha=1, c=colors[8], edgecolors='grey')  # C4
    # plt.scatter(train_proj_emb[4000:5000,0], train_proj_emb[4000:5000, 1], s=20, alpha=1, c=colors[8], edgecolors='grey') # C5
    # plt.scatter(train_proj_emb[5000:6000,0], train_proj_emb[5000:6000, 1], s=20, alpha=1, c=colors[8], edgecolors='grey') # rand
    plt.scatter(train_proj_emb[6000:7000,0], train_proj_emb[6000:7000, 1], s=20, alpha=1, c=colors[8], edgecolors='grey')  # final

    # achbp
    # plt.scatter(train_proj_emb[7000:7007,0], train_proj_emb[7000:7007, 1], s=100, alpha=1, c=colors[10], edgecolors='k')
    # plt.scatter(train_proj_emb[7007:7015,0], train_proj_emb[7007:7015, 1], s=100, alpha=1, c=colors[9], edgecolors='k')
    # plt.scatter(train_proj_emb[7014:7015,0], train_proj_emb[7014:7015, 1], s=500, alpha=1, c=colors[0], edgecolors='k')

    # ca22
    # plt.scatter(train_proj_emb[7000:7039,0], train_proj_emb[7000:7039, 1], s=100, alpha=1, c=colors[10], edgecolors='k')
    # plt.scatter(train_proj_emb[7039:7044,0], train_proj_emb[7039:7044, 1], s=100, alpha=1, c=colors[9], edgecolors='k')
    # plt.scatter(train_proj_emb[7006:7007,0], train_proj_emb[7006:7007, 1], s=500, alpha=1, c=colors[1], edgecolors='k')

    # a3b4
    # plt.scatter(train_proj_emb[7000:7055,0], train_proj_emb[7000:7055, 1], s=100, alpha=1, c=colors[10], edgecolors='k')
    # plt.scatter(train_proj_emb[7055:7151,0], train_proj_emb[7055:7151, 1], s=100, alpha=1, c=colors[9], edgecolors='k')
    # plt.scatter(train_proj_emb[7039:7040,0], train_proj_emb[7039:7040, 1], s=500, alpha=1, c=colors[2], edgecolors='k')

    # na12
    # plt.scatter(train_proj_emb[7000:7012,0], train_proj_emb[7000:7012, 1], s=100, alpha=1, c=colors[10], edgecolors='k')
    # plt.scatter(train_proj_emb[7012:7019,0], train_proj_emb[7012:7019, 1], s=100, alpha=1, c=colors[9], edgecolors='k')
    # plt.scatter(train_proj_emb[7005:7006,0], train_proj_emb[7005:7006, 1], s=500, alpha=1, c=colors[3], edgecolors='k')

    # a3b2
    # plt.scatter(train_proj_emb[7000:7146,0], train_proj_emb[7000:7146, 1], s=100, alpha=1, c=colors[10], edgecolors='k')
    # plt.scatter(train_proj_emb[7146:7245,0], train_proj_emb[7146:7245, 1], s=100, alpha=1, c=colors[9], edgecolors='k')
    # plt.scatter(train_proj_emb[7180:7181,0], train_proj_emb[7180:7181, 1], s=500, alpha=1, c=colors[4], edgecolors='k')

    # a7
    plt.scatter(train_proj_emb[7000:7067,0], train_proj_emb[7000:7067, 1], s=100, alpha=1, c=colors[10], edgecolors='k')
    plt.scatter(train_proj_emb[7067:7158,0], train_proj_emb[7067:7158, 1], s=100, alpha=1, c=colors[9], edgecolors='k')
    plt.scatter(train_proj_emb[7017:7018,0], train_proj_emb[7017:7018, 1], s=500, alpha=1, c=colors[5], edgecolors='k')

    # plt.scatter(train_proj_emb[6559:6560,0], train_proj_emb[6559:6560, 1], s=500, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6106:6107,0], train_proj_emb[6106:6107, 1], s=100, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6041:6042,0], train_proj_emb[6041:6042, 1], s=500, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6883:6884,0], train_proj_emb[6883:6884, 1], s=500, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6238:6239,0], train_proj_emb[6238:6239, 1], s=500, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6657:6658,0], train_proj_emb[6657:6658, 1], s=500, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6544:6545,0], train_proj_emb[6544:6545, 1], s=500, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6751:6752,0], train_proj_emb[6751:6752, 1], s=500, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6580:6581,0], train_proj_emb[6580:6581, 1], s=100, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6461:6462,0], train_proj_emb[6461:6462, 1], s=100, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6644:6645,0], train_proj_emb[6644:6645, 1], s=100, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6013:6014,0], train_proj_emb[6013:6014, 1], s=100, alpha=1, c="yellow", edgecolors='k')
    # plt.scatter(train_proj_emb[6718:6719,0], train_proj_emb[6718:6719, 1], s=100, alpha=1, c="yellow", edgecolors='k')
    
    # a9a10
    # plt.scatter(train_proj_emb[7000:7044,0], train_proj_emb[7000:7044, 1], s=100, alpha=1, c=colors[10], edgecolors='k')
    # plt.scatter(train_proj_emb[7044:7092,0], train_proj_emb[7044:7092, 1], s=100, alpha=1, c=colors[9], edgecolors='k')
    # plt.scatter(train_proj_emb[7043:7044,0], train_proj_emb[7043:7044, 1], s=500, alpha=1, c=colors[6], edgecolors='k')

    # a4b2
    # plt.scatter(train_proj_emb[7000:7010,0], train_proj_emb[7000:7010, 1], s=100, alpha=1, c=colors[10], edgecolors='k')
    # plt.scatter(train_proj_emb[7010:7088,0], train_proj_emb[7010:7088, 1], s=100, alpha=1, c=colors[9], edgecolors='k')
    # plt.scatter(train_proj_emb[7064:7065,0], train_proj_emb[7064:7065, 1], s=500, alpha=1, c=colors[7], edgecolors='k')

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.show()
    plt.savefig('a7_final.png', bbox_inches='tight', pad_inches=0)

plot_embedding()
