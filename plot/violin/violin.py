import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv("foldx1.csv")
# ['AchBP',"Cav22","α4β2","a9a10","a7","α3β4","Nav12","α3β2"]
# ['AchBP',"Cav2.2","α4β2","α9α10","α7","α3β4","Nav1.2","α3β2"]
tasks = ["AchBP","Cav2.2","α4β2","α9α10","α7","α3β4","Nav1.2","α3β2"]

for task in tasks:
    fig = go.Figure()
    colors = {
        'α3β2': '#845ec2',
        'α3β4': '#2c73d2',
        'α7': '#b0a8b9',
        'α9α10': '#ffc75f',
        'AChBP': '#ff8066',
        'Cav2.2': '#00c9a7',
        'Nav1.2': '#926c00',
        'α4β2': 'pink',
    }

    # random
    fig.add_trace(go.Violin(
        y=df[f'{task}_random'],
        box_visible=True,
        line=dict(color='black', width=1),
        marker=dict(size=3, color='black'),
        box=dict(line=dict(color='black', width=1)),
        line_color='black',
        meanline_visible=True,
        fillcolor=colors[task],
        opacity=1,
        name=f'{task}_random'
    ))

    # C1
    fig.add_trace(go.Violin(
        y=df[f'{task}_C1'],
        box_visible=True,
        line=dict(color='black', width=1),
        marker=dict(size=3, color='black'),
        box=dict(line=dict(color='black', width=1)),
        line_color='black',
        meanline_visible=True,
        fillcolor=colors[task],
        opacity=1,
        name=f'{task}_C1'
    ))

    # C2
    fig.add_trace(go.Violin(
        y=df[f'{task}_C2'],
        box_visible=True,
        line=dict(color='black', width=1),
        marker=dict(size=3, color='black'),
        box=dict(line=dict(color='black', width=1)),
        line_color='black',
        meanline_visible=True,
        fillcolor=colors[task],
        opacity=1,
        name=f'{task}_C2'
    ))

    # C3
    fig.add_trace(go.Violin(
        y=df[f'{task}_C3'],
        box_visible=True,
        line=dict(color='black', width=1),
        marker=dict(size=3, color='black'),
        box=dict(line=dict(color='black', width=1)),
        line_color='black',
        meanline_visible=True,
        fillcolor=colors[task],
        opacity=1,
        name=f'{task}_C3'
    ))

    # C4
    fig.add_trace(go.Violin(
        y=df[f'{task}_C4'],
        box_visible=True,
        line=dict(color='black', width=1),
        marker=dict(size=3, color='black'),
        box=dict(line=dict(color='black', width=1)),
        line_color='black',
        meanline_visible=True,
        fillcolor=colors[task],
        opacity=1,
        name=f'{task}_C4'
    ))

    # C5
    fig.add_trace(go.Violin(
        y=df[f'{task}_C5'],
        box_visible=True,
        line=dict(color='black', width=1),
        marker=dict(size=3, color='black'),
        box=dict(line=dict(color='black', width=1)),
        line_color='black',
        meanline_visible=True,
        fillcolor=colors[task],
        opacity=1,
        name=f'{task}_C5'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        autosize=True,
        width=1000,
        height=200,
        xaxis=dict(showticklabels=False), 
        font=dict(
            family="Times New Roman",
            size=17,
            color="#000000"
        )
    )
    fig.show()
    fig.write_image(f"violin_{task}.png",scale=4)
