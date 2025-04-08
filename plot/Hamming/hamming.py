import os
import plotly
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
join = os.path.join
cycles = ["C1", "C2", "C3", "C4", "C5", "final", "rand"]

for cycle in cycles:
    print(cycle)
    all_data_excel = f"./CreoPep/plot/Hamming/hamming_{cycle}.csv"

    team_sheet = pd.read_csv(all_data_excel)
    colors = {
        'a3b2': '#845ec2',  # Tomato
        'a3b4': '#2c73d2',  # LightGreen
        'a7': '#b0a8b9',    # LightBlue
        'a9a10': '#ffc75f', # LightYellow
        'AChBP': '#ff8066', # LightPink
        'Ca22': '#00c9a7', # DarkSeaGreen
        'Na12': '#926c00', # LightSkyBlue
        'a4b2': 'pink',  # Tomato
    }
    tasks = ["AChBP","Ca22","a3b4","Na12","a3b2","a7","a9a10","a4b2"]
    tasks_name = ["AChBP","Ca22","α3β4","Na12","α3β2","α7","α9α10","α4β2"]

    fig = go.Figure()
    fig.add_trace(go.Box(y=team_sheet[tasks_name[0]], name=tasks_name[0],marker_size=10,boxpoints='outliers',pointpos=0,jitter=0.05,
    line_color='rgba(0,0,0,1)',line_width=1,fillcolor=colors[tasks[0]],marker_color='rgba(128,128,128,0.6)',boxmean=True))
    fig.add_trace(go.Box(y=team_sheet[tasks_name[1]], name=tasks_name[1],marker_size=10,boxpoints='outliers',pointpos=0,jitter=0.05,
    line_color='rgba(0,0,0,1)',line_width=1,fillcolor=colors[tasks[1]],marker_color='rgba(128,128,128,0.6)',boxmean=True))
    fig.add_trace(go.Box(y=team_sheet[tasks_name[2]], name=tasks_name[2],marker_size=10,boxpoints='outliers',pointpos=0,jitter=0.05,
    line_color='rgba(0,0,0,1)',line_width=1,fillcolor=colors[tasks[2]],marker_color='rgba(128,128,128,0.6)',boxmean=True))
    fig.add_trace(go.Box(y=team_sheet[tasks_name[3]], name=tasks_name[3],marker_size=10,boxpoints='outliers',pointpos=0,jitter=0.05,
    line_color='rgba(0,0,0,1)',line_width=1,fillcolor=colors[tasks[3]],marker_color='rgba(128,128,128,0.6)',boxmean=True))
    fig.add_trace(go.Box(y=team_sheet[tasks_name[4]], name=tasks_name[4],marker_size=10,boxpoints='outliers',pointpos=0,jitter=0.05,
    line_color='rgba(0,0,0,1)',line_width=1,fillcolor=colors[tasks[4]],marker_color='rgba(128,128,128,0.6)',boxmean=True))
    fig.add_trace(go.Box(y=team_sheet[tasks_name[5]], name=tasks_name[5],marker_size=10,boxpoints='outliers',pointpos=0,jitter=0.05,
    line_color='rgba(0,0,0,1)',line_width=1,fillcolor=colors[tasks[5]],marker_color='rgba(128,128,128,0.6)',boxmean=True))
    fig.add_trace(go.Box(y=team_sheet[tasks_name[6]], name=tasks_name[6],marker_size=10,boxpoints='outliers',pointpos=0,jitter=0.05,
    line_color='rgba(0,0,0,1)',line_width=1,fillcolor=colors[tasks[6]],marker_color='rgba(128,128,128,0.6)',boxmean=True))
    fig.add_trace(go.Box(y=team_sheet[tasks_name[7]], name=tasks_name[7],marker_size=10,boxpoints='outliers',pointpos=0,jitter=0.05,
    line_color='rgba(0,0,0,1)',line_width=1,fillcolor=colors[tasks[7]],marker_color='rgba(128,128,128,0.6)',boxmean=True))
    fig.update_layout(
        font=dict(
            family="Times New Roman",
            size=18,
            color="#000000"
        ),
        plot_bgcolor='white',
        # paper_bgcolor='white'
    )


    fig.update_yaxes(showticklabels=True, gridcolor='rgba(128,128,128,0.5)', gridwidth=0.3, range=[0, 1])
    fig.update_xaxes(showticklabels=False)

    fig.update_layout(
        autosize=False,
        width=400,
        height=300

    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    # fig.show()
    fig.write_image(f"./CreoPep/plot/Hamming/hamming_{cycle}.png",scale=4)
