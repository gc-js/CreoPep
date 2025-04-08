import os
import plotly.graph_objects as go
import pandas as pd
cycles = ["C1", "C2", "C3", "C4", "C5", "final", "rand"]
lihuas = ["GRAVY","Charge","Isoelectric_point"]

for lihua in lihuas:
    for cycle in cycles:
        all_data_excel = f"/CreoPep/plot/physicochemical_properties/{cycle}/{lihua}.csv"
        all_data_train = r"/CreoPep/plot/physicochemical_properties/output_train_high.csv"
        team_sheet = pd.read_csv(all_data_excel)
        train_data = pd.read_csv(all_data_train, index_col="target")

        tasks = ["AChBP", "Ca22", "a3b4", "Na12", "a3b2", "a7", "a9a10", "a4b2"]
        tasks_name = ["AChBP", "Ca22", "α3β4", "Na12", "α3β2", "α7", "α9α10", "α4β2"]
        seqs = ['GCCSHPACNVDHPEIC','CKGKGAKCSRLMYDCCTGSCRSGKC','GCCSYPPCFATNPDC','CCNCSSKWCRDHSRCC',
                'GCCSHPACSVNHPELC','GCCSDPRCAWRC','GCCSDPRCNYDHPEIC','IRDECCSNPACRVNNPHVC']
        colors = {
            'a3b2': '#845ec2',  # Tomato
            'a3b4': '#2c73d2',  # LightGreen
            'a7': '#b0a8b9',    # LightBlue
            'a9a10': '#ffc75f', # LightYellow
            'AChBP': '#ff8066', # LightPink
            'Ca22': '#00c9a7',  # DarkSeaGreen
            'Na12': '#926c00',  # LightSkyBlue
            'a4b2': 'pink',     # Tomato
        }
        conotoxins = []
        for task, seq in zip(tasks, seqs):
            filtered_data = train_data[(train_data.index == task) & (train_data['seq'] == seq)]
            if not filtered_data.empty:
                conotoxins.append(filtered_data[lihua].values[0])

        fig = go.Figure()

        for i, task in enumerate(tasks):
            fig.add_trace(
                go.Violin(
                    x=[i] * len(train_data.loc[task, lihua]),
                    y=train_data.loc[task, lihua].tolist(), 
                    line_color='rgba(0,0,0,1)',
                    line_width=1,
                    fillcolor=colors[task],
                    box_visible=False,
                    meanline_visible=False,
                    points=False,
                    side='negative',
                    width=0.8
                )
            )

            fig.add_trace(
                go.Violin(
                    x=[i] * len(team_sheet[tasks_name[i]]),
                    y=team_sheet[tasks_name[i]], 
                    line_color='rgba(0,0,0,1)',
                    line_width=1,
                    fillcolor='lightgrey',
                    box_visible=False,
                    meanline_visible=False,
                    points=False,
                    side='positive',
                    width=0.8
                )
            )
            
            max_value = conotoxins[i]
            fig.add_trace(
                go.Scatter(
                    x=[i] * len(train_data.loc[task, lihua]),
                    y=[max_value],
                    mode='markers',
                    marker=dict(
                        color=colors[task],
                        size=13,
                        symbol='circle',
                        line=dict(width=1.5,color='black'),
                        opacity=1
                    ),
                )
            )

        fig.update_layout(
            font=dict(
                family="Times New Roman",
                size=18,
                color="#000000"
            ),
            plot_bgcolor='white',
            # paper_bgcolor='white'
        )

        fig.update_yaxes(showticklabels=True, gridcolor='rgba(128,128,128,0.5)', gridwidth=0.3)
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
        # fig.update_layout(violingap=0, violingroupgap=0, violinmode='overlay')
        fig.show()
        fig.write_image(f"./CreoPep/plot/physicochemical_properties/{cycle}_{lihua}.png",scale=4)
