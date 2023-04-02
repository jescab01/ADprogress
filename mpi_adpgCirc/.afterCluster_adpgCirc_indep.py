
import pandas as pd
import numpy as np

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\mpi_adpgCirc\PSE\\'
simulations_tag = "PSEmpi_ADpgCirc_vCC-m03d09y2023-t23h.51m.41s"  # Tag cluster job

results = pd.read_csv(main_folder + simulations_tag + "/results.csv")

title = "indepParams_vCC"


# minHi_vals = sorted(set(list(results.minHi)))
measures = results.columns[[5, 6, 7, 9]]
tbar = ["Hz", "dB", "Hz", "PLV"]
cmaps = ["Plasma", "Viridis", "Cividis",  "Turbo"]

params = ["maxHe", "minCie", "minCee", "maxTAU2SC"]
# params = ["HAdamrate"]

hlines = [(0.35, 0),(75, 0), (20.5, 0), (0.3, 0)]

fig = make_subplots(rows=len(params), cols=len(measures), x_title="Time (years)",
                    shared_xaxes=True, horizontal_spacing=0.075, shared_yaxes=True,
                    column_titles=list(measures.values), row_titles=params)

for i, param in enumerate(params):

    subset = results.groupby([param, "time"]).mean().reset_index()

    # subset = results.loc[results["minHi"] == minHi]
    sl = True if i == 0 else False

    for j, measure in enumerate(measures):

        fig.add_trace(
            go.Heatmap(z=subset[measure].values, x=subset.time, y=subset[param].values,
                       zmin=results[measure].min(), zmax=results[measure].max(), showscale=sl, colorscale=cmaps[j],
                       colorbar=dict(title=tbar[j], thickness=8, x=-0.045+(j*1.045/len(measures)+1/len(measures)))),
            row=i+1, col=j+1)

        # # Add line for no change
        # fig.add_hline(y=, color=)
        #
        # # Add line for current value
        # fig.add_hline(y=, color=)

    # Add independently the figure for braid surface
# fig.update_layout(yaxis1=dict(type="log"), yaxis2=dict(type="log"), yaxis3=dict(type="log"), yaxis4=dict(type="log"), yaxis5=dict(type="log"))

pio.write_html(fig, file=main_folder + simulations_tag + "/PSE_ADpgCirc" + title + ".html", auto_open=True)

