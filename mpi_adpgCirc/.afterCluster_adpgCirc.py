
import pandas as pd

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\mpi_adpgCirc\PSE\\'
simulations_tag = "PSEmpi_ADpgCirc_vCC-m03d09y2023-t09h.45m.04s"  # Tag cluster job

results = pd.read_csv(main_folder + simulations_tag + "/results.csv")

title = "indepParams_vCC"

minHi_vals = sorted(set(list(results.minHi)))
measures = results.columns[[3, 4, 5, 7]]
tbar = ["Hz", "dB", "Hz", "PLV"]
cmaps = ["Plasma", "Viridis", "Cividis",  "Turbo"]

fig = make_subplots(rows=4, cols=len(minHi_vals), x_title="Time (years)", y_title="maxTAU2SC",
                    shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.075,
                    row_titles=["min Hi == " + str(hi) for hi in minHi_vals],
                    column_titles=list(measures.values))

for i, minHi in enumerate(minHi_vals):

    subset = results.loc[results["minHi"] == minHi]
    sl = True if i == 0 else False

    for j, measure in enumerate(measures):

        fig.add_trace(
            go.Heatmap(z=subset[measure].values, x=subset.time, y=subset.maxTAU2SC,
                       zmin=results[measure].min(), zmax=results[measure].max(), showscale=sl, colorscale=cmaps[j],
                       colorbar=dict(title=tbar[j], thickness=8, x=-0.045+(j*1.045/len(measures)+1/len(measures)))),
            row=i+1, col=j+1)

pio.write_html(fig, file=main_folder + simulations_tag + "/PSE_ADpgCirc" + title + ".html", auto_open=True)

