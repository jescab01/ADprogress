
import pandas as pd
import numpy as np

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\mpi_adpgCirc\PSE\\'
simulations_tag = "PSEmpi_ADpgCirc_vCC-m03d29y2023-t11h.50m.47s"  # Tag cluster job

results = pd.read_csv(main_folder + simulations_tag + "/results.csv")

title = "indepParams_vCC"

# Work on braid surface
r_names = ["rI", "rII", "rIII", "rIV", "rV"]
                                    # (-)rxs_avg to sort in descending order
results["sortRx_TIME"] = [str(np.asarray(r_names)[np.argsort(-row)]) for i, row in results.iloc[:, -5:].iterrows()]

results["sortRx_INT"] = np.nan

corresp = []
for i, pattern in enumerate(sorted(set(results["sortRx_TIME"]))):

    results["sortRx_INT"].loc[results["sortRx_TIME"] == pattern] = i
    corresp.append([i, pattern])

# auxiliar: divide for seed
results_both = results.loc[results.seedAB == results.seedTAU]
results = results.loc[results.seedAB != results.seedTAU]


# minHi_vals = sorted(set(list(results.minHi)))
measures = ["fpeak", "relpow_alpha", "avgrate_pos", "avgfc_pos", "sortRx_INT"]  # results.columns[[8, 9, 10, 12, -1]]
tbar = ["Hz", "dB", "Hz", "PLV", "order"]
cmaps = ["Plasma", "Viridis", "Cividis",  "Turbo", "portland"]

# params = ["maxHe", "minCie", "minCee", "maxTAU2SC"]
# params = ["HAdamrate"]
params = ["seedAB", "seedTAU", "both"]

fig = make_subplots(rows=len(params), cols=len(measures), x_title="Time (years)",
                    shared_xaxes=True, horizontal_spacing=0.075, shared_yaxes=True,
                    column_titles=measures, row_titles=params)

for i, param in enumerate(params):

    if param == "both":
        param="seedAB"

    subset = results.groupby([param, "time"]).mean().reset_index()


    # subset = results.loc[results["minHi"] == minHi]
    sl = True if i == 0 else False

    for j, measure in enumerate(measures):

        fig.add_trace(
            go.Heatmap(z=subset[measure].values, x=subset.time, y=subset[param].values,
                       zmin=results[measure].min(), zmax=results[measure].max(), showscale=sl, colorscale=cmaps[j],
                       colorbar=dict(title=tbar[j], thickness=8, x=-0.045+(j*1.045/len(measures)+1/len(measures)))),
            row=i+1, col=j+1)

    # Add independently the figure for braid surface
fig.update_layout(yaxis1=dict(type="log"), yaxis2=dict(type="log"), yaxis3=dict(type="log"), yaxis4=dict(type="log"), yaxis5=dict(type="log"),
                  yaxis6=dict(type="log"), yaxis7=dict(type="log"), yaxis8=dict(type="log"), yaxis9=dict(type="log"), yaxis10=dict(type="log"),
                  yaxis11=dict(type="log"), yaxis12=dict(type="log"), yaxis13=dict(type="log"), yaxis14=dict(type="log"), yaxis15=dict(type="log"))

pio.write_html(fig, file=main_folder + simulations_tag + "/PSE_ADpgCirc" + title + ".html", auto_open=True)

