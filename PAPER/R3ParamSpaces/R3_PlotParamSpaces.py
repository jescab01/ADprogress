
import pandas as pd
import numpy as np
from collections import Counter

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
simulations_tag = "PSEmpi_ADpgCirc_vCC-m03d29y2023-t16h.32m.36s"  # Tag cluster job
main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\PAPER\\R3ParamSpaces\\'

results = pd.read_csv(main_folder + simulations_tag + "/results.csv")

title = "indepParams_vCC"

# Init param values

# Columns for measures with colorscales
measures = ['fpeak', 'relpow_alpha', 'avgrate_pos', 'avgfc_pos']
tbar = ["Hz", "dB", "kHz", "PLV"]
cmaps = ["Plasma", "Viridis", "Cividis",  "Turbo"]

# Rows for explored parameters
params = ["maxHe", "minCie", "minCee", "maxTAU2SC"]
init_paramvalues = [3.25, 33.75, 108, "scaledWeights"]

results["maxHe_p"] = results.maxHe.values / init_paramvalues[0] * 100
results["minCie_p"] = - results.minCie.values / init_paramvalues[1] * 100
results["minCee_p"] = - results.minCee.values / init_paramvalues[2] * 100
results["maxTAU2SC_p"] = results.maxTAU2SC.values * 100

params = ["maxHe_p", "minCie_p", "minCee_p", "maxTAU2SC"]


# Identify fixed parameters and save in readme.txt
fixed_params = [Counter(results[param].values).most_common(1)[0][0] for param in params]
with open(main_folder + simulations_tag + "\\fixed_params.txt", "w") as f:
    f.write("Init parameters:  He" + str(init_paramvalues[0]) + "; Cie" + str(init_paramvalues[1]) + "; minCee" + str(init_paramvalues[2]) + "; maxTAU2SC" + str(init_paramvalues[3]))
    f.write("Fixed change limits:  maxHe" + str(fixed_params[0]) + "; minCie" + str(fixed_params[1]) + "; minCee" + str(fixed_params[2]) + "; maxTAU2SC" + str(fixed_params[3]))
    f.close()

x_bar = [-0.045+(j*1.045/len(measures)+1/len(measures)) for j, measure in enumerate(measures)]
x_bar = [0.195, 0.466, 0.735, 1.005]
fig = make_subplots(rows=len(params), cols=len(measures), x_title="Time (years)",
                    shared_xaxes=True, horizontal_spacing=0.075, shared_yaxes=True,
                    column_titles=["Frequency peak", "Relative alpha power", "Firing rate", "Functional connectivity"])

for i, param in enumerate(params):

    subset = results.groupby([param, "time"]).mean().reset_index()
    # subset = results.loc[results["minHi"] == minHi]
    sl = True if i == 0 else False
    for j, measure in enumerate(measures):
        fig.add_trace(
            go.Heatmap(z=subset[measure].values, x=subset.time, y=subset[param].values,
                       zmin=results[measure].min(), zmax=results[measure].max(), showscale=sl, colorscale=cmaps[j],
                       colorbar=dict(title=tbar[j], thickness=8, x=x_bar[j])),
            row=i+1, col=j+1)

        # Add line for no change
        fig.add_hline(y=fixed_params[i], line=dict(width=6, color="red"), opacity=0.3, row=i+1, col=j+1)


tickvals = [[0, 20, 40, 60], [0, -20, -40, -60, -80], [0, -20, -40, -60, -80]]
ticktext = [[str(val) + "%  " + str(round(init_paramvalues[i] + init_paramvalues[i] * val/100, 2)) for val in set_tickvals] for i, set_tickvals in enumerate(tickvals)]

fig.update_layout(yaxis1=dict(title="He<br>upper limit", title_standoff=0, tickvals=tickvals[0], ticktext=ticktext[0]),
                  yaxis5=dict(title="Cie<br>lower limit", title_standoff=0, tickvals=tickvals[1], ticktext=ticktext[1]),
                  yaxis9=dict(title="Cee<br>lower limit", title_standoff=0, tickvals=tickvals[2], ticktext=ticktext[2]),
                  yaxis13=dict(title="SC<br>damage limit"), height=750, width=1200)

pio.write_html(fig, file=main_folder + simulations_tag + "/PSE_ADpgCirc" + title + ".html", auto_open=True)
pio.write_image(fig, file=main_folder + simulations_tag + "/PSE_ADpgCirc" + title + ".svg")

