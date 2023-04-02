
import os
import time
import pandas as pd

from dash import Dash, html, dcc, Input, Output

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

app = Dash(__name__)


# Define PSE folder
main_folder = 'E:\LCCN_Local\PycharmProjects\\brainModels\FrequencyChart\data\\'
simulations_tag = "PSEmpi_FreqCharts2.0-m11d07y2022-t17h.02m.56s"  # Tag cluster job
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")

df.He = [float(row["He"]) if "classical" in row["mode"] else float(row["He"][1:-1]) for i, row in df.iterrows()]
df.Hi = [float(row["Hi"]) if "classical" in row["mode"] else float(row["Hi"][1:-1]) for i, row in df.iterrows()]
df.taue = [float(row["taue"]) if "classical" in row["mode"] else float(row["taue"][1:-1]) for i, row in df.iterrows()]
df.taui = [float(row["taui"]) if "classical" in row["mode"] else float(row["taui"][1:-1]) for i, row in df.iterrows()]

df["roi1_Hz"].loc[df["roi1_auc"] < 1e-6] = 0

# Average out repetitions
df_avg_ = df.groupby(['mode', 'He', 'Hi', 'taue', 'taui', 'exp']).mean().reset_index()


# Plot results

# for mode1 in ["classical", "prebif"]:
#     for mode2 in ["fixed", "balanced"]:
mode1 = "classical"
mode2 = "fixed"

df_avg = df_avg_.loc[df_avg_["mode"].str.contains(mode2)]

cmax_freq, cmin_freq = max(df_avg["roi1_Hz"].values), min(df_avg["roi1_Hz"].values)
cmax_pow, cmin_pow = max(df_avg["roi1_auc"].values), min(df_avg["roi1_auc"].values)
cmax_ms, cmin_ms = max(df_avg["roi1_meanS"].values), min(df_avg["roi1_meanS"].values)

mode = mode1 + "&" + mode2

fig = make_subplots(rows=2, cols=3, vertical_spacing=0.1, shared_yaxes=True,
                    subplot_titles=["Frequency", "Power", "meanSignal"], horizontal_spacing=0.12)

# Plot He-Hi
sub_df = df.loc[(df["mode"]==mode) & (df["exp"]=="exp_H")]
fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.He, y=sub_df.Hi, colorbar_x=0.26,
                         colorbar=dict(thickness=10, title="Hz"), zmax=cmax_freq, zmin=cmin_freq), row=1, col=1)

fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.He, y=sub_df.Hi, colorbar_x=0.64,
                         colorbar=dict(thickness=10, title="dB"), zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=1, col=2)

fig.add_trace(go.Heatmap(z=sub_df.roi1_meanS, x=sub_df.He, y=sub_df.Hi, colorbar=dict(thickness=10, title="mV"),
                         zmax=cmax_ms, zmin=cmin_ms, colorscale="Cividis"), row=1, col=3)

# Plot taue-taui
sub_df = df.loc[(df["mode"] == mode) & (df["exp"] == "exp_tau")]
fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.taue, y=sub_df.taui, colorbar_x=0.26,
                         showscale=False, zmax=cmax_freq, zmin=cmin_freq), row=2, col=1)
fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.taue, y=sub_df.taui, colorbar_x=0.64,
                         showscale=False, zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=2, col=2)
fig.add_trace(go.Heatmap(z=sub_df.roi1_meanS, x=sub_df.taue, y=sub_df.taui, showscale=False,
                         zmax=cmax_ms, zmin=cmin_ms, colorscale="Cividis"), row=2, col=3)

fig.update_layout(xaxis1=dict(title="He (mV)"), yaxis1=dict(title="Hi (mV)"),
                  xaxis2=dict(title="He (mV)"), yaxis2=dict(title="Hi (mV)"),
                  xaxis3=dict(title="He (mV)"), yaxis3=dict(title="Hi (mV)"),
                  xaxis4=dict(title="tau_e (mV)"), yaxis4=dict(title="tau_i (mV)"),
                  xaxis5=dict(title="tau_e (mV)"), yaxis5=dict(title="tau_i (mV)"),
                  xaxis6=dict(title="tau_e (mV)"), yaxis6=dict(title="tau_i (mV)"),
                  title="Frequency charts   _" + mode, height=800)

# pio.write_html(fig, file=main_folder + simulations_tag + "/FreqCharts_" + mode + ".html", auto_open=True)


## Once figure done, configure the app
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    html.Div(children='''Dash: A web application framework for your data.'''),
    dcc.Graph(id='indicator-graphic', figure=fig),
    dcc.RangeSlider(id='slider', min=cmin_pow, max=1000, tooltip={"placement": "bottom"}),

])

@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('slider', 'value'),
)
def update_graph(value):

    fig.data[1].zmin = value[0]
    fig.data[1].zmax = value[1]
    fig.data[4].zmin = value[0]
    fig.data[4].zmax = value[1]

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

