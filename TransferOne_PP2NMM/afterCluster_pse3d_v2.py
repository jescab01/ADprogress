
import os
import time
import pandas as pd
import numpy as np

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTpeaks

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"



simulations_tag = "PSEmpi_ADpg_PSE3d-m11d10y2022-t17h.09m.44s"  # Tag cluster job
main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\TransferOne_PP2NMM\PSE\\'
df = pd.read_pickle(main_folder + simulations_tag + "/results.pkl")

df = df.astype({"He": "float", "Hi": "float", "taue":"float", "taui":"float", "meanS":"float", "freq":"float", "pow":"float"})

df["logpow"] = [np.log(row["pow"]) if row["pow"] != 0 else 0 for i, row in df.iterrows()]
df["logpow"] = df["logpow"] - min(df["logpow"]) + 1

## Plot signals to check --

# for i, row in df.loc[-df["type"].str.contains("flat")].iterrows():
#     fig = go.Figure()
#     for ii in range(10):
#         fig.add_trace(go.Scatter(x=np.arange(4000), y=row["signals"][ii]))
#     fig.show("browser")
#
# sub = df.loc[df["taui"] == 20]
# fig = px.scatter_3d(sub, x="He", y="Hi", z="taue", symbol="type")
# fig.show("browser")


## MULTIPLE PLOTS APPROACH - if this is too heavy for plotting and slows down the interaction, go below.
fig = make_subplots(rows=1, cols=3, subplot_titles=["tau_i==%i" % taui for taui in sorted(set(df.taui))],
                    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]], horizontal_spacing=0.01)

for j, taui in enumerate(sorted(set(df.taui))):
    subset = df.loc[df["taui"] == taui]
    sl = True if j==0 else False
    # 1. Flat traces (small black diamonds, low opacity)
    dfsub = subset.loc[subset["type"]=="flat"]
    fig.add_trace(go.Scatter3d(x=dfsub.He, y=dfsub.Hi, z=dfsub.taue, mode="markers", name="flats", legendgroup="flats", showlegend=sl,
                               marker=dict(color="black", symbol="diamond", size=3),  opacity=0.1), row=1, col=1+j)

    # 2. endFlat traces (small black diamonds, low opacity)
    dfsub = subset.loc[subset["type"]=="endflat"]
    fig.add_trace(go.Scatter3d(x=dfsub.He, y=dfsub.Hi, z=dfsub.taue, mode="markers", name="endflats", legendgroup="endflats", showlegend=sl,
                               marker=dict(color="black", symbol="diamond", size=4),  opacity=0.2), row=1, col=1+j)

    # 3. Oscillations (scatter) [-8][8-12][12-]
    dfsub = subset.loc[(-subset["type"].str.contains("flat")) & (subset["freq"]<8)]
    hovertext = ["He %0.2f, Hi %0.2f<br> taue %0.2f, taui %0.2f<br><br>Frequency (Hz) - %0.3f<br>Power (dB) - %0.3f" %
                 (row.He, row.Hi, row.taue, row.taui, row["freq"], row["pow"]) for i, row in dfsub.iterrows()]
    fig.add_trace(go.Scatter3d(
        x=dfsub.He, y=dfsub.Hi, z=dfsub.taue, mode="markers", name="oscillating -8Hz", legendgroup="oscillating -8Hz", hovertext=hovertext, hoverinfo="text",
        showlegend=sl, marker=dict(color=dfsub.freq, size=dfsub.logpow, colorscale="Plasma", cmax=np.max(df.freq), reversescale=True,
                                   cmin=np.min(df.loc[df["freq"]!=0].freq), colorbar=dict(thickness=20, len=0.5, y=0.5)),
        opacity=1), row=1, col=1+j)

    dfsub = subset.loc[(-subset["type"].str.contains("flat")) & (subset["freq"]>8) & (subset["freq"]<12)]
    hovertext = ["He %0.2f, Hi %0.2f<br> taue %0.2f, taui %0.2f<br><br>Frequency (Hz) - %0.3f<br>Power (dB) - %0.3f" %
                 (row.He, row.Hi, row.taue, row.taui, row["freq"], row["pow"]) for i, row in dfsub.iterrows()]
    fig.add_trace(go.Scatter3d(
        x=dfsub.He, y=dfsub.Hi, z=dfsub.taue, mode="markers", name="oscillating 8-12Hz", legendgroup="oscillating 8-12Hz", hovertext=hovertext, hoverinfo="text",
        showlegend=sl, marker=dict(color=dfsub.freq, size=dfsub.logpow, colorscale="Plasma", cmax=np.max(df.freq), reversescale=True,
                                   cmin=np.min(df.loc[df["freq"]!=0].freq)), opacity=1), row=1, col=1+j)

    dfsub = subset.loc[(-subset["type"].str.contains("flat")) & (subset["freq"]>12)]
    hovertext = ["He %0.2f, Hi %0.2f<br> taue %0.2f, taui %0.2f<br><br>Frequency (Hz) - %0.3f<br>Power (dB) - %0.3f" %
                 (row.He, row.Hi, row.taue, row.taui, row["freq"], row["pow"]) for i, row in dfsub.iterrows()]
    fig.add_trace(go.Scatter3d(
        x=dfsub.He, y=dfsub.Hi, z=dfsub.taue, mode="markers", name="oscillating 12- Hz", legendgroup="oscillating 12- Hz", hovertext=hovertext, hoverinfo="text",
        showlegend=sl, marker=dict(color=dfsub.freq, size=dfsub.logpow, colorscale="Plasma", cmax=np.max(df.freq), reversescale=True,
                                   cmin=np.min(df.loc[df["freq"]!=0].freq)),
        opacity=1), row=1, col=1+j)

scene = dict(xaxis=dict(title="He", range=[np.min(df.He), np.max(df.He)], autorange=False),
             yaxis=dict(title="Hi", range=[np.min(df.Hi), np.max(df.Hi)], autorange=False),
             zaxis=dict(title="tau_e", range=[np.min(df.taue), np.max(df.taue)], autorange=False),
             camera=dict(eye=dict(x=1.5, y=2.5, z=2.5)))

fig.update_layout(template="plotly_white",
                  scene1=scene, scene2=scene, scene3=scene)

pio.write_html(fig, file=main_folder + simulations_tag + "/Cube4D_paramspace.html", auto_open=True)




## SECOND APPROACH: animation over tau_e

df_ani = df.copy()
df_ani["freq"].loc[df_ani["freq"] == 0] = None


fig = make_subplots(rows=3, cols=3, subplot_titles=["tau_i==%i" % taui for taui in sorted(set(df.taui))],
                    specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]], shared_yaxes=True, shared_xaxes=True)

for j, taui in enumerate(sorted(set(df_ani.taui))):
    subset = df_ani.loc[df["taui"] == taui]
    sl = True if j == 0 else False

    # 1. freq
    dfsub = subset.loc[subset["taue"] == 6].dropna()
    fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub.freq, zmin=min(df.freq), zmax=max(df.freq), colorbar=dict(len=0.3, y=0.9, thickness=15)), row=1, col=1+j)

    # 2. pow
    fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub["pow"], colorscale="Viridis", zmin=min(df["pow"]), zmax=max(df["pow"]), colorbar=dict(len=0.3, y=0.5, thickness=15)), row=2, col=1+j)

    # 3. mean signal
    fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub.meanS, colorscale="Cividis", zmin=min(df.meanS), zmax=max(df.meanS), colorbar=dict(len=0.3, y=0.1, thickness=15)), row=3, col=1+j)


frames = []
for i, taue in enumerate(sorted(set(df.taue))):

    sub = df_ani.loc[df_ani["taue"] == taue].dropna()

    sub_1 = sub.loc[sub["taui"]==16]
    sub_2 = sub.loc[sub["taui"]==20]
    sub_3 = sub.loc[sub["taui"]==24]

    frames.append(go.Frame(data=[
        go.Heatmap(x=sub_1.He, y=sub_1.Hi, z=sub_1.freq),
        go.Heatmap(x=sub_1.He, y=sub_1.Hi, z=sub_1["pow"]),
        go.Heatmap(x=sub_1.He, y=sub_1.Hi, z=sub_1.meanS),

        go.Heatmap(x=sub_2.He, y=sub_2.Hi, z=sub_2.freq),
        go.Heatmap(x=sub_2.He, y=sub_2.Hi, z=sub_2["pow"]),
        go.Heatmap(x=sub_2.He, y=sub_2.Hi, z=sub_2.meanS),

        go.Heatmap(x=sub_3.He, y=sub_3.Hi, z=sub_3.freq),
        go.Heatmap(x=sub_3.He, y=sub_3.Hi, z=sub_3["pow"]),
        go.Heatmap(x=sub_3.He, y=sub_3.Hi, z=sub_3.meanS)],

        traces=[0, 1, 2, 3, 4, 5, 6, 7, 8], name=str(round(taue, 2))))


fig.update(frames=frames)

xaxis = dict(title="He", range=[min(df.He), max(df.He)], autorange=False)
yaxis = dict(title="Hi", range=[min(df.Hi), max(df.Hi)], autorange=False)

# CONTROLS : Add sliders and buttons
fig.update_layout(title="4D parameter space - BNM simulations <br> init. conditions reference [He3.25, Hi22, taue=10, taui=20]",
    template="plotly_white", xaxis1=xaxis, yaxis1=yaxis,
    xaxis2=xaxis, yaxis2=yaxis, xaxis3=xaxis, yaxis3=yaxis, xaxis4=xaxis, yaxis4=yaxis, xaxis5=xaxis, yaxis5=yaxis,
    xaxis6=xaxis, yaxis6=yaxis, xaxis7=xaxis, yaxis7=yaxis, xaxis8=xaxis, yaxis8=yaxis, xaxis9=xaxis, yaxis9=yaxis,

    updatemenus=[dict(type="buttons", showactive=True, y=1.30, x=1.05, xanchor="right",
                      buttons=[
                          dict(label="Play", method="animate",
                               args=[None,
                                     dict(frame=dict(duration=500, redraw=True, easing="cubic-in-out"),
                                          transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                          dict(label="Pause", method="animate",
                               args=[[None],
                                     dict(frame=dict(duration=0, redraw=False, easing="cubic-in-out"),
                                          transition=dict(duration=0), mode="immediate")])])],
    sliders=[dict(
        steps=[dict(args=[[f.name],
                          dict(mode="immediate", frame=dict(duration=0, redraw=True),
                               transition=dict(duration=0))], label=f.name, method='animate',)
               for f in frames],
        x=0.97, xanchor="right", y=1.35, len=0.5,
        currentvalue=dict(font=dict(size=15), prefix="taue - ", visible=True, xanchor="left"),
        tickcolor="white")],
)

pio.write_html(fig, file=main_folder + simulations_tag + "/Animated4D_paramspace.html", auto_open=True)

