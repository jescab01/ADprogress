
import os

import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
simulations_tag = "PSEmpi_FreqCharts3.0-m02d01y2023-t12h.31m.46s"  # Tag cluster job

main_folder = 'E:\LCCN_Local\PycharmProjects\\brainModels\FrequencyChart\data\\'
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")

df.He = [float(row["He"]) if "classical" in row["mode"] else float(row["He"][1:-1]) for i, row in df.iterrows()]
df.Hi = [float(row["Hi"]) if "classical" in row["mode"] else float(row["Hi"][1:-1]) for i, row in df.iterrows()]
df.taue = [float(row["taue"]) if "classical" in row["mode"] else float(row["taue"][1:-1]) for i, row in df.iterrows()]
df.taui = [float(row["taui"]) if "classical" in row["mode"] else float(row["taui"][1:-1]) for i, row in df.iterrows()]


df["roi1_Hz"].loc[df["roi1_auc"] < 1e-6] = 0

# Average out repetitions
df_avg_ = df.groupby(['mode', "p", 'He', 'Hi', 'taue', 'taui', 'Cee', 'Cie', 'exp']).mean().reset_index()


# Plot results
for mode1 in ["classical"]:
    for mode2 in ["fixed"]:

        df_avg = df_avg_.loc[df_avg_["mode"].str.contains(mode2)]

        cmax_freq, cmin_freq = max(df_avg["roi1_Hz"].values), min(df_avg["roi1_Hz"].values)
        cmax_pow, cmin_pow = max(df_avg["roi1_auc"].values), min(df_avg["roi1_auc"].values)
        cmax_fr, cmin_fr = max(df_avg["roi1_meanFR"].values), min(df_avg["roi1_meanFR"].values)

        mode = mode1 + "&" + mode2

        cols = 3
        fig = make_subplots(rows=4, cols=cols, vertical_spacing=0.1,
                            column_titles=["Frequency", "Power", "meanFiringRate"], horizontal_spacing=0.12)

        # Plot He-Hi
        sub_df = df.loc[(df["mode"]==mode) & (df["exp"]=="exp_H")]
        fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.He, y=sub_df.Hi, colorbar_x=0.26,
                                 colorbar=dict(thickness=10, title="Hz"), zmax=cmax_freq, zmin=cmin_freq), row=1, col=1)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.He, y=sub_df.Hi, colorbar_x=0.64,
                                 colorbar=dict(thickness=10, title="dB"), zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.He, y=sub_df.Hi, colorbar=dict(thickness=10, title="Hz"),
                                 zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=1, col=3)
        for i in range(cols):
            fig.add_trace(go.Scatter(x=[3.25], y=[22], mode="markers", marker=dict(color="red"), showlegend=False), row=1, col=i+1)

        # Plot taue-taui
        sub_df = df.loc[(df["mode"] == mode) & (df["exp"] == "exp_tau")]
        fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.taue, y=sub_df.taui,
                                 showscale=False, zmax=cmax_freq, zmin=cmin_freq), row=2, col=1)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.taue, y=sub_df.taui,
                                 showscale=False, zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=2, col=2)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.taue, y=sub_df.taui, showscale=False,
                                 zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=2, col=3)
        for i in range(cols):
            fig.add_trace(go.Scatter(x=[10], y=[20], mode="markers", marker=dict(color="red"), showlegend=False), row=2, col=i+1)

        # Plot He-Cie
        sub_df = df.loc[(df["mode"]==mode) & (df["exp"]=="exp_1")]
        fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.He, y=sub_df.Cie, showscale=False,
                                zmax=cmax_freq, zmin=cmin_freq), row=3, col=1)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.He, y=sub_df.Cie, showscale=False,
                                 zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=3, col=2)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.He, y=sub_df.Cie, showscale=False, colorbar=dict(thickness=10, title="mV"),
                                 zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=3, col=3)
        for i in range(cols):
            fig.add_trace(go.Scatter(x=[3.25], y=[33.75], mode="markers", marker=dict(color="red"), showlegend=False), row=3, col=i+1)


        # Plot He-Cie
        sub_df = df.loc[(df["mode"] == mode) & (df["exp"] == "exp_2")]
        fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.p, y=sub_df.Cee,
                                 showscale=False, zmax=cmax_freq, zmin=cmin_freq), row=4, col=1)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.p, y=sub_df.Cee,
                                 showscale=False, zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=4, col=2)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.p, y=sub_df.Cee, showscale=False,
                                 zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=4, col=3)
        for i in range(cols):
            fig.add_trace(go.Scatter(x=[0.22], y=[108], mode="markers", marker=dict(color="red"), showlegend=False), row=4, col=i+1)


        fig.update_layout(xaxis1=dict(title="He (mV)"), yaxis1=dict(title="Hi (mV)"),
                          xaxis2=dict(title="He (mV)"), yaxis2=dict(title="Hi (mV)"),
                          xaxis3=dict(title="He (mV)"), yaxis3=dict(title="Hi (mV)"),

                          xaxis4=dict(title="taue (ms)"), yaxis4=dict(title="taui (ms)"),
                          xaxis5=dict(title="taue (ms)"), yaxis5=dict(title="taui (ms)"),
                          xaxis6=dict(title="taue (ms)"), yaxis6=dict(title="taui (ms)"),

                          xaxis7=dict(title="He (mV)"), yaxis7=dict(title="Cie"),
                          xaxis8=dict(title="He (mV)"), yaxis8=dict(title="Cie"),
                          xaxis9=dict(title="He (mV)"), yaxis9=dict(title="Cie"),

                          xaxis10=dict(title="p (mV)"), yaxis10=dict(title="Cee|Cie"),
                          xaxis11=dict(title="p (mV)"), yaxis11=dict(title="Cee|Cie"),
                          xaxis12=dict(title="p (mV)"), yaxis12=dict(title="Cee|Cie"),
                          title="Frequency charts   _" + mode)

        pio.write_html(fig, file=main_folder + simulations_tag + "/FreqCharts_" + mode + ".html", auto_open=True)

        # Save dataframe per cond
        # df.loc[df["mode"]==mode].to_csv(main_folder + simulations_tag + "/FreqCharts_" + mode + ".csv")

