
import os

import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
simulations_tag = "PSEmpi_FreqCharts4.0-m03d08y2023-t08h.24m.10s"  # Tag cluster job

main_folder = 'E:\LCCN_Local\PycharmProjects\\ADprogress\FrequencyCharts\data\\'
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
        cmax_rel, cmin_rel = 0.75, 0

        mode = mode1 + "&" + mode2

        cols = 7
        fig = make_subplots(rows=3, cols=cols, vertical_spacing=0.1, horizontal_spacing=0.05,
                            column_titles=["Frequency", "meanFiringRate", "Peak Power", "relPowBeta", "relPowAlpha",
                                           "relPowTheta", "relPowDelta"])

        # Plot He-Hi
        sub_df = df.loc[(df["mode"]==mode) & (df["exp"]=="exp_HeHi")]
        fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.He, y=sub_df.Hi, colorbar_x=0.095,
                                 colorbar=dict(thickness=3, title="Hz"), zmax=cmax_freq, zmin=cmin_freq), row=1, col=1)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.He, y=sub_df.Hi, colorbar_x=0.245,
                                 colorbar=dict(thickness=3, title="Hz"),
                                 zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.He, y=sub_df.Hi, colorbar_x=0.4,
                                 colorbar=dict(thickness=3, title="dB"), zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"),
                      row=1, col=3)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.He, y=sub_df.Hi, colorbar_x=0.995,
                                 colorbar=dict(thickness=3),
                                 zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=1, col=4)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.He, y=sub_df.Hi, showscale=False,
                                zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=1, col=5)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.He, y=sub_df.Hi, showscale=False,
                                 zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=1, col=6)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucDelta, x=sub_df.He, y=sub_df.Hi, showscale=False,
                                 zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=1, col=7)

        for i in range(cols):
            sl = True if i==0 else False
            fig.add_trace(go.Scatter(x=[3.25], y=[22], mode="markers", marker=dict(color="red"),
                                     legendgroup="dot", name="initRef", showlegend=sl), row=1, col=i+1)

        # Plot He-p
        sub_df = df.loc[(df["mode"]==mode) & (df["exp"]=="exp_pHe")]
        fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.p, y=sub_df.He,
                                 showscale=False, zmax=cmax_freq, zmin=cmin_freq), row=2, col=1)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.p, y=sub_df.He,
                                 showscale=False,
                                 zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=2, col=2)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.p, y=sub_df.He,
                                 showscale=False, zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"),
                      row=2, col=3)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.p, y=sub_df.He,
                                 showscale=False,
                                 zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=2, col=4)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.p, y=sub_df.He, showscale=False,
                                zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=2, col=5)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.p, y=sub_df.He, showscale=False,
                                 zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=2, col=6)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucDelta, x=sub_df.p, y=sub_df.He, showscale=False,
                                 zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=2, col=7)

        for i in range(cols):
            sl = True if i==0 else False
            fig.add_trace(go.Scatter(x=[0.22], y=[3.25], mode="markers", marker=dict(color="red"),
                                     legendgroup="dot", name="initRef", showlegend=sl), row=2, col=i+1)

        # Plot He-p
        sub_df = df.loc[(df["mode"]==mode) & (df["exp"]=="exp_pHi")]
        fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.p, y=sub_df.Hi,
                                 showscale=False, zmax=cmax_freq, zmin=cmin_freq), row=3, col=1)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.p, y=sub_df.Hi,
                                 showscale=False,
                                 zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=3, col=2)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.p, y=sub_df.Hi,
                                 showscale=False, zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"),
                      row=3, col=3)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.p, y=sub_df.Hi,
                                 showscale=False,
                                 zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=3, col=4)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.p, y=sub_df.Hi, showscale=False,
                                zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=3, col=5)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.p, y=sub_df.Hi, showscale=False,
                                 zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=3, col=6)
        fig.add_trace(go.Heatmap(z=sub_df.roi1_aucDelta, x=sub_df.p, y=sub_df.Hi, showscale=False,
                                 zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=3, col=7)

        for i in range(cols):
            sl = True if i==0 else False
            fig.add_trace(go.Scatter(x=[0.22], y=[22], mode="markers", marker=dict(color="red"),
                                     legendgroup="dot", name="initRef", showlegend=sl), row=3, col=i+1)

        # # Plot taue-taui
        # sub_df = df.loc[(df["mode"] == mode) & (df["exp"] == "exp_tau")]
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.taue, y=sub_df.taui,
        #                          showscale=False, zmax=cmax_freq, zmin=cmin_freq), row=2, col=1)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.taue, y=sub_df.taui,
        #                          showscale=False, zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=2, col=2)
        #
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.taue, y=sub_df.taui,
        #                          showscale=False, zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=2, col=3)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.taue, y=sub_df.taui,
        #                          showscale=False, zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=2, col=4)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.taue, y=sub_df.taui,
        #                          showscale=False, zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=2, col=5)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucDelta, x=sub_df.taue, y=sub_df.taui,
        #                          showscale=False, zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=2, col=6)
        #
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.taue, y=sub_df.taui, showscale=False,
        #                          zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=2, col=7)
        # for i in range(cols):
        #     fig.add_trace(go.Scatter(x=[10], y=[20], mode="markers", marker=dict(color="red"), showlegend=False, legendgroup="dot"), row=2, col=i+1)

        # # Plot He - Cie
        # sub_df = df.loc[(df["mode"]==mode) & (df["exp"]=="exp_1")]
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.He, y=sub_df.Cie, showscale=False,
        #                         zmax=cmax_freq, zmin=cmin_freq), row=3, col=1)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.He, y=sub_df.Cie, showscale=False,
        #                          zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=3, col=2)
        #
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.He, y=sub_df.Cie, showscale=False,
        #                          zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=3, col=3)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.He, y=sub_df.Cie, showscale=False,
        #                          zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=3, col=4)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.He, y=sub_df.Cie, showscale=False,
        #                          zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=3, col=5)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucDelta, x=sub_df.He, y=sub_df.Cie, showscale=False,
        #                          zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=3, col=6)
        #
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.He, y=sub_df.Cie, showscale=False, colorbar=dict(thickness=10, title="mV"),
        #                          zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=3, col=7)

        # for i in range(cols):
        #     fig.add_trace(go.Scatter(x=[3.25], y=[33.75], mode="markers", marker=dict(color="red"), showlegend=False, legendgroup="dot"), row=3, col=i+1)
        #
        # # Plot Cee|Cie - p (input)
        # sub_df = df.loc[(df["mode"] == mode) & (df["exp"] == "exp_2")]
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.p, y=sub_df.Cee,
        #                          showscale=False, zmax=cmax_freq, zmin=cmin_freq), row=4, col=1)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.p, y=sub_df.Cee,
        #                          showscale=False, zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=4, col=2)
        #
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.p, y=sub_df.Cee,
        #                          showscale=False, zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=4, col=3)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.p, y=sub_df.Cee,
        #                          showscale=False, zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=4, col=4)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.p, y=sub_df.Cee,
        #                          showscale=False, zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=4, col=5)
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_aucDelta, x=sub_df.p, y=sub_df.Cee,
        #                          showscale=False, zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=4, col=6)
        #
        # fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.p, y=sub_df.Cee, showscale=False,
        #                          zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=4, col=7)
        # for i in range(cols):
        #     fig.add_trace(go.Scatter(x=[0.22], y=[108], mode="markers", marker=dict(color="red"), showlegend=False, legendgroup="dot"), row=4, col=i+1)

        fig.update_layout(xaxis1=dict(title="He (mV)"), yaxis1=dict(title="Hi (mV)"),
                          xaxis2=dict(title="He (mV)"),
                          xaxis3=dict(title="He (mV)"),
                          xaxis4=dict(title="He (mV)"),
                          xaxis5=dict(title="He (mV)"),
                          xaxis6=dict(title="He (mV)"),
                          xaxis7=dict(title="He (mV)"),

                          xaxis8=dict(title="p (mV)"), yaxis8=dict(title="He (mV)"),
                          xaxis9=dict(title="p (mV)"),
                          xaxis10=dict(title="p (mV)"),
                          xaxis11=dict(title="p (mV)"),
                          xaxis12=dict(title="p (mV)"),
                          xaxis13=dict(title="p (mV)"),
                          xaxis14=dict(title="p (mV)"),

                          xaxis15=dict(title="p (mV)"), yaxis15=dict(title="Hi (mV)"),
                          xaxis16=dict(title="p (mV)"),
                          xaxis17=dict(title="p (mV)"),
                          xaxis18=dict(title="p (mV)"),
                          xaxis19=dict(title="p (mV)"),
                          xaxis20=dict(title="p (mV)"),
                          xaxis21=dict(title="p (mV)"),

                          # xaxis8=dict(title="taue (ms)"), yaxis8=dict(title="taui (ms)"),
                          # xaxis9=dict(title="taue (ms)"),
                          # xaxis10=dict(title="taue (ms)"),
                          # xaxis11=dict(title="taue (ms)"),
                          # xaxis12=dict(title="taue (ms)"),
                          # xaxis13=dict(title="taue (ms)"),
                          # xaxis14=dict(title="taue (ms)"),
                          #
                          #
                          # xaxis15=dict(title="He (mV)"), yaxis15=dict(title="Cie"),
                          # xaxis16=dict(title="He (mV)"),
                          # xaxis17=dict(title="He (mV)"),
                          # xaxis18=dict(title="He (mV)"),
                          # xaxis19=dict(title="He (mV)"),
                          # xaxis20=dict(title="He (mV)"),
                          # xaxis21=dict(title="He (mV)"),
                          #
                          # xaxis22=dict(title="p (mV)"), yaxis22=dict(title="Cee|Cie"),
                          # xaxis23=dict(title="p (mV)"),
                          # xaxis24=dict(title="p (mV)"),
                          # xaxis25=dict(title="p (mV)"),
                          # xaxis26=dict(title="p (mV)"),
                          # xaxis27=dict(title="p (mV)"),
                          # xaxis28=dict(title="p (mV)"),
                          title="Frequency charts   _" + mode)

        pio.write_html(fig, file=main_folder + simulations_tag + "/FreqCharts_" + mode + ".html", auto_open=True)



        # Save dataframe per cond
        # df.loc[df["mode"]==mode].to_csv(main_folder + simulations_tag + "/FreqCharts_" + mode + ".csv")