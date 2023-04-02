
import os

import pandas as pd
import numpy as np
import time
from itertools import chain

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\mpi_jr\PSE\\'
simulations_tag = "PSEmpi_ADpg_v2-m02d28y2023-t23h.08m.17s"  # Tag cluster job
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")

# Average out repetitions
df_avg = df.groupby(["subject", "model",  "g", "s", "sigma", "band"]).mean().reset_index()

##   PLOT PER GROUP CONDITION  -
title = "CONDs_paramSpace"
fig = make_subplots(rows=6, cols=6, horizontal_spacing=0.05,
                    row_titles=("HC-fam", "FAM", "HC","QSM", "MCI", "MCI-conv"),
                    specs=[[{}, {}, {}, {}, {}, {}]]*6, shared_yaxes=True, shared_xaxes=True,
                    x_title="Conduction speed (m/s)", y_title="Coupling factor",
                    column_titles=["rPLV emp-sim", "bifurcation", "IAF", "Power", "PLV mean", "PLV sd"])

len, space, y, pad = 0.12, 0.171, 0.94, 0.05

for j, cond in enumerate(["HC-fam", "FAM", "HC", "QSM", "MCI", "MCI-conv"]):

    subset = df_avg.loc[(df_avg["subject"] == cond) & (df_avg["band"] == "3-alpha") & (df_avg["sigma"] == 0)]

    sl = True if j == 0 else False

    fig.add_trace(
        go.Heatmap(z=subset.rPLV, x=subset.s, y=subset.g, colorscale='RdBu', reversescale=True,
                   zmin=-0.5, zmax=0.5, showscale=sl, colorbar=dict(thickness=7, len=len, x=-pad+space*1, y=y)),
        row=(1 + j), col=1)

    fig.add_trace(
        go.Heatmap(z=subset.max_cx-subset.min_cx, x=subset.s, y=subset.g, colorscale='Viridis', showscale=sl,
                   colorbar=dict(thickness=7, len=len, x=-pad+space*2, y=y),
                   zmin=df_avg.min_cx.min(), zmax=15),
        row=(1 + j), col=2)

    fig.add_trace(
        go.Heatmap(z=subset.IAF, x=subset.s, y=subset.g, showscale=sl,
                   colorbar=dict(thickness=7, len=len, x=-pad+space*3, title="Hz", y=y),
                   zmin=df_avg.IAF.min(), zmax=df_avg.IAF.max()),
        row=(1 + j), col=3)

    fig.add_trace(
        go.Heatmap(z=subset.bModule, x=subset.s, y=subset.g, colorscale='Viridis', showscale=sl,
                   colorbar=dict(thickness=7, len=len, x=-pad+space*4, title="dB", y=y),
                   zmin=df_avg.bModule.min(), zmax=1.5),
        row=(1 + j), col=4)

    fig.add_trace(
        go.Heatmap(z=subset.plv_avg, x=subset.s, y=subset.g, colorscale='Turbo', showscale=sl,
                   colorbar=dict(thickness=7, len=len, x=-pad+space*5, y=y),
                   zmin=df_avg.plv_avg.min(), zmax=df_avg.plv_avg.max()),
        row=(1 + j), col=5)

    fig.add_trace(
        go.Heatmap(z=subset.plv_sd, x=subset.s, y=subset.g, colorscale='Turbo', showscale=sl,
                   colorbar=dict(thickness=7, len=len, x=-pad+space*6+0.025, y=y),
                   zmin=df_avg.plv_sd.min(), zmax=df_avg.plv_sd.max()),
        row=(1 + j), col=6)


fig.update(frames=[go.Frame(data=list(chain.from_iterable([

    [go.Heatmap(z=df_avg.loc[(df_avg["subject"] == cond) & (df_avg["band"] == "3-alpha") & (df_avg["sigma"] == sigma)].rPLV),

    go.Heatmap(
        z=df_avg.loc[(df_avg["subject"] == cond) & (df_avg["band"] == "3-alpha") & (df_avg["sigma"] == sigma)].max_cx -
          df_avg.loc[(df_avg["subject"] == cond) & (df_avg["band"] == "3-alpha") & (df_avg["sigma"] == sigma)].min_cx),

     go.Heatmap(z=df_avg.loc[(df_avg["subject"] == cond) & (df_avg["band"] == "3-alpha") & (df_avg["sigma"] == sigma)].IAF),
     go.Heatmap(z=df_avg.loc[(df_avg["subject"] == cond) & (df_avg["band"] == "3-alpha") & (df_avg["sigma"] == sigma)].bModule),
     go.Heatmap(z=df_avg.loc[(df_avg["subject"] == cond) & (df_avg["band"] == "3-alpha") & (df_avg["sigma"] == sigma)].plv_avg),
     go.Heatmap(z=df_avg.loc[(df_avg["subject"] == cond) & (df_avg["band"] == "3-alpha") & (df_avg["sigma"] == sigma)].plv_sd)]

                            for j, cond in enumerate(["HC-fam", "FAM", "HC", "QSM", "MCI", "MCI-conv"])])),

    traces=list(np.arange(6*6)), name=str(i)) for i, sigma in enumerate(sorted(set(df_avg.sigma)))])

fig.update_layout(title_text=title + " . alpha band",
                  height=1800,
                  sliders=[dict(
                      steps=[
                          dict(method='animate',
                               args=[[str(i)], dict(mode="immediate", frame=dict(duration=100, redraw=True,
                                                                                 easing="cubic-in-out"),
                                                    transition=dict(duration=300))], label=str(sigma)) for i, sigma
                          in enumerate(sorted(set(df_avg.sigma)))],
                      transition=dict(duration=100), x=0.35, xanchor="left", y=1.1,
                      currentvalue=dict(font=dict(size=15), prefix="Sigma - ", visible=True, xanchor="right"),
                      len=0.5, tickcolor="white")],
                  )
pio.write_html(fig, file=main_folder + simulations_tag + "/" + title + "-g&s&sigma.html", auto_open=True, auto_play=False)







# ##   PLOT PER SUBJECT  -
# if os.path.isdir(main_folder + simulations_tag + "\\perSubject") == False:
#     os.mkdir(main_folder + simulations_tag + "\\perSubject")
#
# bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
#
# for subject in list(set(df_avg.subject)):
#
#     title = subject + "_paramSpace"
#
#     fig = make_subplots(rows=1, cols=5,
#                         column_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma"),
#                         specs=[[{}, {}, {}, {}, {}]],
#                         shared_yaxes=True, shared_xaxes=True,
#                         x_title="Conduction speed (m/s)", y_title="Coupling factor")
#
#     for j, band in enumerate(bands[0]):
#
#         subset = df_avg.loc[(df_avg["subject"] == subject) & (df_avg["band"] == band)]
#
#         sl = True if j == 0 else False
#
#         fig.add_trace(
#             go.Heatmap(z=subset.rPLV, x=subset.s, y=subset.g, colorscale='RdBu', reversescale=True,
#                        zmin=-0.5, zmax=0.5, showscale=sl, colorbar=dict(thickness=7)),
#             row=1, col=(1 + j))
#
#     fig.update_layout(title_text=title)
#     pio.write_html(fig, file=main_folder + simulations_tag + "/perSubject/" + title + "-g&s.html", auto_open=False)
#






#
# ## PLOT AVERAGE LINES -
#
# df_groupavg = df.groupby(["model", "th", "cer", "g", "sigma"]).mean().reset_index()
# df_groupstd = df.groupby(["model", "th", "cer", "g", "sigma"]).std().reset_index()
#
# ## Line plots for wo, w, and wp thalamus. Solid lines for active and dashed for passive.
# cmap_s = px.colors.qualitative.Set1
# cmap_p = px.colors.qualitative.Pastel1
#
# # Graph objects approach
# fig_lines = make_subplots(rows=1, cols=3, subplot_titles=("without Thalamus", "Thalamus single node", "Thalamus parcellated"),
#                           specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
#                           shared_yaxes=True, shared_xaxes=True, x_title="Coupling factor")
#
# for i, th in enumerate(structure_th):
#
#     sl = True if i < 1 else False
#
#     # Plot rPLV - active
#     df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0.022)]
#     fig_lines.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name='rPLV - active', legendgroup='rPLV - active', mode="lines",
#                                    line=dict(width=4, color=cmap_p[1]), showlegend=sl), row=1, col=1+i)
#
#     # Plot rPLV - passive
#     df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0)]
#     fig_lines.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name='rPLV - passive', legendgroup='rPLV - passive', mode="lines",
#                                    line=dict(dash='dash', color=cmap_s[1]), showlegend=sl), row=1, col=1+i)
#
#     # Plot dFC_KSD - active
#     df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0.022)]
#     fig_lines.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, name='dFC_KSD - active', legendgroup='dFC_KSD - active', mode="lines",
#                                    line=dict(width=4, color=cmap_p[0]), showlegend=sl), secondary_y=True, row=1, col=1+i)
#
#     # Plot dFC_KSD - passive
#     df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0)]
#     fig_lines.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, name='dFC_KSD - passive', legendgroup='dFC_KSD - passive', mode="lines",
#                                    line=dict(dash='dash', color=cmap_s[0]), showlegend=sl), secondary_y=True, row=1, col=1+i)
#
# fig_lines.update_layout(template="plotly_white", title="Importance of thalamus parcellation and input (avg. 10 subjects)",
#                         yaxis1=dict(title="<b>rPLV<b>", color=cmap_s[1]), yaxis2=dict(title="<b>KSD<b>", color=cmap_p[0]),
#                         yaxis3=dict(title="<b>rPLV<b>", color=cmap_s[1]), yaxis4=dict(title="<b>KSD<b>", color=cmap_p[0]),
#                         yaxis5=dict(title="<b>rPLV<b>", color=cmap_s[1]), yaxis6=dict(title="<b>KSD<b>", color=cmap_p[0]))
#
# pio.write_html(fig_lines, file=main_folder + simulations_tag + "/lineSpace-g&FC.html", auto_open=True)
#
#
#
#
#
#
#
#
# ## Maximum rPLV - statistical group comparisons
# # Extract best rPLV per subject and structure
# df_max = df_avg.groupby(["subject", "model", "th", "cer", "ct", "act"]).max().reset_index()
#
#
# from statsmodels.stats.anova import AnovaRM
# anova = AnovaRM(df_max, depvar="rPLV", subject="subject", within=["th", "model"]).fit().anova_table
#
# import pingouin as pg
# pg.plot_paired(df_max, dv="rPLV", within="th", subject="subject",)
# pg.plot_paired(df_max, dv="rPLV", within="th", subject="subject",)
#
#
#
# # JR stats
# df_max_jr = df_max.loc[df_max["model"] == "jr"]
# anova_jr = AnovaRM(df_max_jr, depvar="rPLV", subject="subject", within=["ct", "act", "th", "cer"]).fit().anova_table
#
# pwc_jr_ct = pg.pairwise_ttests(df_max_jr, dv="rPLV", within=["ct"], subject="subject")
# pwc_jr_th = pg.pairwise_ttests(df_max_jr, dv="rPLV", within=["th"], subject="subject")
#
# fig = px.box(df_max_jr, x="th", y="rPLV", color="cer", facet_col="ct", facet_row="act",
#              category_orders={"cer": ["woCer", "Cer", "pCer"], "th": ["pTh", "Th", "woTh"]}, title="JR model -")
# pio.write_html(fig, file=main_folder + simulations_tag + "/JR_rPLV_performance-boxplots.html", auto_open=True)
#
#
#
# # JRD stats
# df_max_jrd = df_max.loc[df_max["model"] == "jrd"]
# anova_jrd = AnovaRM(df_max_jrd, depvar="rPLV", subject="subject", within=["ct", "act", "th", "cer"]).fit().anova_table
#
# pwc_jrd_ct = pg.pairwise_ttests(df_max_jrd, dv="rPLV", within=["ct"], subject="subject")
# pwc_jrd_act = pg.pairwise_ttests(df_max_jrd, dv="rPLV", within=["act"], subject="subject")
# # pwc_jrd_th = pg.pairwise_ttests(df_max_jrd, dv="rPLV", within=["th"], subject="subject") Just 2 categories
#
#
# df_max_jrd = df_max.loc[df_max["model"] == "jrd"]
# fig = px.box(df_max_jrd, x="th", y="rPLV", color="cer", facet_col="ct", facet_row="act",
#              category_orders={"cer": ["woCer", "Cer", "pCer"], "th": ["pTh", "Th"]}, title="JRD model -")
# pio.write_html(fig, file=main_folder + simulations_tag + "/JRD_rPLV_performance-boxplots.html", auto_open=True)
#
#
#
#
#
#
#
#
#
#
#
#
# # dataframe for lineplot with err bars
# df_line = df_max.groupby(["th", "cer", "ct", "act"]).mean()
# df_line["std"] = df_max.groupby(["th", "cer", "ct", "act"]).std()["rPLV"]
# df_line = df_line.reset_index()
#
# fig = px.line(df_line, x="cer", y="rPLV", color="th", facet_col="ct", facet_row="act", error_y="std")
# fig.show(renderer="browser")
#
#
#
# anova = pg.rm_anova(df_max, dv="rPLV", subject="subject", within=["ct", "th"])
#
# anova = pg.rm_anova(df_max, dv="rPLV", within=["ct", "act"], subject="subject",)
# anova = pg.rm_anova(df_max, dv="rPLV", within=["th", "cer"], subject="subject",)
#
#
#
#
# pwc = pg.pairwise_ttests(df_max, dv="rPLV", within=["ct"], subject="subject")
# pwc = pg.pairwise_ttests(df_max, dv="rPLV", within=["th"], subject="subject")
#
#
#
#
#
#
#
#
# for mode in modes:
#     df_temp = df.loc[df["Mode"] == mode]
#     df_temp = df_temp.groupby(["G", "speed"]).mean().reset_index()
#     (g, s) = df_temp.groupby(["G", "speed"]).mean().idxmax(axis=0).rPLV
#
#     specific_folder = main_folder + "\\PSE_allWPs-AVGg" + str(g) + "s" + str(s) + "_" + mode + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
#     os.mkdir(specific_folder)
#
#     for subj in list(set(df.Subject)):
#
#         # subset data per mode and subject
#         df_temp = df.loc[(df["Subject"] == subj) & (df["Mode"] == mode)]
#
#         # Avg repetitions
#         df_temp = df_temp.groupby(["G", "speed", "band"]).mean().reset_index()
#         df_temp.drop("rep", inplace=True, axis=1)
#
#         # Calculate WP
#         (g, s) = df_temp.groupby(["G", "speed"]).mean().idxmax(axis=0).rPLV
#
#         name = subj + "_" + mode + "-g" + str(g) + "s" + str(s)
#
#         # save data
#         df_temp.to_csv(specific_folder + "/" + name +"-3reps.csv")
#
#         # plot paramspace
#         WPplot(df_temp, z=0.5, title=name, type="linear", folder=specific_folder, auto_open=False)
#
#
# # Plot 3 by 3 Alpha PSEs
# for subj in list(set(df.Subject)):
#
#     fig_thcer = make_subplots(rows=3, cols=3, column_titles=("Parcelled Thalamus", "Single node Thalamus", "Without Thalamus"),
#                            row_titles=("Parcelled Cerebellum", "Single node Cerebellum", "Without Cerebellum"),
#                         specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
#                         x_title="Conduction speed (m/s)", y_title="Coupling factor")
#
#     df_sub = df.loc[(df["band"] == "3-alpha") & (df["Subject"] == subj)]
#
#     for i, mode in enumerate(modes):
#
#         df_temp = df_sub.loc[df_sub["Mode"] == mode]
#
#         df_temp = df_temp.groupby(["G", "speed", "band"]).mean().reset_index()
#         df_temp.drop("rep", inplace=True, axis=1)
#
#         fig_thcer.add_trace(go.Heatmap(z=df_temp.rPLV, x=df_temp.speed, y=df_temp.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
#                              reversescale=True, zmin=-0.5, zmax=0.5), row=(i+3)//3, col=i%3+1)
#
#
#     fig_thcer.update_layout(
#         title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % subj)
#     pio.write_html(fig_thcer, file=main_folder + "/ThCer_paramSpace-g&s_%s.html" % subj, auto_open=True)
#
