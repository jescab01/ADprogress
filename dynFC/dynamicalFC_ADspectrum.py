"""
    Calculating dynamical functional connectivity distributions in the AD spectrum
"""

import numpy as np
import pandas as pd
import scipy.io
import scipy.stats

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

data_folder = "E:\LCCN_Local\PycharmProjects\ADprogress_data\\"

##   1. C3N dataset    #########################

# Read participants.csv to define subj-condition relationship
c3n_participants = pd.read_csv(data_folder + "participants.csv")
groups = ["HC-fam", "FAM", "HC", "QSM", "MCI", "MCI-conv"]

####  1.1 Create list for dFC matrices [n_groups, n_subjects, ntimes, ntimes] -
dFC_matrices, dFC_ids, min_ts = [], [], []
for group in groups:
    print('Working on dFC - %s group  ::  ' % group, end="\r")
    subset = c3n_participants.loc[c3n_participants["Joint_sample"] == group]

    dFC_temp, ids_temp = [], []
    for subj in list(set(subset.ID_BIDS)):
        print('Working on dFC - %s group  :: %s ' % (group, subj), end="\r")

        plvs = scipy.io.loadmat(data_folder + "FCavg_matrices\\" + subj + "_3-alpha_all_plvs.mat")["plvs"]

        dFC_temp.append(np.asarray([[np.corrcoef(plvs[:, :, i][np.triu_indices(len(plvs[:, :, i]), 1)],
                                       plvs[:, :, j][np.triu_indices(len(plvs[:, :, j]), 1)])[0, 1]
                           for i in range(plvs.shape[2])] for j in range(plvs.shape[2])]))

        ids_temp.append([group, subj, plvs.shape[2]])
        min_ts.append(plvs.shape[2])

    print('Working on dFC - %s group  ::  end' % group, end="\n")
    dFC_matrices.append(dFC_temp)
    dFC_ids.append(ids_temp)


## Accumulated distribution
# min_t = min(np.array(min_ts))

group_distributions = [[subj_[np.triu_indices(len(subj_), 1)] for subj_ in group_] for group_ in dFC_matrices]


## Plotting scheme 3: full. As it is low computationally expensive lets do it everything in one plot.
sp_titles = ["", "", "", "", "", ""] + groups + \
            [dFC_ids[j][i][1] for i in range(len(dFC_ids[0])) for j, group_ in enumerate(dFC_ids)]
specs = [[{"colspan": 6}, {}, {}, {}, {}, {}]] + [[{}, {}, {}, {}, {}, {}]] * 21

height_main = 0.2
heights = [height_main] + [(1-height_main)/21] * 21

cmap = px.colors.qualitative.Plotly
fig = make_subplots(rows=22, cols=6, specs=specs, subplot_titles=sp_titles, row_heights=heights)

for j, group_ in enumerate(group_distributions):

    group_accum_distr_arr = np.hstack(group_)
    fig.add_trace(go.Histogram(x=group_accum_distr_arr, name=dFC_ids[j][0][0], legendgroup=dFC_ids[j][0][0], opacity=0.7,
                               histnorm="probability", marker_color=cmap[j], showlegend=True, xbins=dict(size=0.005)), row=1, col=1)
    fig.add_trace(go.Histogram(x=group_accum_distr_arr, marker_color=cmap[j], name=dFC_ids[j][0][0], legendgroup=dFC_ids[j][0][0], opacity=0.7,
                               histnorm="probability", showlegend=False, xbins=dict(size=0.005)), row=2, col=1+j)

    for i, subj_ in enumerate(group_):
        fig.add_trace(go.Histogram(x=subj_, opacity=0.8,
                                   marker_color=cmap[j],
                                   showlegend=False, xbins=dict(size=0.005)), row=i + 3, col=j + 1)

fig.update_layout(title="Initial output", barmode='overlay', template="plotly_white", font_family="Arial", height=3000)

pio.write_html(fig, file="dynFC/figures/1first_output_full.html", auto_open=True)



## Remove bad subjects from analysis
bad_HCfam = ["sub-40",  "sub-26", "sub-32", "sub-27"]
bad_FAM = ["sub-36", "sub-30"]
group_distributions_v2 = [[subj_[np.triu_indices(len(subj_), 1)]
                        for i, subj_ in enumerate(group_) if dFC_ids[j][i][1] not in bad_HCfam+bad_FAM]
                          for j, group_ in enumerate(dFC_matrices)]

## Plotting scheme 3: full. As it is low computationally expensive lets do it everything in one plot.
fig = make_subplots(rows=22, cols=6, specs=specs, subplot_titles=sp_titles, row_heights=heights)

for j, group_ in enumerate(group_distributions_v2):

    group_accum_distr_arr = np.hstack(group_)
    fig.add_trace(go.Histogram(x=group_accum_distr_arr, name=dFC_ids[j][0][0], legendgroup=dFC_ids[j][0][0], opacity=0.7,
                               histnorm="probability", marker_color=cmap[j], showlegend=True, xbins=dict(size=0.005)), row=1, col=1)
    fig.add_trace(go.Histogram(x=group_accum_distr_arr, marker_color=cmap[j], name=dFC_ids[j][0][0], legendgroup=dFC_ids[j][0][0], opacity=0.7,
                               histnorm="probability", showlegend=False, xbins=dict(size=0.005)), row=2, col=1+j)

for j, group_ in enumerate(group_distributions):
    for i, subj_ in enumerate(group_):
        if dFC_ids[j][i][1] not in bad_HCfam+bad_FAM:
            fig.add_trace(go.Histogram(x=subj_, opacity=0.8,
                                       marker_color=cmap[j],
                                       showlegend=False, xbins=dict(size=0.005)), row=i + 3, col=j + 1)

fig.update_layout(title="Second output: after removing bad subjects",
                  barmode='overlay', template="plotly_white", font_family="Arial", height=3000)

pio.write_html(fig, file="dynFC/figures/2second_output_full.html", auto_open=True)


## Plotting scheme 1: accumulated distributions.
fig = go.Figure()
for j, group_ in enumerate(group_distributions_v2):
    group_accum_distr_arr = np.hstack(group_)
    fig.add_trace(go.Histogram(x=group_accum_distr_arr, name=dFC_ids[j][0][0], legendgroup=dFC_ids[j][0][0], opacity=0.7,
                               histnorm="probability", marker_color=cmap[j], showlegend=True, xbins=dict(size=0.005)))

fig.update_layout(title="Second output: after removing bad subjects", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 0.045]),
                  barmode='overlay', template="plotly_white", font_family="Arial")

pio.write_html(fig, file="dynFC/figures/2second_output_accum.html", auto_open=True)


## Compute some statistics:: all distributions show statistically significant differences
results, results_ksd, results_pval = [], [], []
for i, group1_ in enumerate(group_distributions_v2):
    results_temp, results_ksd_temp, results_pval_temp = [], [], []
    for j, group2_ in enumerate(group_distributions_v2):

        min_points = min(len(np.hstack(group1_)), len(np.hstack(group2_)))
        results_ksd_temp.append(scipy.stats.ks_2samp(np.hstack(group1_[:min_points]),
                                          np.hstack(group2_[:min_points]))[0])
        results_pval_temp.append(scipy.stats.ks_2samp(np.hstack(group1_[:min_points]),
                                          np.hstack(group2_[:min_points]))[1])
        results_temp.append(scipy.stats.ks_2samp(np.hstack(group1_[:min_points]),
                                          np.hstack(group2_[:min_points])))

    results.append(results_temp)
    results_ksd.append(results_ksd_temp)
    results_pval.append(results_pval_temp)

results_pval, results_ksd = np.asarray(results_pval), np.asarray(results_ksd)
del results_temp, results_ksd_temp, results_pval_temp





##   OTHER FREQUENCY BANDS   #########################

for band in ["2-theta", "3-alpha", "4-beta"]:
    ####  1.1 Create list for dFC matrices [n_groups, n_subjects, ntimes, ntimes] -
    dFC_matrices, dFC_ids, min_ts = [], [], []
    for group in groups:
        print('Working on dFC - %s group  ::  ' % group, end="\r")
        subset = c3n_participants.loc[c3n_participants["Joint_sample"] == group]

        dFC_temp, ids_temp = [], []
        for subj in list(set(subset.ID_BIDS)):
            print('Working on dFC - %s group  :: %s ' % (group, subj), end="\r")

            plvs = scipy.io.loadmat(data_folder + "FCavg_matrices\\" + subj + "_" + band + "_all_plvs.mat")["plvs"]

            dFC_temp.append(np.asarray([[np.corrcoef(plvs[:, :, i][np.triu_indices(len(plvs[:, :, i]), 1)],
                                           plvs[:, :, j][np.triu_indices(len(plvs[:, :, j]), 1)])[0, 1]
                               for i in range(plvs.shape[2])] for j in range(plvs.shape[2])]))

            ids_temp.append([group, subj, plvs.shape[2]])
            min_ts.append(plvs.shape[2])

        print('Working on dFC - %s group  ::  end' % group, end="\n")
        dFC_matrices.append(dFC_temp)
        dFC_ids.append(ids_temp)


    ## Accumulated distribution
    # min_t = min(np.array(min_ts))

    group_distributions_v2 = [[subj_[np.triu_indices(len(subj_), 1)]
                               for i, subj_ in enumerate(group_) if dFC_ids[j][i][1] not in bad_HCfam + bad_FAM]
                              for j, group_ in enumerate(dFC_matrices)]

    ## Plotting scheme 3: full. As it is low computationally expensive lets do it everything in one plot.
    sp_titles = ["", "", "", "", "", ""] + groups + \
                [dFC_ids[j][i][1] for i in range(len(dFC_ids[0])) for j, group_ in enumerate(dFC_ids)]
    specs = [[{"colspan": 6}, {}, {}, {}, {}, {}]] + [[{}, {}, {}, {}, {}, {}]] * 21

    height_main = 0.2
    heights = [height_main] + [(1-height_main)/21] * 21

    cmap = px.colors.qualitative.Plotly
    fig = make_subplots(rows=22, cols=6, specs=specs, subplot_titles=sp_titles, row_heights=heights, shared_xaxes=True)

    for j, group_ in enumerate(group_distributions_v2):

        group_accum_distr_arr = np.hstack(group_)
        fig.add_trace(go.Histogram(x=group_accum_distr_arr, name=dFC_ids[j][0][0], legendgroup=dFC_ids[j][0][0], opacity=0.7,
                                   histnorm="probability", marker_color=cmap[j], showlegend=True, xbins=dict(size=0.005)), row=1, col=1)
        fig.add_trace(go.Histogram(x=group_accum_distr_arr, marker_color=cmap[j], name=dFC_ids[j][0][0], legendgroup=dFC_ids[j][0][0], opacity=0.7,
                                   histnorm="probability", showlegend=False, xbins=dict(size=0.005)), row=2, col=1+j)

        for i, subj_ in enumerate(group_):
            fig.add_trace(go.Histogram(x=subj_, opacity=0.8,
                                       marker_color=cmap[j],
                                       showlegend=False, xbins=dict(size=0.005)), row=i + 3, col=j + 1)

    fig.update_layout(title="Distributions of PLV correlations in time - " + band, barmode='overlay', xaxis1=dict(title="Pearson's r", range=[0, 1]),
                      template="plotly_white", font_family="Arial", height=3000)

    pio.write_html(fig, file="dynFC/figures/CorrDistrs_wSubjs-" + band + ".html", auto_open=True)



    figi = go.Figure()

    for j, group_ in enumerate(group_distributions_v2):
        group_accum_distr_arr = np.hstack(group_)
        figi.add_trace(
            go.Histogram(x=group_accum_distr_arr, name=dFC_ids[j][0][0], legendgroup=dFC_ids[j][0][0], opacity=0.7,
                         histnorm="probability", marker_color=cmap[j], showlegend=True, xbins=dict(size=0.005)))

    figi.update_layout(title="Distributions of PLV correlations in time - " + band,
                       xaxis=dict(range=[0, 1], title="Pearson's r"), yaxis=dict(range=[0, 0.045]),
                       barmode='overlay', template="plotly_white", font_family="Arial")

    pio.write_html(figi, file="dynFC/figures/CorrDistrs-" + band + ".html", auto_open=True)
