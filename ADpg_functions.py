
import time
import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats
from mne import filter

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from tvb.simulator.models.JansenRit_WilsonCowan import JansenRit_WilsonCowan



## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys

    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order

## Folder structure - CLUSTER
else:
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order

    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"


class ProteinSpreadModel:
    # Spread model variables following Alexandersen 2022.
    # M is an arbitrary unit of concentration

    def __init__(self, AB_initMap, TAU_initMAp, ABt_initMap, TAUt_initMap, rho=0.001, toxicSynergy=2,
                 prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
                 prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66):

        self.rho = {"label": "rho", "value": np.array([rho]), "doc": "effective diffusion constant (cm/year)"}

        self.prodAB = {"label": ["k0", "a0"], "value": np.array([prodAB]), "doc": "production rate for a-beta (M/year)"}
        self.clearAB = {"label": ["k1", "a1"], "value": np.array([clearAB]), "doc": "clearance rate for a-beta (1/M*year)"}
        self.transAB2t = {"label": ["k2", "a2"], "value": np.array([transAB2t]),
                     "doc": "transformation of a-beta into its toxic variant (M/year)"}
        self.clearABt = {"label": ["k1t", "a1t"], "value": np.array([clearABt]), "doc": "clearance rate for toxic a-beta (1/M*year)"}

        self.prodTAU = {"label": ["k3", "b0"], "value": np.array([prodTAU]), "doc": "production rate for p-tau (M/year)"}
        self.clearTAU = {"label": ["k4", "b1"], "value": np.array([clearTAU]), "doc": "clearance rate for p-tau (1/M*year)"}
        self.transTAU2t = {"label": ["k5", "b2"], "value": np.array([transTAU2t]),
                      "doc": "transformation of p-tau into its toxic variant (M/year)"}
        self.clearTAUt = {"label": ["k4t", "b1t"], "value": np.array([clearTAUt]), "doc": "clearance rate for toxic p-tau (1/M*year)"}

        self.toxicSynergy = {"label": ["k6", "b3"], "value": np.array([toxicSynergy]),
                        "doc": "synergistic effect between toxic a-beta and toxic p-tau production (1/M^2*Year"}

        self.AB_initMap = {"label": "", "value": AB_initMap, "doc": "mapping of initial roi concentration of AB"}
        self.TAU_initMap = {"label": "", "value": TAU_initMAp, "doc": "mapping of initial roi concentration of TAU"}

        self.ABt_initMap = {"label": "", "value": ABt_initMap, "doc": "mapping of initial roi concentration of AB toxic"}
        self.TAUt_initMap = {"label": "", "value": TAUt_initMap, "doc": "mapping of initial roi concentration of TAU toxic"}

    def Laplacian(self, conn):
        # Weighted adjacency, Diagonal and Laplacian matrices
        Wij = np.divide(conn.weights, np.square(conn.tract_lengths),
                        where=np.square(conn.tract_lengths) != 0,  # Where to compute division; else out
                        out=np.zeros_like(conn.weights))  # array allocation
        Dii = np.eye(len(Wij)) * np.sum(Wij, axis=0)
        Lij = Dii - Wij

        return Lij

    def dfun(self, state_variables, Lij):
        # Here we want to model the spread of proteinopathies.
        # Approach without activity dependent spread/generation. Following Alexandersen 2022.

        AB = state_variables[0]
        ABt = state_variables[1]
        TAU = state_variables[2]
        TAUt = state_variables[3]

        # Derivatives
        ###  Amyloid-beta
        dAB = -self.rho["value"] * np.sum(Lij * AB, axis=1) + self.prodAB["value"] - self.clearAB["value"] * AB - self.transAB2t["value"] * AB * ABt
        dABt = -self.rho["value"] * np.sum(Lij * ABt, axis=1) - self.clearABt["value"] * ABt + self.transAB2t["value"] * AB * ABt

        ###  phosphorilated Tau
        dTAU = -self.rho["value"] * np.sum(Lij * TAU, axis=1) + self.prodTAU["value"] - self.clearTAU["value"] * TAU - \
               self.transTAU2t["value"] * TAU * TAUt - self.toxicSynergy["value"] * ABt * TAU * TAUt
        dTAUt = -self.rho["value"] * np.sum(Lij * TAUt, axis=1) - self.clearTAUt["value"] * TAUt + self.transTAU2t[
            "value"] * TAU * TAUt + self.toxicSynergy["value"] * ABt * TAU * TAUt

        derivative = np.array([dAB, dABt, dTAU, dTAUt])

        return derivative

    def run(self, conn, time, dt, sim=False):

        ## 1. Initiate state variables
        state_variables = [self.AB_initMap["value"],
                           self.ABt_initMap["value"],
                           self.TAU_initMap["value"],
                           self.TAUt_initMap["value"]]

        if sim:
            subj, model, g, s, simLength = sim
            _, _, plv, _, plv_r, _, _, _, reqtime = simulate(subj, model, g, s, p_th=0.09, sigma=0, t=simLength)
            BNMevo = [plv]

        Lij = self.Laplacian(conn)

        ## 2. loop over time
        print("Simulating protein spread  . for %0.2fts (dt=%0.2f)   _simulate: %s" % (time, dt, sim ))
        evolution = [np.array(state_variables.copy())]
        for t in np.arange(dt, time, dt):
            deriv = self.dfun(state_variables, Lij)
            state_variables = state_variables + dt * deriv
            evolution.append(state_variables)

            if sim:
                _, _, plv, _, plv_r, _, _, _, reqtime = simulate(subj, model, g, s, p_th=0.09, sigma=0, t=simLength)
                BNMevo.append(plv)
                print("   . ts%0.2f/%0.2f  _  SIMULATION REQUIRED %0.2f seconds  -  rPLV(%0.2f)" % (
                    t, time, reqtime, plv_r), end="\r")
            else:
                print("   . ts%0.2f/%0.2f" % (t, time), end="\r")

        if sim:
            print("   . ts%0.2f/%0.2f  _  SIMULATION REQUIRED %0.2f seconds  -  rPLV(%0.2f)" % (
                t, time, reqtime, plv_r))
            return [np.arange(0, time, dt), evolution, BNMevo]

        else:
            print("   . ts%0.2f/%0.2f" % (
                t, time))
            return [np.arange(0, time, dt), evolution]


def animate_propagation_v2(output, conn, timeref=True):

    # Create text labels per ROI
    hovertext3d = [["<b>" + roi + "</b><br>"
                    + str(round(output[1][ii][0, i], 5)) + "(M) a-beta<br>"
                    + str(round(output[1][ii][1, i], 5)) + "(M) a-beta toxic <br>"
                    + str(round(output[1][ii][2, i], 5)) + "(M) pTau <br>"
                    + str(round(output[1][ii][3, i], 5)) + "(M) pTau toxic <br>"
                    for i, roi in enumerate(conn.region_labels)] for ii, t in enumerate(output[0])]

    sz_ab, sz_t = 22, 7  # Different sizes for AB and pT nodes

    ## ADD INITIAL TRACE for 3dBrain - t0
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {}]], column_widths=[0.6, 0.4],
                        subplot_titles=(['<b>Protein accumulation dynamics</b>', '']))

    # Add trace for AB
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="AB", showlegend=True, legendgroup="AB",
                               marker=dict(size=np.abs(output[1][0][0, :]) * sz_ab, color=output[1][0][0, :], opacity=0.5, cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="YlOrBr")), row=1, col=1)
    # Add trace for ABt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="AB toxic", showlegend=True, legendgroup="AB toxic",
                               marker=dict(size=np.abs(output[1][0][1, :]) * sz_ab, color=output[1][0][1, :], opacity=0.5, cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="YlOrRd")), row=1, col=1)
    # Add trace for TAU
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="pTAU", showlegend=True, legendgroup="pTAU",
                               marker=dict(size=np.abs(output[1][0][2, :]) * sz_t, color=output[1][0][2, :], opacity=1, cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="BuPu", symbol="diamond")), row=1, col=1)
    # Add trace for TAUt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="pTAU toxic", showlegend=True, legendgroup="pTAU toxic",
                               marker=dict(size=np.abs(output[1][0][3, :]) * sz_t, color=output[1][0][3, :], opacity=1, cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="Greys", symbol="diamond")), row=1, col=1)


    ## ADD INITIAL TRACE for lines
    sim_pet_avg = np.average(np.asarray(output[1]), axis=2)
    cmap_s, cmap_d = px.colors.qualitative.Set2, px.colors.qualitative.Dark2

    # Add static lines
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 0], mode="lines", name="AB", legendgroup="AB",
                             line=dict(color=cmap_s[1], width=3), showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 1], mode="lines", name="AB toxic", legendgroup="AB toxic",
                             line=dict(color=cmap_s[3], width=3), showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 2], mode="lines", name="pTAU", legendgroup="pTAU",
                             line=dict(color=cmap_s[2],  width=3), showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 3], mode="lines", name="pTAU toxic", legendgroup="pTAU toxic",
                             line=dict(color=cmap_s[7],  width=3), showlegend=True), row=1, col=2)

    # Add dynamic reference - t0
    if timeref:
        fig.add_trace(go.Scatter(x=[output[0][0]], y=[sim_pet_avg[0, 0]], mode="markers", legendgroup="AB", marker=dict(color=cmap_d[1]), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[output[0][0]], y=[sim_pet_avg[0, 1]], mode="markers", legendgroup="AB toxic", marker=dict(color=cmap_d[3]), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[output[0][0]], y=[sim_pet_avg[0, 2]], mode="markers", legendgroup="pTAU", marker=dict(color=cmap_d[2]), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[output[0][0]], y=[sim_pet_avg[0, 3]], mode="markers", legendgroup="pTAU toxic", marker=dict(color=cmap_d[7]), showlegend=False), row=1, col=2)

    # fig.show("browser")


    ## ADD FRAMES - t[1:end]
    fig.update(frames=[go.Frame(data=[
        go.Scatter3d(hovertext=hovertext3d[i], marker=dict(size=np.abs(output[1][i][0, :]) * sz_ab, color=output[1][i][0, :])),
        go.Scatter3d(hovertext=hovertext3d[i], marker=dict(size=np.abs(output[1][i][1, :]) * sz_ab, color=output[1][i][1, :])),
        go.Scatter3d(hovertext=hovertext3d[i], marker=dict(size=np.abs(output[1][i][2, :]) * sz_t, color=output[1][i][2, :])),
        go.Scatter3d(hovertext=hovertext3d[i], marker=dict(size=np.abs(output[1][i][3, :]) * sz_t, color=output[1][i][3, :])),

        go.Scatter(x=[output[0][i]], y=[sim_pet_avg[i, 0]]),
        go.Scatter(x=[output[0][i]], y=[sim_pet_avg[i, 1]]),
        go.Scatter(x=[output[0][i]], y=[sim_pet_avg[i, 2]]),
        go.Scatter(x=[output[0][i]], y=[sim_pet_avg[i, 3]])

    ],
        traces=[0, 1, 2, 3,  8, 9, 10, 11], name=str(i)) for i, t in enumerate(output[0])])

    # CONTROLS : Add sliders and buttons
    fig.update_layout(
        template="plotly_white", legend=dict(x=1.05, y=1.1),
        scene=dict(xaxis=dict(title="Sagital axis<br>(L-R)"),
                   yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                   zaxis=dict(title="Horizontal axis<br>(inf-sup)")),
        yaxis=dict(range=[0,1]),
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(i)], dict(mode="immediate", frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                               transition=dict(duration=300))], label=str(t)) for i, t
                   in enumerate(output[0])],
            transition=dict(duration=100), x=0.15, xanchor="left", y=1.4,
            currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
            len=0.8, tickcolor="white")],
        updatemenus=[dict(type="buttons", showactive=False, y=1.35, x=0, xanchor="left",
                          buttons=[
                              dict(label="Play", method="animate",
                                   args=[None,
                                         dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"), transition=dict(duration=300),
                                              fromcurrent=True, mode='immediate')]),
                              dict(label="Pause", method="animate",
                                   args=[[None],
                                         dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"), transition=dict(duration=300),
                                              mode="immediate")])])])
    fig.show("browser")


def correlations(output, refgroups, reftype="PET", band="3-alpha"):

    # Here we wanna get a simplify array with (n_refcond, time) containing correlation values
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

    # PET correlations
    if "PET" in reftype:

        #    ADNI PET DATA       ##########
        ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)

        # Check label order
        conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/HC-fam_aparc_aseg-mni_09c.zip")
        PETlabs = list(ADNI_AVG.columns[12:])
        PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]

        # loop over refgroups: ["CN", "SMC", "EMCI", "LMCI", "AD"]
        corr = []
        for group in refgroups:

            AB_emp = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == group)].iloc[:, 12:]))
            AB_emp = AB_emp[PET_idx]

            TAU_emp = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == group)].iloc[:, 12:]))
            TAU_emp = TAU_emp[PET_idx]

            cond = []
            for i in range(len(output[1])):
                sumAB_ABt = output[1][i][0, :] + output[1][i][1, :]
                sumTAU_TAUt = output[1][i][2, :] + output[1][i][3, :]
                cond.append([np.corrcoef(AB_emp, sumAB_ABt)[0, 1], np.corrcoef(TAU_emp, sumTAU_TAUt)[0, 1]])

            corr.append(cond)

        return np.array(corr)

    # PLV correlations
    elif "PLV" in reftype:

        # loop over refgroups
        corr = []
        for group in refgroups:
            plv_emp = np.loadtxt(data_folder + "FC_matrices/" + group + "_" + band + "_plv_rms.txt", delimiter=',')

            t1 = np.zeros(shape=(2, len(plv_emp) ** 2 // 2 - len(plv_emp) // 2))
            t1[0, :] = plv_emp[np.triu_indices(len(plv_emp), 1)]

            cond = []
            for i in range(len(output[2])):
                # Comparisons
                t1[1, :] = output[2][i][np.triu_indices(len(plv_emp), 1)]
                cond.append(np.corrcoef(t1)[0, 1])

            corr.append(cond)

        return np.array(corr)

    # SC correlations
    elif "SC" in reftype:

        # loop over refgroups
        corr = []
        for group in refgroups:
            conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + group + "_aparc_aseg-mni_09c.zip")
            conn.weights = conn.scaled_weights(mode="tract")

            t1 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
            t1[0, :] = conn.weights[np.triu_indices(len(conn.region_labels), 1)]

            # correlate and save
            cond = []
            for i in range(len(output[3])):
                t1[1, :] = output[3][i][np.triu_indices(len(conn.region_labels), 1)]
                cond.append(np.corrcoef(t1)[0, 1])

            corr.append(cond)

            return np.array(corr)


def animate_propagation_v3(output, corrs, refgroups, reftype, conn, timeref=True):

    # Create text labels per ROI
    hovertext3d = [["<b>" + roi + "</b><br>"
                    + str(round(output[1][ii][0, i], 5)) + "(M) a-beta<br>"
                    + str(round(output[1][ii][1, i], 5)) + "(M) a-beta toxic <br>"
                    + str(round(output[1][ii][2, i], 5)) + "(M) pTau <br>"
                    + str(round(output[1][ii][3, i], 5)) + "(M) pTau toxic <br>"
                    for i, roi in enumerate(conn.region_labels)] for ii, t in enumerate(output[0])]

    sz_ab, sz_t = 25, 10  # Different sizes for AB and pT nodes

    ## ADD INITIAL TRACE for 3dBrain - t0
    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2, "type": "surface"}, {}], [{}, {}]],
                        column_widths=[0.6, 0.4], shared_xaxes=True,
                        subplot_titles=(['<b>Protein accumulation dynamics</b>', '', '', 'Correlations (emp-sim)']))

    # Add trace for AB + ABt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="AB", showlegend=True, legendgroup="AB",
                               marker=dict(size=(np.abs(output[1][0][0, :]) + np.abs(output[1][0][1, :])) * sz_ab, cmax=0.5, cmin=-0.25,
                                           color=np.abs(output[1][0][1, :])/np.abs(output[1][0][0, :]), opacity=0.5,
                                           line=dict(color="grey", width=1), colorscale="YlOrBr")), row=1, col=1)

    # Add trace for TAU + TAUt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="TAU", showlegend=True, legendgroup="TAU",
                               marker=dict(size=(np.abs(output[1][0][2, :]) + np.abs(output[1][0][3, :])) * sz_t, cmax=0.5, cmin=-0.25,
                                           color=np.abs(output[1][0][3, :])/np.abs(output[1][0][2, :]), opacity=1,
                                           line=dict(color="grey", width=1), colorscale="BuPu", symbol="diamond")), row=1, col=1)


    ## ADD INITIAL TRACE for lines
    sim_pet_avg = np.average(np.asarray(output[1]), axis=2)

    if timeref:
        # Add dynamic reference - t0
        min_lp, max_lp = np.min(sim_pet_avg) - 0.15, np.max(sim_pet_avg) + 0.15
        fig.add_trace(go.Scatter(x=[output[0][0], output[0][0]], y=[min_lp, max_lp], mode="lines", legendgroup="timeref",
                                 line=dict(color="black", width=1), showlegend=False), row=1, col=2)

        min_r, max_r = np.min(corrs) - 0.15, np.max(corrs) + 0.15
        fig.add_trace(go.Scatter(x=[output[0][0], output[0][0]], y=[min_r, max_r], mode="lines", legendgroup="timeref",
                                 line=dict(color="black", width=1), showlegend=False), row=2, col=2)

    # Add static lines - PET proteins concentrations
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 0], mode="lines", name="AB", legendgroup="AB",
                             line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7, showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 1], mode="lines", name="AB toxic", legendgroup="AB",
                             line=dict(color=px.colors.sequential.YlOrBr[5], width=3), opacity=0.7, showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 2], mode="lines", name="TAU", legendgroup="TAU",
                             line=dict(color=px.colors.sequential.BuPu[3],  width=3), opacity=0.7, showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 3], mode="lines", name="TAU toxic", legendgroup="TAU",
                             line=dict(color=px.colors.sequential.BuPu[5],  width=3), opacity=0.7, showlegend=True), row=1, col=2)


    # Add static lines - data correlations
    cmap_p = px.colors.qualitative.Pastel2
    for ii, group in enumerate(refgroups):
        if "PET" in reftype:
            fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 0], mode="lines", name=group + " - AB",
                                     legendgroup="corr",
                                     line=dict(color=cmap_p[ii], width=3), showlegend=True), row=2, col=2)
            
            fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 1], mode="lines", name=group + " - rTAU",
                                     legendgroup="corr",
                                 line=dict(color=cmap_p[ii], width=2, dash="dash"), visible="legendonly", showlegend=True), row=2, col=2)

        else:
            fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :], mode="lines", name=group + " - r" + reftype,
                                     legendgroup="corr",
                                 line=dict(color=cmap_p[ii], width=3), showlegend=True), row=2, col=2)


    # fig.show("browser")

    ## ADD FRAMES - t[1:end]
    if timeref:

        fig.update(frames=[go.Frame(data=[
            go.Scatter3d(hovertext=hovertext3d[i],
                         marker=dict(size=(np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                     color=np.abs(output[1][i][1, :])/np.abs(output[1][0][0, :]))),

            go.Scatter3d(hovertext=hovertext3d[i],
                         marker=dict(size=(np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                     color=np.abs(output[1][i][3, :])/np.abs(output[1][0][2, :]))),

            go.Scatter(x=[output[0][i], output[0][i]]),
            go.Scatter(x=[output[0][i], output[0][i]])
        ],
            traces=[0, 1, 2, 3], name=str(i)) for i, t in enumerate(output[0])])
    else:
        fig.update(frames=[go.Frame(data=[
            go.Scatter3d(hovertext=hovertext3d[i],
                         marker=dict(size=np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :]) * sz_ab,
                                     color=np.abs(output[1][i][1, :])/np.abs(output[1][0][0, :]))),

            go.Scatter3d(hovertext=hovertext3d[i],
                         marker=dict(size=np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :]) * sz_t,
                                     color=np.abs(output[1][i][3, :])/np.abs(output[1][0][2, :]))),
        ],
            traces=[0, 1], name=str(i)) for i, t in enumerate(output[0])])

    # CONTROLS : Add sliders and buttons
    fig.update_layout(
        template="plotly_white", legend=dict(x=1.05, y=0.55, tracegroupgap=40, groupclick="toggleitem"),
        scene=dict(xaxis=dict(title="Sagital axis<br>(L-R)"),
                   yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                   zaxis=dict(title="Horizontal axis<br>(inf-sup)")),
        xaxis1=dict(title="Time (Years)"), xaxis3=dict(title="Time (Years)"),
        yaxis1=dict(title="Concentration (M)"), yaxis3=dict(title="Pearson's corr"),
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(i)], dict(mode="immediate", frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                               transition=dict(duration=300))], label=str(t)) for i, t
                   in enumerate(output[0])],
            transition=dict(duration=100), x=0.15, xanchor="left", y=1.4,
            currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
            len=0.8, tickcolor="white")],
        updatemenus=[dict(type="buttons", showactive=False, y=1.35, x=0, xanchor="left",
                          buttons=[
                              dict(label="Play", method="animate",
                                   args=[None,
                                         dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"), transition=dict(duration=300),
                                              fromcurrent=True, mode='immediate')]),
                              dict(label="Pause", method="animate",
                                   args=[[None],
                                         dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"), transition=dict(duration=300),
                                              mode="immediate")])])])


    pio.write_html(fig, file="figures/ProteinPropagation_&corr" + reftype + ".html", auto_open=True)



def simulate(subj, model, g, s, g_wc=None, p_th=0.12, sigma=0.022, t=10):

    # Prepare simulation parameters
    simLength = t * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 2000  # ms

    tic = time.time()

    # STRUCTURAL CONNECTIVITY      #########################################

    conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
    conn.weights = conn.scaled_weights(mode="tract")
    conn.speed = np.array([s])

    # Define regions implicated in Functional analysis: not considering subcortical ROIs
    cortical_rois = ['ctx-lh-bankssts', 'ctx-rh-bankssts', 'ctx-lh-caudalanteriorcingulate',
                     'ctx-rh-caudalanteriorcingulate',
                     'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-rh-cuneus',
                     'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
                     'ctx-lh-fusiform', 'ctx-rh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
                     'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal', 'ctx-lh-insula', 'ctx-rh-insula',
                     'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                     'ctx-rh-lateraloccipital',
                     'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-rh-lingual',
                     'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-middletemporal',
                     'ctx-rh-middletemporal',
                     'ctx-lh-paracentral', 'ctx-rh-paracentral', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
                     'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-rh-parsorbitalis',
                     'ctx-lh-parstriangularis', 'ctx-rh-parstriangularis', 'ctx-lh-pericalcarine',
                     'ctx-rh-pericalcarine',
                     'ctx-lh-postcentral', 'ctx-rh-postcentral', 'ctx-lh-posteriorcingulate',
                     'ctx-rh-posteriorcingulate',
                     'ctx-lh-precentral', 'ctx-rh-precentral', 'ctx-lh-precuneus', 'ctx-rh-precuneus',
                     'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
                     'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
                     'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal', 'ctx-lh-superiorparietal',
                     'ctx-rh-superiorparietal',
                     'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'ctx-lh-supramarginal',
                     'ctx-rh-supramarginal',
                     'ctx-lh-temporalpole', 'ctx-rh-temporalpole', 'ctx-lh-transversetemporal',
                     'ctx-rh-transversetemporal']

    #  Load FC labels, transform to SC format; check if match SC.
    FClabs = list(np.loadtxt(data_folder + "FC_matrices/" + subj + "_roi_labels_rms.txt", dtype=str))
    FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    FC_cortex_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois

    # load SC labels.
    SClabs = list(conn.region_labels)
    SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

    #   NEURAL MASS MODEL  &  COUPLING FUNCTION   #########################################################
    sigma_array = np.asarray([sigma if 'Thal' in roi else 0 for roi in conn.region_labels])
    p_array = np.asarray([p_th if 'Thal' in roi else 0.09 for roi in conn.region_labels])

    if model == "jrd":  # JANSEN-RIT-DAVID
        # Parameters edited from David and Friston (2003).
        m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                               tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                               He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                               tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                               w=np.array([0.8]), c=np.array([135.0]),
                               c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                               c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                               v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                               p=np.array([p_array]), sigma=np.array([sigma_array]))

        # Remember to hold tau*H constant.
        m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
        m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

        coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=np.array([0.8]), e0=np.array([0.005]),
                                                v0=np.array([6.0]), r=np.array([0.56]))

    elif model == "jrwc":  # JANSEN-RIT(cx) + WILSON-COWAN(th)

        jrMask_wc = [[False] if 'Thal' in roi else [True] for roi in conn.region_labels]

        m = JansenRit_WilsonCowan(
            # Jansen-Rit nodes parameters. From Stefanovski et al. (2019)
            He=np.array([3.25]), Hi=np.array([22]),
            tau_e=np.array([10]), tau_i=np.array([20]),
            c=np.array([135.0]), p=np.array([p_array]),
            c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
            c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
            v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
            # Wilson-Cowan nodes parameters. From Abeysuriya et al. (2018)
            P=np.array([0.31]), sigma=np.array([sigma_array]), Q=np.array([0]),
            c_ee=np.array([3.25]), c_ei=np.array([2.5]),
            c_ie=np.array([3.75]), c_ii=np.array([0]),
            tau_e_wc=np.array([10]), tau_i_wc=np.array([20]),
            a_e=np.array([4]), a_i=np.array([4]),
            b_e=np.array([1]), b_i=np.array([1]),
            c_e=np.array([1]), c_i=np.array([1]),
            k_e=np.array([1]), k_i=np.array([1]),
            r_e=np.array([0]), r_i=np.array([0]),
            theta_e=np.array([0]), theta_i=np.array([0]),
            alpha_e=np.array([1]), alpha_i=np.array([1]),
            # JR mask | WC mask
            jrMask_wc=np.asarray(jrMask_wc))

        m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])

        coup = coupling.SigmoidalJansenRit_Linear(
            a=np.array([g]), e0=np.array([0.005]), v0=np.array([6]), r=np.array([0.56]),
            # Jansen-Rit Sigmoidal coupling
            a_linear=np.asarray([g_wc]),  # Wilson-Cowan Linear coupling
            jrMask_wc=np.asarray(jrMask_wc))  # JR mask | WC mask

    else:  # JANSEN-RIT
        # Parameters from Stefanovski 2019.
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([p_array]), sigma=np.array([sigma_array]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))

    # OTHER PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    # print("Simulating %s (%is)  ||  PARAMS: g%i sigma%0.2f" % (model, simLength / 1000, g, 0.022))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    if model == "jrd":
        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
    else:
        raw_data = output[0][1][transient:, 0, :, 0].T
    raw_time = output[0][0][transient:]

    # Further analysis on Cortical signals
    raw_data = raw_data[SC_cortex_idx, :]
    regionLabels = conn.region_labels[SC_cortex_idx]

    # PLOTs :: Signals and spectra
    # timeseries_spectra(raw_data[:], simLength, transient, regionLabels, mode="inline", freqRange=[2, 40], opacity=1)

    bands = [["3-alpha"], [(8, 12)]]
    # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

    for b in range(len(bands[0])):
        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut, verbose=False)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals", verbose=False)

        # Obtain Analytical signal
        efPhase = list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.angle(analyticalSignal))
            efEnvelope.append(np.abs(analyticalSignal))

        # Check point
        # from toolbox import timeseriesPlot, plotConversions
        # regionLabels = conn.region_labels
        # timeseriesPlot(raw_data, raw_time, regionLabels)
        # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

        # CONNECTIVITY MEASURES
        ## PLV and plot
        plv = PLV(efPhase, verbose=False)

        # Load empirical data to make simple comparisons
        plv_emp = \
            np.loadtxt(data_folder + "FC_matrices/" + subj + "_" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
            FC_cortex_idx][
                FC_cortex_idx]

        # Comparisons
        t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
        t1[0, :] = plv[np.triu_indices(len(plv), 1)]
        t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
        plv_r = np.corrcoef(t1)[0, 1]

        # ## dynamical Functional Connectivity
        # # Sliding window parameters
        # window, step = 4, 2  # seconds
        #
        # ## dFC and plot
        # dFC = dynamic_fc(raw_data, samplingFreq, transient, window, step, "PLV")
        #
        # dFC_emp = np.loadtxt(data_folder + "FC_matrices/" + subj + "_" + bands[0][b] + "_dPLV4s_rms.txt")
        #
        # # Compare dFC vs dFC_emp
        # t2 = np.zeros(shape=(2, len(dFC) ** 2 // 2 - len(dFC) // 2))
        # t2[0, :] = dFC[np.triu_indices(len(dFC), 1)]
        # t2[1, :] = dFC_emp[np.triu_indices(len(dFC), 1)]
        # dFC_ksd = scipy.stats.kstest(dFC[np.triu_indices(len(dFC), 1)], dFC_emp[np.triu_indices(len(dFC), 1)])[0]

        ## PLOTs :: PLV + dPLV
        # print("REPORT_ \nrPLV = %0.2f " % (plv_r))

    # print("SIMULATION REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return raw_data, raw_time, plv, plv_emp, plv_r, regionLabels, simLength, transient, time.time() - tic


def animate_propagation(output, conn):

    # Create text labels per ROI
    hovertext3d = [["<b>" + roi + "</b><br>"
                    + str(round(output[1][ii][0, i], 5)) + "(M) a-beta<br>"
                    + str(round(output[1][ii][1, i], 5)) + "(M) a-beta toxic <br>"
                    + str(round(output[1][ii][2, i], 5)) + "(M) pTau <br>"
                    + str(round(output[1][ii][3, i], 5)) + "(M) pTau toxic <br>"
                    for i, roi in enumerate(conn.region_labels)] for ii, t in enumerate(output[0])]

    sz_ab, sz_t = 22, 7  # Different sizes for AB and pT nodes

    ## ADD INITIAL TRACE - t0
    fig = go.Figure()
    # Add trace for AB
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="amyloid-beta", showlegend=True,
                               marker=dict(size=output[1][0][0, :] * sz_ab, color=output[1][0][0, :], opacity=0.5, cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="YlOrBr")))
    # Add trace for ABt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="a-beta toxic", showlegend=True,
                               marker=dict(size=output[1][0][1, :] * sz_ab, color=output[1][0][1, :], opacity=0.5,cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="YlOrRd")))
    # Add trace for TAU
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="pTAU", showlegend=True,
                               marker=dict(size=output[1][0][2, :] * sz_t, color=output[1][0][2, :], opacity=1,cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="BuPu", symbol="diamond")))
    # Add trace for TAUt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="pTAU toxic", showlegend=True,
                               marker=dict(size=output[1][0][3, :] * sz_t, color=output[1][0][3, :], opacity=1,cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="Greys", symbol="diamond")))
    ## ADD FRAMES - t[1:end]
    fig.update(frames=[go.Frame(data=[
        go.Scatter3d(hovertext=hovertext3d[i], marker=dict(size=output[1][i][0, :] * sz_ab, color=output[1][i][0, :])),
        go.Scatter3d(hovertext=hovertext3d[i], marker=dict(size=output[1][i][1, :] * sz_ab, color=output[1][i][1, :])),
        go.Scatter3d(hovertext=hovertext3d[i], marker=dict(size=output[1][i][2, :] * sz_t, color=output[1][i][2, :])),
        go.Scatter3d(hovertext=hovertext3d[i], marker=dict(size=output[1][i][3, :] * sz_t, color=output[1][i][3, :]))],
        traces=[0, 1, 2, 3], name=str(i)) for i, t in enumerate(output[0])])

    # CONTROLS : Add sliders and buttons
    fig.update_layout(
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(i)], dict(mode="immediate", frame=dict(duration=500, redraw=True),
                                                               transition=dict(duration=200))], label=str(t)) for i, t
                   in enumerate(output[0])],
            transition=dict(duration=100), x=0.15, xanchor="left", y=1.1,
            currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
            len=0.8, tickcolor="white")],
        updatemenus=[dict(type="buttons", showactive=False, y=1.05, x=0, xanchor="left",
                          buttons=[
                              dict(label="Play", method="animate",
                                   args=[None,
                                         dict(frame=dict(duration=500, redraw=True), transition=dict(duration=100),
                                              fromcurrent=True, mode='immediate')]),
                              dict(label="Pause", method="animate",
                                   args=[[None],
                                         dict(frame=dict(duration=500, redraw=False), transition=dict(duration=100),
                                              mode="immediate")])])])
    fig.show("browser")


