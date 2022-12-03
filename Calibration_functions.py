
import time
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTpeaks, FFTplot
    from toolbox.signals import epochingTool, timeseriesPlot
    from toolbox.fc import PLV
    from toolbox.mixes import timeseries_spectra

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"


def simulations(params, conn, out="fft", mode="classic", rois="bnm"):
    """
    Returning peaks and powers for every simulated region;
    Whole spectra only for the last one (due to FFTpeaks function design).

    :param params:
    :return:
    """

    # This simulation will generate FC for a virtual "Subject".
    # Define identifier (i.e. could be 0,1,11,12,...)
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

    tic0 = time.time()

    samplingFreq = 1000  # Hz
    simLength = 5000  # ms - relatively long simulation to be able to check for power distribution
    transient = 1000  # seconds to exclude from timeseries due to initial transient

    if rois == "pair":
        if "classic" in mode:
            m = JansenRit1995(He=np.array([params[0]]), Hi=np.array([params[1]]),
                              tau_e=np.array([params[2]]), tau_i=np.array([params[3]]),
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array([0.22]), sigma=np.array([0]),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

            coup = coupling.SigmoidalJansenRit(a=np.array([0]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))

        elif "prebif" in mode:
            m = JansenRit1995(He=np.array([params[0], 3.25]), Hi=np.array([params[1], 22]),
                              tau_e=np.array([params[2], 10]), tau_i=np.array([params[3], 20]),
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array([0, 0.15]), sigma=np.array([0, 0.22]),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

            # Coupling function
            coup = coupling.SigmoidalJansenRit(a=np.array([10]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))

    elif rois == "bnm":
        if "classic" in mode:
            m = JansenRit1995(He=params[0], Hi=params[1],
                              tau_e=params[2], tau_i=params[3],
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array([0.09]), sigma=np.array([0]),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

            coup = coupling.SigmoidalJansenRit(a=np.array([4]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))

        elif "prebif" in mode:

            sigma_array = [0.22 if 'Thal' in roi else 0 for roi in conn.region_labels]
            p_array = [0.15 if 'Thal' in roi else 0.09 for roi in conn.region_labels]

            m = JansenRit1995(He=params[0], Hi=params[1],
                              tau_e=params[2], tau_i=params[3],
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array(p_array), sigma=np.array(sigma_array),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

            # Coupling function
            coup = coupling.SigmoidalJansenRit(a=np.array([2]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))


    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.EulerDeterministic(dt=1000 / samplingFreq)

    conn.weights = conn.scaled_weights(mode="tract")
    conn.speed = np.array([15.5])

    if rois=="pair":
        # Subset of 2 nodes is enough
        conn.weights = conn.weights[:2][:, :2]
        conn.tract_lengths = conn.tract_lengths[:2][:, :2]
        conn.region_labels = conn.region_labels[:2]

    mon = (monitors.Raw(),)

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator,
                              monitors=mon)
    sim.configure()

    output = sim.run(simulation_length=simLength)
    # print("Simulation time: %0.2f sec" % (time.time() - tic0,))
    # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
    raw_data = output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T

    # Check initial transient and cut data
    # timeseriesPlot(raw_data, raw_time, conn.region_labels, "figures", mode="html", title=title)

    if out == "fft":
        # Fourier Analysis plot
        peaks, modules, band_modules, fft, freqs = FFTpeaks(raw_data, simLength - transient, curves=True)
        return peaks, modules, band_modules, fft, freqs

    elif out == "signals":
        return raw_data


def TransferOne(out_prop, out_simpleSim=None, out_networkSim=None, skip=1, mode="classic", folder="figures"):

    timepoints = out_prop[0]
    if "circ" in mode:
        _, ABt, _, TAUt, ABdam, TAUdam, He, Hi, taue, taui, POWdam = np.average(out_prop[1], axis=2).transpose()

    else:
        _, ABt, _, TAUt, ABdam, TAUdam, He, Hi, taue, taui = np.average(out_prop[1], axis=2).transpose()

    fig = make_subplots(rows=2, cols=3, specs=[[{"secondary_y": True}, {}, {}], [{"secondary_y": True}, {}, {}]],
                        subplot_titles=["Protein Propagation Model", "", "", "Simulated NMM dynamics", "", ""],
                        column_widths=[0.45, 0.275, 0.275], row_titles=["Power", "Frequency"], shared_xaxes=True, horizontal_spacing=0.1)

    ## 1. HEATMAPS
    main_folder = "E:\LCCN_Local\PycharmProjects\\brainModels\FrequencyChart\\data\\"
    simulations_tag = "PSEmpi_FreqCharts2.0-m11d07y2022-t17h.02m.56s"  # Tag cluster job
    # mode = "classical" if mode=="classic" else mode
    df = pd.read_csv(main_folder + simulations_tag + "/FreqCharts_classical&fixed.csv")

    df_avg = df.groupby(['mode', 'He', 'Hi', 'taue', 'taui', 'exp']).mean().reset_index()
    cmax_freq, cmin_freq = max(df_avg["roi1_Hz"].values), min(df_avg["roi1_Hz"].values)
    cmax_pow, cmin_pow = max(df_avg["roi1_auc"].values), min(df_avg["roi1_auc"].values)


    # 1.1 Add heatmaps
    # He-Hi
    Hchart_df = df.loc[(df["mode"] == "classical" + "&fixed") & (df["exp"] == "exp_H")]
    fig.add_trace(go.Heatmap(z=Hchart_df.roi1_auc, x=Hchart_df.He, y=Hchart_df.Hi, coloraxis="coloraxis1"), row=1, col=2)
    fig.add_trace(go.Heatmap(z=Hchart_df.roi1_Hz, x=Hchart_df.He, y=Hchart_df.Hi, coloraxis="coloraxis2"), row=2, col=2)

    # Taue-Taui
    tauchart_df = df.loc[(df["mode"] == "classical" + "&fixed") & (df["exp"] == "exp_tau")]
    fig.add_trace(go.Heatmap(z=tauchart_df.roi1_auc, x=tauchart_df.taue, y=tauchart_df.taui, coloraxis="coloraxis1"), row=1, col=3)
    fig.add_trace(go.Heatmap(z=tauchart_df.roi1_Hz, x=tauchart_df.taue, y=tauchart_df.taui, coloraxis="coloraxis2"), row=2, col=3)

    # 1.2 Add trajectories and hovertexts
    # HeHi power
    hovertext = ["   <b>t%0.2f</b><br>He = %0.2f  |  Hi = %0.2f<br>Power (dB) %0.4f"
                 % (timepoints[ii], He[ii], Hi[ii], Hchart_df["roi1_auc"].iloc[np.argsort(np.abs(Hchart_df["He"] - He[ii]) + np.abs(Hchart_df["Hi"] - Hi[ii])).values[0]])
                 for ii in range(len(He))]

    fig.add_trace(go.Scatter(x=He, y=Hi, mode="lines+markers", showlegend=False, hoverinfo="text",
                             hovertext=hovertext, name="HeHi-power",
                             line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7), row=1, col=2)

    fig.add_trace(go.Scatter(x=[He[0]], y=[Hi[0]], mode="markers", showlegend=False, hoverinfo="text",
                             hovertext=hovertext[0], name="HeHi-power",
                             line=dict(color="red", width=4)),
                  row=1, col=2)  # add initial point

    # He-Hi frequency
    hovertext = ["   <b>t%0.2f</b><br>He = %0.2f  |  Hi = %0.2f<br>Frequency (Hz) %0.4f"
                 % (timepoints[ii], He[ii], Hi[ii],
                    Hchart_df["roi1_Hz"].iloc[
                        np.argsort(np.abs(Hchart_df["He"] - He[ii]) + np.abs(Hchart_df["Hi"] - Hi[ii])).values[0]])
                 for ii in range(len(He))]

    fig.add_trace(go.Scatter(x=He, y=Hi, mode="lines+markers", showlegend=False, hoverinfo="text",
                             hovertext=hovertext, name="HeHi-frequency",
                             line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7),
                  row=2, col=2)

    fig.add_trace(go.Scatter(x=[He[0]], y=[Hi[0]], mode="markers", showlegend=False, hoverinfo="text",
                             hovertext=hovertext[0], name="HeHi-power",
                             line=dict(color="red", width=4)),
                  row=2, col=2)  # add initial point

    # taue-taui Power
    hovertext = ["   <b>t%0.2f</b><br>tau_e = %0.2f  |  tau_i = %0.2f<br>Power (dB) %0.4f"
                 % (timepoints[ii], taue[ii], taui[ii],
                    tauchart_df["roi1_auc"].iloc[
                        np.argsort(np.abs(tauchart_df["taue"] - taue[ii]) + np.abs(tauchart_df["taui"] - taui[ii])).values[0]])
                 for ii in range(len(He))]

    fig.add_trace(go.Scatter(x=taue, y=taui, mode="lines+markers", showlegend=False, hoverinfo="text",
                             hovertext=hovertext,
                             line=dict(color=px.colors.sequential.BuPu[3], width=3), opacity=0.7),
                  row=1, col=3)
    fig.add_trace(go.Scatter(x=[taue[0]], y=[taui[0]], mode="markers", showlegend=False, hoverinfo="text",
                             hovertext=hovertext[0],
                             line=dict(color="red", width=4)),
                  row=1, col=3)

    # taue-taui Frequency
    hovertext = ["   <b>t%0.2f</b><br>tau_e = %0.2f  |  tau_i = %0.2f<br>Frequency (Hz) %0.4f"
                 % (timepoints[ii], taue[ii], taui[ii], tauchart_df["roi1_Hz"].iloc[np.argsort(np.abs(tauchart_df["taue"] - taue[ii]) + np.abs(tauchart_df["taui"] - taui[ii])).values[0]])
                 for ii in range(len(He))]
    fig.add_trace(go.Scatter(x=taue, y=taui, mode="lines+markers", showlegend=False, hoverinfo="text",
                             hovertext=hovertext,
                             line=dict(color=px.colors.sequential.BuPu[3], width=3), opacity=0.7), row=2, col=3)
    fig.add_trace(go.Scatter(x=[taue[0]], y=[taui[0]], mode="markers", showlegend=False, hoverinfo="text",
                             hovertext=hovertext[0],
                             line=dict(color="red", width=4)), row=2, col=3)

    # 2. STATE VARIABLES
    cmap_p, cmap_s = px.colors.qualitative.Pastel, px.colors.qualitative.Pastel2
    # 2.0 Concentrations of toxic proteins
    for i, pair in enumerate([[ABt, "ABt"], [TAUt, "TAUt"]]):
        trace, name = pair
        fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="M",
                                 line=dict(width=3, color=cmap_s[i])), row=1, col=1)

    # 2.1 Damage
    for i, pair in enumerate([[ABdam, "ABdam"], [TAUdam, "TAUdam"]]):
        trace, name = pair
        fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="dam",
                                 line=dict(width=2, color=cmap_s[i]), visible="legendonly"), row=1, col=1)

    # 2.2 NMM parameters
    for i, pair in enumerate([[He, "He"], [Hi, "Hi"], [taue, "taue"]]):
        trace, name = pair
        fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="nmm",
                                 line=dict(width=3, dash="dash", color=cmap_p[i])), secondary_y=True, row=1, col=1)

    # 3. SIMULATIONS
    if out_simpleSim:
        ss_pow = [t_res[2][0] for t_res in out_simpleSim[1]]
        ss_freq = [t_res[1][0] if ss_pow[i] > 1e-5 else 0 for i, t_res in enumerate(out_simpleSim[1])]

        fig.add_trace(go.Scatter(x=timepoints[::skip], y=ss_pow, name="power_SimpleSim", legendgroup="ss", line=dict(width=4, color="gray")), row=2, col=1)
        fig.add_trace(go.Scatter(x=timepoints[::skip], y=ss_freq, name="freq_SimpleSim", legendgroup="ss", line=dict(width=2, color="silver")), secondary_y=True, row=2, col=1)

    if out_networkSim:
        ss_pow = [np.average(t_res[2]) for t_res in out_networkSim[1]]
        ss_freq = [np.average(t_res[1]) if ss_pow[i] > 1e-5 else 0 for i, t_res in enumerate(out_networkSim[1])]

        fig.add_trace(go.Scatter(x=timepoints[::skip], y=ss_pow, name="power_NetworkSim", legendgroup="netsim", line=dict(width=4, color="lawngreen")), row=2, col=1)
        fig.add_trace(go.Scatter(x=timepoints[::skip], y=ss_freq, name="freq_NetworkSim", legendgroup="netsim", line=dict(width=2, color="mediumvioletred")), secondary_y=True, row=2, col=1)

    fig.update_layout(
        xaxis2=dict(title="He (mV)"), xaxis3=dict(title="tau_e (ms)"), xaxis4=dict(title="Time (years)"),
        xaxis5=dict(title="He (mV)"), xaxis6=dict(title="tau_e (ms)"),
        yaxis1=dict(title="Protein Concentration (M)"), yaxis2=dict(title="Parameter value"),
        yaxis3=dict(title="Hi (mV)"), yaxis4=dict(title="tau_i (ms)"),
        yaxis5=dict(title="<b>Power (dB)"), yaxis6=dict(title="Frequency (Hz)", range=[0, 14]),
        yaxis7=dict(title="Hi (mV)"), yaxis8=dict(title="tau_i (ms)"),
        coloraxis1=dict(colorbar_title="dB", colorbar_x=0.97, colorbar_y=0.8, colorbar_len=0.4, colorbar_thickness=10, colorscale="Viridis", cmin=0, cmax=cmax_pow),
        coloraxis2=dict(colorbar_title="Hz", colorbar_x=0.97, colorbar_y=0.2, colorbar_len=0.4, colorbar_thickness=10, cmin=0, cmax=20),
        title="Calibration TransferOne (PP->NMM)  _" + mode, legend=dict(orientation="h"), template="plotly_white")

    pio.write_html(fig, file=folder + "/CALIB_TransferOne_" + mode + ".html", auto_open=True)


def propagationtrajectory_on4D(out_prop, mode, PSE3d_tag, folder="figures"):

    main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\TransferOne_PP2NMM\PSE\\'
    df = pd.read_pickle(main_folder + PSE3d_tag + "/results.pkl")

    df = df.astype({"He": "float", "Hi": "float", "taue": "float", "taui": "float", "meanS": "float", "freq": "float",
                    "pow": "float"})

    He_PSEvals, Hi_PSEvals, taue_PSEvals, taui_PSEvals = \
        sorted(set(df.He)), sorted(set(df.Hi)), sorted(set(df.taue)), sorted(set(df.taui))


    # define the combination of params for each timestep
    params_Prop = np.average(np.array(out_prop[1]), axis=2)[:, 6:]

    init_taue = taue_PSEvals[np.argmin(abs(params_Prop[0][2] - taue_PSEvals))]
    init_taui = taui_PSEvals[np.argmin(abs(params_Prop[0][3] - taui_PSEvals))]

    assocPSEvals_inProp = pd.DataFrame(
        np.array([(i, out_prop[0][i],
                   He_PSEvals[np.argmin(abs(params[0] - He_PSEvals))],
                   Hi_PSEvals[np.argmin(abs(params[1] - Hi_PSEvals))],
                   taue_PSEvals[np.argmin(abs(params[2] - taue_PSEvals))],
                   taui_PSEvals[np.argmin(abs(params[3] - taui_PSEvals))])
                  for i, params in enumerate(params_Prop)]), columns=["i", "t", "He", "Hi", "taue", "taui"])

    setsPSEvals_inProp = set([(He_PSEvals[np.argmin(abs(params[0] - He_PSEvals))],
                            Hi_PSEvals[np.argmin(abs(params[1] - Hi_PSEvals))],
                            taue_PSEvals[np.argmin(abs(params[2] - taue_PSEvals))],
                            taui_PSEvals[np.argmin(abs(params[3] - taui_PSEvals))])
                           for i, params in enumerate(params_Prop)])


    minmaxt_inSets = \
        pd.DataFrame(np.array([(set + (np.min(assocPSEvals_inProp["t"].
                        loc[(assocPSEvals_inProp["He"]==set[0]) & (assocPSEvals_inProp["Hi"]==set[1]) &
                            (assocPSEvals_inProp["taue"]==set[2]) & (assocPSEvals_inProp["taui"]==set[3])].values),
                 np.max(assocPSEvals_inProp["t"].loc[(assocPSEvals_inProp["He"]==set[0]) & (assocPSEvals_inProp["Hi"]==set[1]) &
                                                     (assocPSEvals_inProp["taue"]==set[2]) & (assocPSEvals_inProp["taui"]==set[3])].values)))
         for set in setsPSEvals_inProp]), columns=["He", "Hi", "taue", "taui", "tmin", "tmax"])


    df["tmin"], df["tmax"] = None, None

    for i, row in minmaxt_inSets.iterrows():
        df["tmin"].loc[(df["He"]==row.He) & (df["Hi"]==row.Hi) & (df["taue"]==row.taue) & (df["taui"]==row.taui)] = row.tmin
        df["tmax"].loc[(df["He"]==row.He) & (df["Hi"]==row.Hi) & (df["taue"]==row.taue) & (df["taui"]==row.taui)] = row.tmax



    ## PLOTTING: animation over tau_e

    df_ani = df.iloc[:, :-2].copy()
    df_ani["freq"].loc[df_ani["freq"] == 0] = None

    fig = make_subplots(rows=3, cols=3, subplot_titles=["taui==%i" % taui for taui in sorted(set(df.taui))],
                        specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                        row_titles=["Frequency (Hz)", "Power (dB)", "meanSignal (mV)"])

    for j, taui in enumerate(sorted(set(df_ani.taui))):
        subset = df_ani.loc[df["taui"] == taui]
        sl = True if j == 0 else False

        # 1. freq
        dfsub = subset.loc[subset["taue"] == init_taue].dropna()
        fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub.freq, zmin=min(df.freq), zmax=max(df.freq),
                                 colorbar=dict(len=0.3, y=0.9, thickness=15)), row=1, col=1 + j)

        # 2. pow
        fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub["pow"], colorscale="Viridis", zmin=min(df["pow"]),
                                 zmax=max(df["pow"]), colorbar=dict(len=0.3, y=0.5, thickness=15)), row=2, col=1 + j)

        # 3. mean signal
        fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub.meanS, colorscale="Cividis", zmin=min(df.meanS),
                                 zmax=max(df.meanS), colorbar=dict(len=0.3, y=0.1, thickness=15)), row=3, col=1 + j)


    for j, taui in enumerate(sorted(set(df_ani.taui))):
        # 4. Plot scatters for trajectory
        sub_traj = df[(df["taui"] == taui) & (df["taue"] == init_taue)].dropna()
        hover = ["He%0.2f, Hi%0.2f<br>taue%0.2f, taui%0.2f<br><br>tmin - %0.2f  |  tmax - %0.2f" %
                 (row.He, row.Hi, row.taue, row.taui, row.tmin, row.tmax) for i, row in sub_traj.iterrows()]
        fig.add_trace(go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=1+j)
        fig.add_trace(go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=2, col=1 + j)
        fig.add_trace(go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=3, col=1 + j)

    frames = []

    for i, taue in enumerate(sorted(set(df.taue))):
        sub = df_ani.loc[df_ani["taue"] == taue].dropna()

        sub_1 = sub.loc[sub["taui"] == 16]
        sub_2 = sub.loc[sub["taui"] == 20]
        sub_3 = sub.loc[sub["taui"] == 24]

        sub_traj = df[(df["taui"]==init_taui) & (df["taue"]==taue)].dropna()

        hover = ["He%0.2f, Hi%0.2f<br>taue%0.2f, taui%0.2f<br><br>tmin - %0.2f  |  tmax - %0.2f" %
                 (row.He, row.Hi, row.taue, row.taui, row.tmin, row.tmax) for i, row in sub_traj.iterrows()]

        frames.append(go.Frame(data=[
            go.Heatmap(x=sub_1.He, y=sub_1.Hi, z=sub_1.freq),
            go.Heatmap(x=sub_1.He, y=sub_1.Hi, z=sub_1["pow"]),
            go.Heatmap(x=sub_1.He, y=sub_1.Hi, z=sub_1.meanS),

            go.Heatmap(x=sub_2.He, y=sub_2.Hi, z=sub_2.freq),
            go.Heatmap(x=sub_2.He, y=sub_2.Hi, z=sub_2["pow"]),
            go.Heatmap(x=sub_2.He, y=sub_2.Hi, z=sub_2.meanS),

            go.Heatmap(x=sub_3.He, y=sub_3.Hi, z=sub_3.freq),
            go.Heatmap(x=sub_3.He, y=sub_3.Hi, z=sub_3["pow"]),
            go.Heatmap(x=sub_3.He, y=sub_3.Hi, z=sub_3.meanS),

            go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover),
            go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover),
            go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover)],

        traces=[0, 1, 2,  3, 4, 5,  6, 7, 8,   9, 10, 11], name=str(round(taue, 2))))

    fig.update(frames=frames)

    xaxis = dict(title="He", range=[min(df.He)-0.5, max(df.He)+0.5], autorange=False, showticklabels=True)
    yaxis = dict(title="Hi", range=[min(df.Hi)-0.5, max(df.Hi)+0.5], autorange=False, showticklabels=True)

    # CONTROLS : Add sliders and buttons
    fig.update_layout(
        title="4D parameter space - BNM simulations %s <br> init. conditions reference [He3.25, Hi22, taue=10, taui=20]" % mode,
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
                                   transition=dict(duration=0))], label=f.name, method='animate', )
                   for f in frames],
            x=0.97, xanchor="right", y=1.35, len=0.5,
            currentvalue=dict(font=dict(size=15), prefix="taue - ", visible=True, xanchor="left"),
            tickcolor="white")],
        )

    pio.write_html(fig, file=folder + "/Trajectory_onAnimatedPSE4D_"+mode+".html", auto_open=True)


def TransferTwo(out_circ, skip=1, mode="classic", folder="figures"):

    timepoints = out_circ[0]
    if "circ" in mode:
        _, ABt, _, TAUt, ABdam, TAUdam, He, Hi, taue, taui, POWdam = np.average(out_circ[1], axis=2).transpose()
    else:
        _, ABt, _, TAUt, ABdam, TAUdam, He, Hi, taue, taui = np.average(out_circ[1], axis=2).transpose()

    fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True}, {"rowspan": 2}], [{"secondary_y": True}, {}]],
                        subplot_titles=["Protein Propagation Model", "delta(Power)", "Simulated NMM dynamics", ""],
                        horizontal_spacing=0.25)

    # 2. STATE VARIABLES
    cmap_p, cmap_s = px.colors.qualitative.Pastel, px.colors.qualitative.Pastel2
    # 2.0 Concentrations of toxic proteins
    for i, pair in enumerate([[ABt, "ABt"], [TAUt, "TAUt"]]):
        trace, name = pair
        fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="M",
                                 line=dict(width=3, color=cmap_s[i])), row=1, col=1)

    # 2.1 Damage
    for i, pair in enumerate([[ABdam, "ABdam"], [TAUdam, "TAUdam"], [POWdam, "POWdam"]]):
        trace, name = pair
        fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="dam",
                                 line=dict(width=2, color=cmap_s[i]), visible="legendonly"), row=1, col=1)

    # 2.2 NMM parameters
    for i, pair in enumerate([[He, "He"], [Hi, "Hi"], [taue, "taue"]]):
        trace, name = pair
        fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="nmm",
                                 line=dict(width=3, dash="dash", color=cmap_p[i])), secondary_y=True, row=1, col=1)


    # 3. Add averaged lines for power and frequency
    ss_pow = [np.average(t_res[2]) for t_res in out_circ[2] if len(t_res) > 1]
    ss_freq = [np.average(t_res[1]) for t_res in out_circ[2] if len(t_res) > 1]

    fig.add_trace(go.Scatter(x=timepoints[::skip], y=ss_pow, name="power_NetworkSim", legendgroup="netsim", line=dict(width=4, color="lawngreen")), row=2, col=1)
    fig.add_trace(go.Scatter(x=timepoints[::skip], y=ss_freq, name="freq_NetworkSim", legendgroup="netsim", line=dict(width=2, color="mediumvioletred")), secondary_y=True, row=2, col=1)

    # 4. Add heatmap with power rise per region
    ss_pow = np.array([t_res[2] for t_res in out_circ[2] if len(t_res) > 1]).transpose()
    delta_pow = (ss_pow.transpose() / ss_pow[:, 0]).transpose()

    regionLabels = out_circ[2][0][6]
    fig.add_trace(go.Heatmap(x=timepoints[::skip], y=regionLabels, z=delta_pow, colorscale="RdBu",
                             colorbar=dict(title="delta(dB)", thickness=10, x=0.95),
                             zmid=1, reversescale=True), row=1, col=2)

    fig.update_layout(
        xaxis3=dict(title="Time (years)"), xaxis4=dict(title="Time (years)"),
        yaxis1=dict(title="Protein Concentration (M)"), yaxis2=dict(title="Parameter value"),
        yaxis4=dict(title="<b>Power (dB)"), yaxis5=dict(title="Frequency (Hz)", range=[0, 14]),
        title="Calibration TransferTwo (PP->NMM)  _" + mode, legend=dict(orientation="h"), template="plotly_white")

    pio.write_html(fig, file=folder + "/CALIB_TransferTwo_" + mode + ".html", auto_open=True)

