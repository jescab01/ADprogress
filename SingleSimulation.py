
import time
import numpy as np
import pandas as pd
import scipy
from mne import time_frequency, filter

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995
from mpi4py import MPI
import datetime
import glob

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import multitapper, FFTpeaks, PSDplot, PSD
    from toolbox.fc import PLV
    from toolbox.signals import epochingTool
    from toolbox.mixes import timeseries_spectra

## Folder structure - CLUSTER
else:
    from toolbox import multitapper, PLV, epochingTool, FFTpeaks
    wd = "/home/t192/t192950/mpi/"
    ctb_folder = wd + "CTB_data3/"


mode, emp_subj, g, s = "jr_isolated", "NEMOS_035", 33, 15.5


def simulate(mode, emp_subj, g, s, He, Cie, Cee, p, _):

    # Prepare simulation parameters
    simLength = 30 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 10000  # ms

    # COMMON SIMULATION PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    local_results = list()

    tic = time.time()

    # STRUCTURAL CONNECTIVITY      #########################################
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    # Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
    cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                     'Insula_L', 'Insula_R',
                     'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                     'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                     'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                     'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                     'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                     'Thalamus_L', 'Thalamus_R']

    isolated_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R']

    # load text with FC rois; check if match SC
    FClabs = list(np.loadtxt(ctb_folder + "FCavg_" + emp_subj + "/roi_labels.txt", dtype=str))
    SClabs = list(conn.region_labels)


    # Subset for Cingulum Bundle
    if "cb" in mode:
        FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
        SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
        conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
        conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
        conn.region_labels = conn.region_labels[SC_cb_idx]

    elif "isolated" in mode:
        FC_cb_idx = [FClabs.index(roi) for roi in
                     isolated_rois]  # find indexes in FClabs that matches cortical_rois
        SC_cb_idx = [SClabs.index(roi) for roi in
                     isolated_rois]  # find indexes in FClabs that matches cortical_rois
        conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
        conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
        conn.region_labels = conn.region_labels[SC_cb_idx]


    # NEURAL MASS MODEL    #########################################################

    # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
    m = JansenRit1995(He=np.array([He]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([Cee]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([Cie]),
                          p=np.array([p]), sigma=np.array([0]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))


    # COUPLING FUNCTION   #########################################

    coup = coupling.SigmoidalJansenRit(a=np.array([0]), cmax=np.array([0.005]), midpoint=np.array([6]), r=np.array([0.56]))
    conn.speed = np.array([s])


    print("Simulating for Coupling factor = %i and speed = %i" % (g, s))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.

    raw_data = output[0][1][transient:, 0, :, 0].T

    raw_time = output[0][0][transient:]
    regionLabels = conn.region_labels

    # timeseries_spectra(raw_data, simLength, transient, regionLabels, folder="figures/")

    spectra, freqs = PSD(raw_data, samplingFreq)
    # PSDplot(raw_data, samplingFreq, regionLabels, type="linear", title="test", folder="figures/")

    return raw_data[0], raw_time, spectra[0], freqs


def plot_simulations(params_set, output, title=None):

    main_dir= "E:\LCCN_Local\PycharmProjects\ADprogress\PAPER\R1FrequencyCharts\SimulationDynamics\\"

    sp_titles = ["He%0.2f Cie%0.2f Cee%0.2f p%0.2f" % (set[4], set[5], set[6], set[7]) if i == 0 else "" for set in params_set for i in range(2) ]

    fig = make_subplots(rows=len(params_set), cols=2, column_widths=[0.7, 0.3], subplot_titles=sp_titles, shared_xaxes=True)
    cmap = px.colors.qualitative.Plotly
    for i, out in enumerate(output):

        raw_data, raw_time, spectra, freqs = out

        fig.add_trace(go.Scatter(x=raw_time[-2000:], y=raw_data[-2000:], line=dict(color=cmap[i]),
                                 name=params_set[i][-1], legendgroup=params_set[i][-1]), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=freqs[(freqs > 1) & (freqs < 40)], y=spectra[(freqs > 1) & (freqs < 40)],
                                 legendgroup=params_set[i][-1], showlegend=False,
                                 line=dict(color=cmap[i])), row=i+1, col=2)

    fig.update_layout(template="plotly_white", title=title)
    fig["layout"]["xaxis"+str(len(params_set)*2-1)]["title"] = "Time (ms)"
    fig["layout"]["xaxis"+str(len(params_set)*2)]["title"] = "Frequency (Hz)"

    pio.write_html(fig, main_dir + "SimDyn_" + title + ".html", auto_open=True)


# Param set for Chart 1
                                         # g, speed, He,  Cie,  Cee,  p,   name
params_set = [["jr_isolated", "NEMOS_035", 0, 15.5, 3.25, 33.75, 108, 0.22, "ref"],
             ["jr_isolated", "NEMOS_035", 0, 15.5, 2, 50, 108, 0.22, "a"],
             ["jr_isolated", "NEMOS_035", 0, 15.5, 2.75, 33.75, 108, 0.22, "b"],
             ["jr_isolated", "NEMOS_035", 0, 15.5, 5, 33.75, 108, 0.22, "c"],
              ["jr_isolated", "NEMOS_035", 0, 15.5, 2, 26, 108, 0.22, "d"],
              ["jr_isolated", "NEMOS_035", 0, 15.5, 2.5, 18.5, 108, 0.22, "f"],
              ["jr_isolated", "NEMOS_035", 0, 15.5, 5, 9.5, 108, 0.22, "g"]]

output = [simulate(*set) for set in params_set]

plot_simulations(params_set, output, title="Exp. 1")


# Param set for Chart 2
                                         # g, speed, He,  Cie,  Cee,  p,   name
params_set = [["jr_isolated", "NEMOS_035", 0, 15.5, 3.25, 33.75, 108, 0.22, "ref"],
                ["jr_isolated", "NEMOS_035", 0, 15.5, 3.25, 33.75, 108, 0.14, "ref"],
                ["jr_isolated", "NEMOS_035", 0, 15.5, 3.25, 33.75, 108, 0.135, "ref"],
                ["jr_isolated", "NEMOS_035", 0, 15.5, 3.25, 33.75, 108, 0.13, "ref"],
                ["jr_isolated", "NEMOS_035", 0, 15.5, 3.25, 33.75, 108, 0.125, "ref"],
                ["jr_isolated", "NEMOS_035", 0, 15.5, 3.25, 33.75, 108, 0.12, "ref"]]

output = [simulate(*set) for set in params_set]

plot_simulations(params_set, output, title="Exp. 1")

