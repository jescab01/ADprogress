

import pickle
import time
import numpy as np
import pandas as pd
import os

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from tvb.simulator.models.JansenRit_WilsonCowan import JansenRit_WilsonCowan

from tvb.simulator.lab import connectivity
from ADpg_functions import ProteinSpreadModel

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTpeaks, FFTplot
    from toolbox.signals import epochingTool, timeseriesPlot
    from toolbox.fc import PLV

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"


#   STRUCTURAL CONNECTIVITY   #########
#  Define structure through which the proteins will spread;
#  Not necessarily the same than the one used to simulate activity.
subj = "HC-fam"
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")

#    ADNI PET DATA       ##########
ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)

# Check label order
PETlabs = list(ADNI_AVG.columns[12:])
PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]


def simpleSim(params, out="fft"):
    """
    Returning peaks and powers for every simulated region;
    Whole spectra only for the last one (due to FFTpeaks function design).

    :param params:
    :return:
    """
    He, Hi, taue, taui = params

    # This simulation will generate FC for a virtual "Subject".
    # Define identifier (i.e. could be 0,1,11,12,...)
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
    emp_subj = "NEMOS_035"

    tic0 = time.time()

    samplingFreq = 1000  # Hz
    simLength = 5000  # ms - relatively long simulation to be able to check for power distribution
    transient = 1000  # seconds to exclude from timeseries due to initial transient

    m = JansenRit1995(He=np.array(He), Hi=np.array(Hi),
                      tau_e=np.array(taue), tau_i=np.array(taui),
                      c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                      c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                      p=np.array([0.22]), sigma=np.array([0]),
                      e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.EulerDeterministic(dt=1000 / samplingFreq)

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    # Subset of 2 nodes is enough
    conn.weights = conn.weights[:2][:, :2]
    conn.tract_lengths = conn.tract_lengths[:2][:, :2]
    conn.region_labels = conn.region_labels[:2]

    # Coupling function
    coup = coupling.SigmoidalJansenRit(a=np.array([0]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))
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



## REFERENCE init params -
# init_He = 3.25, init_Hi = 22, init_taue = 10, init_taui = 16
bif, n_vals = [], 100

## 1. Bifurcation for He
He_vals = np.linspace(2.6, 9.75, n_vals)

t0 = time.time()
for i, He in enumerate(He_vals):
    print("SIMPLE SIMULATIONs  He%0.2f of propagation dyn  -  %i/%i     | time: %0.2fs" %
          (He, i+1, n_vals, time.time() - t0), end="\r")
    signals = simpleSim([He, 22, 10, 16], out="signals")
    bif.append(["He", He, np.min(signals[-1]), np.max(signals[-1])])
print("SIMPLE SIMULATIONs  He%0.2f of propagation dyn  -  %i/%i     | time: %0.2fm" % (He, i+1, n_vals, (time.time() - t0)/60))


## 2. Bifurcation for Hi
Hi_vals = np.linspace(17.6, 40, n_vals)

t0 = time.time()
for i, Hi in enumerate(Hi_vals):
    print("SIMPLE SIMULATIONs  Hi%0.2f of propagation dyn  -  %i/%i     | time: %0.2fs" %
          (Hi, i+1, n_vals, time.time() - t0), end="\r")
    signals = simpleSim([3.25, Hi, 10, 16], out="signals")
    bif.append(["Hi", Hi, np.min(signals[-1]), np.max(signals[-1])])
print("SIMPLE SIMULATIONs  Hi%0.2f of propagation dyn  -  %i/%i     | time: %0.2fm" % (Hi, i+1, n_vals, (time.time() - t0)/60))


## 3. Bifurcation for taue
taue_vals = np.linspace(6, 20, n_vals)

t0 = time.time()
for i, taue in enumerate(taue_vals):
    print("SIMPLE SIMULATIONs  taue%0.2f of propagation dyn  -  %i/%i     | time: %0.2fs" %
          (taue, i+1, n_vals, time.time() - t0), end="\r")
    signals = simpleSim([3.25, 22, taue, 16], out="signals")
    bif.append(["taue", taue, np.min(signals[-1][:-1000]), np.max(signals[-1][:-1000])])
print("SIMPLE SIMULATIONs taue%0.2f of propagation dyn  -  %i/%i     | time: %0.2fm" % (taue, i+1, n_vals, (time.time() - t0)/60))

bif_df = pd.DataFrame(bif, columns=["param", "value", "min", "max"])

## PLOT bifurcations
fig = make_subplots(rows=2, cols=3, row_heights=[0.8, 0.2])

cmap = px.colors.qualitative.Plotly
for j, param in enumerate(["He", "Hi", "taue"]):
    sub = bif_df.loc[bif_df["param"] == param]
    fig.add_trace(go.Scatter(x=sub.value.values, y=sub["min"].values, name=param, legendgroup=param, line=dict(color=cmap[j], width=1)), row=1, col=1+j)
    fig.add_trace(go.Scatter(x=sub.value.values, y=sub["max"].values, legendgroup=param, showlegend=False, line=dict(color=cmap[j], width=1)), row=1, col=1+j)
    fig.add_trace(go.Scatter(x=sub.value.values, y=sub["max"].values-sub["min"].values, legendgroup=param, showlegend=False, mode="lines", line=dict(color=cmap[j], width=2)), row=2, col=1+j)

fig.update_layout(xaxis1=dict(title="He (mV)"), xaxis2=dict(title="Hi (mV)"), xaxis3=dict(title="tau_e (ms)"),
                  xaxis4=dict(title="He (mV)"), xaxis5=dict(title="Hi (mV)"), xaxis6=dict(title="tau_e (ms)"),
                  yaxis1=dict(title="signal min&max"), yaxis2=dict(title="signal min&max"), yaxis3=dict(title="signal min&max"),
                  yaxis4=dict(title="signal max-min"), yaxis5=dict(title="signal max-min"), yaxis6=dict(title="signal max-min"),
                  template="plotly_white", title="Bifurcations over NMM parameters<br>ref: He3.25, Hi22, taue10, taui16")
pio.write_html(fig, "figures/bifurcations_overParams.html")
