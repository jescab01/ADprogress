
import time
import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from tvb.simulator.lab import connectivity
from ADpg_functions import CircularADpgModel, animate_propagation_v4, correlations
from Calibration_functions import TransferTwo, simulations, TransferOne, propagationtrajectory_on4D

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


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



#  0. STRUCTURAL CONNECTIVITY   #########
#  Define structure through which the proteins will spread;
#  Not necessarily the same than the one used to simulate activity.
subj = "HC-fam"
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
conn.weights = conn.scaled_weights(mode="tract")  # did they normalize? maybe this affects to the spreading?

#    ADNI PET DATA       ##########
ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)

# Check label order
PETlabs = list(ADNI_AVG.columns[12:])
PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]

mode = "classic_test_circMode"
print(mode)

#  1. SETUP and SIMULATE protein dynamics  #############
# Following Alexandersen (2022)  - same parameters, same initial conditions
"""
Compute Christoffer protein propagation.
TUNE NMM parameters range -
"""
#  REGIONAL SEEDs for toxic proteins
AB_seeds = ["ctx-lh-precuneus", "ctx-lh-isthmuscingulate", "ctx-lh-insula", "ctx-lh-medialorbitofrontal", "ctx-lh-lateralorbitofrontal",
            "ctx-rh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-insula", "ctx-rh-medialorbitofrontal", "ctx-rh-lateralorbitofrontal"]
TAU_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]

AB_initMap, TAU_initMap = [[1 for roi in conn.region_labels]]*2
ABt_initMap = [0.1 / len(AB_seeds) if roi in AB_seeds else 0 for roi in conn.region_labels]
TAUt_initMap = [0.1 / len(TAU_seeds) if roi in TAU_seeds else 0 for roi in conn.region_labels]

AB_initdam, TAU_initdam = [[0 for roi in conn.region_labels]]*2
POW_initdam = [1 for roi in conn.region_labels]

## TUNNING
He, Hi, taue, taui = 3.5, 22, 10, 16
prop_dt, bnm_dt = 0.25, 4

circmodel = CircularADpgModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam, TAU_initdam, POW_initdam,
    TAU_dam2SC=2e-5, POW_damrate=0.01, maxPOWdam=1.5,
    init_He=He, init_Hi=Hi, init_taue=taue, init_taui=taui, rho=100,
    prodAB=2, clearAB=2, transAB2t=2, clearABt=1.6,
    prodTAU=2.2, clearTAU=2, transTAU2t=2, clearTAUt=1.7)

circmodel.init_He["range"] = [He-1, He+2]  # origins (2.6, 9.75) :: (-0.65+x, x+6.5)
circmodel.init_Hi["range"] = [Hi-3, Hi+18]  # origins (17.6, 40) :: (-4.4+x, x+18)
circmodel.init_taue["range"] = [taue-4, taue+10]  # origins (6, 12) :: (-4+x, x+10)
circmodel.init_taui["range"] = [taui-8, taui+20]  # origins (12, 40) :: (-8+x, x+20)

out_circ = circmodel.\
    run(time=40, dt=prop_dt, sim=[subj, "jr", 4, 15.5, 5], sim_dt=bnm_dt)
                    # sim=[False] (xor) [subj, model, g, s, time(s)](simParams)

timepoints = out_circ[0]


## Work over the results
a = [out[1:] for out in out_circ[2] if len(out) == 7]
TransferOne(out_circ, out_networkSim=a, skip=int(bnm_dt/prop_dt), mode=mode)
TransferTwo(out_circ, skip=int(bnm_dt/prop_dt), mode=mode)


propagationtrajectory_on4D(out_circ, mode, PSE3d_tag="PSEmpi_ADpg_PSE3d-m11d10y2022-t17h.09m.44s")

# corrs_PET = correlations(out_circ, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic", plot="s", title=mode)
# animate_propagation_v4(out_circ, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETtoxic", conn, timeref=True, title=mode)


## 3. Simulations with protein timepoints SINGLE NODE  / TESTing TRANSFER FUNCTION _protprop->NMM   #####
# out_avg = np.average(out_prop[1], axis=2).transpose()[6:]
# conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
#
# out_simpleSim, t0, skip = [], time.time(), 10
# for i, t in enumerate(timepoints):
#     if i % skip == 0:
#         print("SIMPLE SIMULATIONs  t%0.2f of propagation dyn  -  %i/%i     | time: %0.2fs" %
#               (t, i/skip, len(timepoints)/skip, time.time()-t0), end="\r")
#         out_simpleSim.append(simulations(out_avg[:, i], conn, mode=mode, rois="pair"))
# print("SIMPLE SIMULATIONs  t%0.2f of propagation dyn  -  %i/%i     | time: %0.2fm" % (t, i/skip, len(timepoints)/skip, (time.time()-t0)/60))
#
# Plot results
# TransferOne(out_prop, out_simpleSim, skip=skip, mode=mode)


##  4. DEEPEN into specific points of the temporal domain
# t = 18
# id = list(out_prop[0]).index(t)
# raw_data = simulations(out_avg[:, id], conn, out="signals", mode=mode, rois="pair")
#
# timeseries_spectra(raw_data, 5000, 1000, ["cx", "th"], mode="html", folder="figures",
#                        freqRange=[2, 40], opacity=1, title=None, auto_open=True)

# print("here")
# ##  5. TEST whole network TRANSFER FUNCTION _protprop->NMM    #####
# out_avg = np.moveaxis(np.array(out_prop[1]), 0, 1)[6:]
# conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
#
# out_netSim, t0, skip = [], time.time(), 10
# for i, t in enumerate(timepoints):
#     if i % skip == 0:
#         print("BNM SIMULATIONs  t%0.2f of propagation dyn  -  %i/%i     | time: %0.2fs" %
#               (t, i / skip, len(timepoints) / skip, time.time() - t0), end="\r")
#         out_netSim.append(simulations(out_avg[:, i, :], conn, mode=mode, rois="bnm"))
# print("BNM SIMULATIONs  t%0.2f of propagation dyn  -  %i/%i     | time: %0.2fm" % (
# t, i / skip, len(timepoints) / skip, (time.time() - t0) / 60))
# # Plot results
# TransferOne(out_prop, out_networkSim=out_netSim, skip=skip, mode=mode)
#
# ##  4. DEEPEN into specific points of the temporal domain
# # t = 25
# # id = list(out_prop[0]).index(t)
# # raw_data = simulations(out_avg[:, id], conn, out="signals", mode=mode, rois="bnm")
# #
# # timeseries_spectra(raw_data, 5000, 1000, conn.region_labels, mode="html", folder="figures",
# #                        freqRange=[2, 40], opacity=1, title=None, auto_open=True)
#
# propagationtrajectory_on4D(out_prop, mode, PSE3d_tag="PSEmpi_ADpg_PSE3d-m11d10y2022-t17h.09m.44s")
