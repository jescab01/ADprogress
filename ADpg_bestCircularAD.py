
import time
import numpy as np
import pandas as pd
import os

from tvb.simulator.lab import connectivity
from ADpg_functions import simulate_v2, correlations, correlations_v2, animateFC, animate_propagation_v4, CircularADpgModel, braidPlot
from Calibration_functions import TransferOne, TransferTwo, propagationtrajectory_on4D

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.mixes import timeseries_spectra

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"


#   STRUCTURAL CONNECTIVITY   #########
#  Define structure through which the proteins will spread;
#  Not necessarily the same than the one used to simulate activity.
subj = "HC-fam"
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
conn.weights = conn.scaled_weights(mode="tract")

#    ADNI PET DATA       ##########
ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)

# Check label order
PETlabs = list(ADNI_AVG.columns[12:])
PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]


#   SETUP and SIMULATE   -   PET dynamics by now  #############
# Following Alexandersen (2022)  - same initial conditions
"""

"""

#  REGIONAL SEEDs for toxic proteins
AB_seeds = ["ctx-lh-precuneus", "ctx-lh-isthmuscingulate", "ctx-lh-insula", "ctx-lh-medialorbitofrontal", "ctx-lh-lateralorbitofrontal",
            "ctx-rh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-insula", "ctx-rh-medialorbitofrontal", "ctx-rh-lateralorbitofrontal"]
TAU_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]

AB_initMap, TAU_initMap = [[1 for roi in conn.region_labels]]*2
ABt_initMap = [0.1 / len(AB_seeds) if roi in AB_seeds else 0 for roi in conn.region_labels]
TAUt_initMap = [0.01 / len(TAU_seeds) if roi in TAU_seeds else 0 for roi in conn.region_labels]

AB_initdam, TAU_initdam = [[0 for roi in conn.region_labels]]*2
POW_initdam = [1 for roi in conn.region_labels]

## PARAMETERS
He, Hi, taue, taui = 3.5, 22, 10, 16
prop_dt, bnm_dt = 0.25, 4
subj, model, g, s, bnm_simlength = subj, "jr", 4, 15.5, 20  # BNM Simulation

circmodel = CircularADpgModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam, TAU_initdam, POW_initdam,
    TAU_dam2SC=2e-5, POW_damrate=0.01, maxPOWdam=1.5,
    init_He=He, init_Hi=Hi, init_taue=taue, init_taui=taui, rho=5,
    prodAB=2, clearAB=2, transAB2t=2, clearABt=1.6,
    prodTAU=2.2, clearTAU=2, transTAU2t=2, clearTAUt=1.7)

circmodel.init_He["range"] = [He-1, He+2]  # origins (2.6, 9.75) :: (-0.65+x, x+6.5)
circmodel.init_Hi["range"] = [Hi-3, Hi+18]  # origins (17.6, 40) :: (-4.4+x, x+18)
circmodel.init_taue["range"] = [taue-4, taue+10]  # origins (6, 12) :: (-4+x, x+10)
circmodel.init_taui["range"] = [taui-8, taui+20]  # origins (12, 40) :: (-8+x, x+20)

out_circ = circmodel.\
    run(time=40, dt=prop_dt, sim=[subj, model, g, s, bnm_simlength], sim_dt=bnm_dt)
                # sim=[False] (xor) [subj, model, g, s, time(s)](simParams)

timepoints = out_circ[0]

## ALL PLOTTING MACHINERY -
title = "circNew_initTt0.01"
spec_fold = "figures/" + title + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(spec_fold)

# P1. Correlations
corrs = correlations_v2(out_circ, scatter="ABS&REL_multstages", band="3-alpha", title=title, folder=spec_fold)
corrs_PETrel = np.array(corrs[4])  # ["CN-SMC", "SMC-EMCI", "EMCI-LMCI", "LMCI-AD"] // ["CN", "SMC", "EMCI", "LMCI", "AD"]

# P2. Original Propagation map
animate_propagation_v4(out_circ, corrs_PETrel, ["CN-SMC", "SMC-EMCI", "EMCI-LMCI", "LMCI-AD"],  "PETrel_toxic",
                       conn, timeref=True, title=title, folder=spec_fold)

# P3. Braid diagram
braidPlot(out_circ, conn, "diagram", title=title, folder=spec_fold)

# P4. dynamical signals and spectra
skip = 4  # Don't plot all signals as it can get very slow
shortout = [np.array([out[7][::skip, :] for i, out in enumerate(out_circ[2]) if len(out) > 1]),
                np.array([out_circ[0][i] for i, out in enumerate(out_circ[2]) if len(out) > 1])]
timeseries_spectra(shortout, bnm_simlength*1000, transient=1000, regionLabels=conn.region_labels[::skip],
                   mode="animated", folder=spec_fold, freqRange=[2, 40], opacity=1, title=title, auto_open=True)

# P5. dynamical FC - we wanna see: 1) posterior increase;  2) posterior decrease and other increse; 3) others decrease
shortout = [np.array([out_circ[0][i] for i, out in enumerate(out_circ[2]) if len(out) > 1]),
            [out for i, out in enumerate(out_circ[2]) if len(out) > 1]]

animateFC(shortout, conn, mode="3Dcortex", threshold=0.05, title="cortex0.05", folder=spec_fold)
animateFC(shortout, conn, mode="3Ddmn", threshold=0.005, title="dmn0.005", folder=spec_fold)

# P6. Transfers reports
full_outs = [out[1:] for out in out_circ[2] if len(out) > 1]
TransferOne(out_circ, out_networkSim=shortout, skip=int(bnm_dt/prop_dt), mode=title, folder=spec_fold)
TransferTwo(out_circ, skip=int(bnm_dt/prop_dt), mode=title, folder=spec_fold)
propagationtrajectory_on4D(out_circ, title, PSE3d_tag="PSEmpi_ADpg_PSE3d-m11d10y2022-t17h.09m.44s", folder=spec_fold)






#
# ## RHO fit
# rho_vals = np.logspace(-3, 3, 30)
# pse, tic = [], time.time()
# for ii, rho in enumerate(rho_vals):
#     print("Simulating for rho%0.4f  -  %i/%i\n" % (rho, ii + 1, len(rho_vals)))
#
#     circmodel = CircularADpgModel(
#         conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam, TAU_initdam, POW_initdam,
#         TAU_dam2SC=2e-5, POW_damrate=0.01, maxPOWdam=1.5,
#         init_He=He, init_Hi=Hi, init_taue=taue, init_taui=taui, rho=rho,
#         prodAB=2, clearAB=2, transAB2t=2, clearABt=1.6,
#         prodTAU=2.2, clearTAU=2, transTAU2t=2, clearTAUt=1.7)
#
#     circmodel.init_He["range"] = [He - 1, He + 2]  # origins (2.6, 9.75) :: (-0.65+x, x+6.5)
#     circmodel.init_Hi["range"] = [Hi - 3, Hi + 18]  # origins (17.6, 40) :: (-4.4+x, x+18)
#     circmodel.init_taue["range"] = [taue - 4, taue + 10]  # origins (6, 12) :: (-4+x, x+10)
#     circmodel.init_taui["range"] = [taui - 8, taui + 20]  # origins (12, 40) :: (-8+x, x+20)
#
#     out_circ = circmodel. \
#         run(time=40, dt=prop_dt, sim=[subj, "jr", 4, 15.5, 5], sim_dt=bnm_dt)
#     # sim=[False] (xor) [subj, model, g, s, time(s)](simParams)
#     pse.append(np.asarray(out_circ[1])[:, 3, :])
#     print("     _time %0.2fs\n\n" % (time.time() - tic))
#
# corresp = braidPlot(np.asarray(pse), conn, "surface", rho_vals, title=mode)

# TODO - check structural matrix damage; tackle rebound (limiting TAUhp?); tackle stability of POWdam and thus AB42;

#
#
# #   SETUP and SIMULATE  #############
# # Following Alexandersen (2022)  - same parameters, ADNI initial conditions
# """
# Initial conditions from ADNI data, assuming the same initial relative concentration of
# toxic proteins used in Alexandersen 2022.
# """
#
# # Healthy AB/TAU initial protein distribution
# AB_initMap = [1 for roi in conn.region_labels]
# TAU_initMap = [1 for roi in conn.region_labels]
#
# # Toxic distribution
# ABt_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
# ABt_initMap = ABt_initMap[PET_idx]  # Sort it
# # Distribute the 0.1M (from Alexandersen) between nodes
# ABt_initMap = [ab / np.sum(ABt_initMap) * 0.1 for ab in ABt_initMap]
#
#
# # Healthy TAU initial protein distribution
# TAUt_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
# TAUt_initMap = TAUt_initMap[PET_idx]
# # Distribute the 0.1M (from Alexandersen) between nodes
# TAUt_initMap = [tau / np.sum(TAU_initMap) * 0.1 for tau in TAUt_initMap]
#
# output = ProteinSpreadModel(
#     conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
#     init_He=3.25, init_Hi=22, init_taue=10, init_taui=20, rho=0.001, toxicSynergy=12,
#     prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
#     prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66).\
#     run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]
#                                                                                         ## [subj, "jr", 3, 4.5, 10]
#
# title = "ALEXparamsADNIinitnormal"
# # corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PET", plot="s&c", title=title)
# # animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PET", conn, timeref=True, title=title)
#
# corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic", plot="s", title=title)
# animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETtoxic", conn, timeref=True, title=title)
#
#
#
# #   SETUP and SIMULATE  #############
# # Following Travis Thompson (2022)  - same parameters, ADNI initial conditions
# """
# Initial conditions from ADNI data, assuming the same initial relative concentration of
# toxic proteins used in Alexandersen 2022.
# """
# # TODO
# # Healthy AB/TAU initial protein distribution
# AB_initMap = [1 for roi in conn.region_labels]
# TAU_initMap = [1 for roi in conn.region_labels]
#
# # Toxic distribution
# ABt_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
# ABt_initMap = ABt_initMap[PET_idx]  # Sort it
# # Distribute the 0.1M (from Alexandersen) between nodes
# ABt_initMap = [ab / np.sum(ABt_initMap) * 0.1 for ab in ABt_initMap]
#
#
# # Healthy TAU initial protein distribution
# TAUt_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
# TAUt_initMap = TAUt_initMap[PET_idx]
# # Distribute the 0.1M (from Alexandersen) between nodes
# TAUt_initMap = [tau / np.sum(TAU_initMap) * 0.1 for tau in TAUt_initMap]
#
# output = ProteinSpreadModel(
#     conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
#     init_He=3.25, init_Hi=22, init_taue=10, init_taui=20, rho=0, toxicSynergy=12,
#     prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
#     prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66).\
#     run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]
#                                                                                         ## [subj, "jr", 3, 4.5, 10]
#
# title = "ALEXparamsADNIinitnormal"
# # corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PET", plot="s&c", title=title)
# # animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PET", conn, timeref=True, title=title)
#
# corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic", plot="s&c", title=title)
# animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETtoxic", conn, timeref=True, title=title)
#
#
#
# # ## PLV dynamics
# # output = ProteinSpreadModel(
# #     AB_initMap, TAU_initMap, AB_initMap, TAU_initMap, rho=0.001, toxicSynergy=1,
# #     prodAB=0.75, clearAB=1, transAB2t=1, clearABt=1,
# #     prodTAU=0.5, clearTAU=1, transTAU2t=1, clearTAUt=1).\
# #     run(conn, time=1, dt=0.1, sim=[subj, "jr", 8, 15, 6])  # sim [False; simParams[subj, model, g, s, time(s)]]
# #
# # corrs_PLV = correlations(output, ["HC-fam", "HC", "QSM", "FAM", "MCI", "MCI-conv"], reftype="PLV", band="3-alpha")
# # animate_propagation_v3(output, corrs_PLV, ["HC-fam", "HC", "QSM", "FAM", "MCI", "MCI-conv"],  "PLV", conn, timeref=True)
#
#
