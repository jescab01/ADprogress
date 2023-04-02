

import time
import numpy as np
import pandas as pd
import os
import pickle

from tvb.simulator.lab import connectivity
from ADpg_functions import simulate_v2, correlations_v2, animateFC, surrogatesFC, animate_propagation_v4, CircularADpgModel, braidPlot
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


#     1.  PREPARE EMPIRICAL DATA      #########################
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


#     2.  MODEL SELECTION AREA       ######################################

# TODO - check structural matrix damage; tackle rebound (limiting TAUhp?);
#  tackle stability of POWdam and thus AB42;

"""
                A)   Alex Init  
Circular model  -  Alexandersen (2022) initial conditions

Seeding areas rise much faster than the rest of the nodes.
"""

title = "circModel_AlexInit"

AB_initMap, TAU_initMap = [[1 for roi in conn.region_labels]]*2

##  REGIONAL SEEDs for toxic proteins
AB_seeds = ["ctx-lh-precuneus", "ctx-lh-isthmuscingulate", "ctx-lh-insula", "ctx-lh-medialorbitofrontal", "ctx-lh-lateralorbitofrontal",
            "ctx-rh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-insula", "ctx-rh-medialorbitofrontal", "ctx-rh-lateralorbitofrontal"]
TAU_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]

ABt_initMap = [0.01 / len(AB_seeds) if roi in AB_seeds else 0 for roi in conn.region_labels]
TAUt_initMap = [0.01 / len(TAU_seeds) if roi in TAU_seeds else 0 for roi in conn.region_labels]

AB_initdam, TAU_initdam = [[0 for roi in conn.region_labels]]*2
POW_initdam = [1 for roi in conn.region_labels]


"""
                B)   ADNI Init  
# Circular model  -  ADNI based initial conditions: Relative Increase @EMCI from CN/SCD

Correlations for the distribution of seeds go from maximal to 0 as the model evolves; 
FC corr has just one peak r=0.45 at t20: why?
"""

# title = "circModel_initADNI-CN2EMCI-onlyAB"

# AB_initMap, TAU_initMap = [[1 for roi in conn.region_labels]]*2
#
# ##  REGIONAL SEEDs for toxic proteins
# # Toxic distribution
# ABt_initMap_t0 = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
# ABt_initMap_t1 = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == "EMCI")].iloc[:, 12:]))
# ABt_initMap_rel = (ABt_initMap_t1-ABt_initMap_t0)[PET_idx]  # Compute the relative and sort it
# # Distribute the 0.1M (from Alexandersen) between nodes
# ABt_initMap = [ab_inc / np.sum(ABt_initMap_rel) * 0.1 if ab_inc > 0 else 0 for ab_inc in ABt_initMap_rel]
#
# # # TAUt initial - from ADNI
# # TAUt_initMap_t0 = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
# # TAUt_initMap_t1 = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == "EMCI")].iloc[:, 12:]))
# # TAUt_initMap_rel = (TAUt_initMap_t1-TAUt_initMap_t0)[PET_idx]
# # # Distribute the 0.1M (from Alexandersen) between nodes
# # TAUt_initMap = [tau_inc / np.sum(TAUt_initMap_rel) * 0.1 if tau_inc > 0 else 0 for tau_inc in TAUt_initMap_rel]
#
# # TAUt initial - from Alex
# TAU_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]
# TAUt_initMap = [0.01 / len(TAU_seeds) if roi in TAU_seeds else 0 for roi in conn.region_labels]
#
# AB_initdam, TAU_initdam = [[0 for roi in conn.region_labels]]*2
# POW_initdam = [1 for roi in conn.region_labels]



#    3. PARAMETERS & SIMULATE         ########################################

He, Hi, taue, taui = 3.5, 22, 10, 16  # origins 3.25, 22, 10, 20
rHe, rHi, rtaue, rtaui = [1, 2], [3, 18], [4, 10], [8, 20]
rho, tSyn = 50, 0.4  # origins 5, 2
pow_damrate, maxpowdam = 0.005, 1.25  # origins 0.01, 1.5

prop_dt, bnm_dt = 0.25, 2
subj, model, g, s, bnm_simlength = subj, "jr", 4, 15.5, 20  # BNM Simulation

circmodel = CircularADpgModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam, TAU_initdam, POW_initdam,
    TAU_dam2SC=2e-5, POW_damrate=pow_damrate, maxPOWdam=maxpowdam,
    init_He=He, init_Hi=Hi, init_taue=taue, init_taui=taui,
    rho=rho, toxicSynergy=tSyn,
    prodAB=2, clearAB=2, transAB2t=2, clearABt=1.6,
    prodTAU=2.2, clearTAU=2, transTAU2t=2, clearTAUt=1.7)

circmodel.init_He["range"] = [He - rHe[0], He + rHe[1]]  # origins (2.6, 9.75) :: (-0.65+x, x+6.5)
circmodel.init_Hi["range"] = [Hi - rHi[0], Hi + rHi[1]]  # origins (17.6, 40) :: (-4.4+x, x+18)
circmodel.init_taue["range"] = [taue - rtaue[0], taue + rtaue[1]]  # origins (6, 12) :: (-4+x, x+10)
circmodel.init_taui["range"] = [taui - rtaui[0], taui + rtaui[1]]  # origins (12, 40) :: (-8+x, x+20)

out_circ = circmodel.\
    run(time=40, dt=prop_dt, sim=[subj, model, g, s, bnm_simlength], sim_dt=bnm_dt)
                # sim=[False] (xor) [subj, model, g, s, time(s)](simParams)

timepoints = out_circ[0]


##    4.  ALL PLOTTING MACHINERY        #########################################

spec_fold = "figures/" + title + "_bnm-dt" + str(bnm_dt) + "sL" + str(bnm_simlength) + "_" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(spec_fold)

# P1. Correlations
corrs = correlations_v2(out_circ, scatter="ABS_color", band="3-alpha", title=title, folder=spec_fold, auto_open=False)
corrs_PETrel = np.array(corrs[4])  # ["CN-SMC", "SMC-EMCI", "EMCI-LMCI", "LMCI-AD"] // ["CN", "SMC", "EMCI", "LMCI", "AD"]

# P2. Original Propagation map
animate_propagation_v4(out_circ, corrs_PETrel, ["CN-SMC", "SMC-EMCI", "EMCI-LMCI", "LMCI-AD"],  "PETrel_toxic",
                       conn, timeref=True, title=title, folder=spec_fold, auto_open=True)

# P3. Braid diagram
braidPlot(out_circ, conn, "diagram", title=title, folder=spec_fold, auto_open=False)

# P4. dynamical signals and spectra
skip = 4  # Don't plot all signals as it can get very slow
shortout = [np.array([out[7][::skip, :] for i, out in enumerate(out_circ[2]) if len(out) > 1]),
                np.array([out_circ[0][i] for i, out in enumerate(out_circ[2]) if len(out) > 1])]
timeseries_spectra(shortout, bnm_simlength*1000, transient=1000, regionLabels=conn.region_labels[::skip],
                   mode="animated", folder=spec_fold, freqRange=[2, 40], opacity=1, title=title, auto_open=False)

# Pre5. generate FC surrogate
surrogates = surrogatesFC(500, subj, conn, model, g, s, bnm_simlength, params=[He, Hi, taue, taui])

# P5. dynamical FC - we wanna see: 1) posterior increase;  2) posterior decrease and other increse; 3) others decrease
shortout = [np.array([out_circ[0][i] for i, out in enumerate(out_circ[2]) if len(out) > 1]),
            [out for i, out in enumerate(out_circ[2]) if len(out) > 1]]

animateFC(shortout, conn, mode="3Dcortex", threshold=0.05, title="cortex0.05", folder=spec_fold, surrogates=surrogates, auto_open=False)
animateFC(shortout, conn, mode="3Ddmn", threshold=0.005, title="dmn0.005", folder=spec_fold, surrogates=surrogates, auto_open=False)

# P6. Transfers reports
full_outs = [out[1:] for out in out_circ[2] if len(out) > 1]
TransferOne(out_circ, out_networkSim=shortout, skip=int(bnm_dt/prop_dt), mode=title, folder=spec_fold, auto_open=False)
TransferTwo(out_circ, skip=int(bnm_dt/prop_dt), mode=title, folder=spec_fold, auto_open=False)
propagationtrajectory_on4D(out_circ, title, PSE3d_tag="PSEmpi_ADpg_PSE3d-m12d12y2022-t22h.00m.48s", folder=spec_fold, auto_open=False)


all_params = {"title": title, "subject": subj, "g": g, "s": s,
              "He": He, "Hi": Hi, "taue": taue, "taui": taui,
              "rho": rho, "toxicSynergy": tSyn,
              "pow_damrate": pow_damrate, "maxpowdam": maxpowdam,
              "rHe": rHe, "rHi": rHi, "rtaue": rtaue, "rtaui": rtaui,
              "prop_dt": prop_dt, "bnm_dt": bnm_dt, "bnm_simL": bnm_simlength,
              "AB_init": AB_initMap, "ABt_init": ABt_initMap,
              "TAU_init": TAU_initMap, "TAUt_init": TAUt_initMap,
              }

with open(spec_fold + "/.PARAMETERS.txt", "w") as f:
    for key, val in all_params.items():
        f.write('%s:%s\n' % (key, val))



##    5.  RHO fit          ###################################

# rho_vals = np.logspace(-3, 3, 20)
# braid_pse, tic = [], time.time()
# for ii, rho in enumerate(rho_vals):
#     print("Simulating for rho%0.4f  -  %i/%i\n" % (rho, ii + 1, len(rho_vals)))
#
#     out_braid = circmodel. \
#         run(time=40, dt=prop_dt, sim=[subj, model, g, s, bnm_simlength], sim_dt=bnm_dt)
#     # sim=[False] (xor) [subj, model, g, s, time(s)](simParams)
#
#     braid_pse.append(np.asarray(out_braid[1])[:, 3, :])
#     print("     _time %0.2fs\n\n" % (time.time() - tic))
#
# corresp = braidPlot(np.asarray(braid_pse), conn, "surface", rho_vals, title=title, folder=spec_fold)
#
#
# # Save RHO PSE results and Parameters
# with open(spec_fold + "/dataset_" + title + ".pkl", "wb") as f:
#     pickle.dump([out_circ, braid_pse, corresp], f)

