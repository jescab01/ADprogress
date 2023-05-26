
import time
import numpy as np
import pandas as pd
import os
import pickle

from tvb.simulator.lab import connectivity
from ADpg_functions import circApproach, CircularADpgModel_vCC, paramtraj_in3D, correlations_v2, braidPlot

# from ADpg_functions import simulate_v2, correlations_v2, animateFC, surrogatesFC, animate_propagation_v4, CircularADpgModel_vCC, braidPlot
# from ADpg_CalibrationFunctions import TransferOne, TransferTwo, propagationtrajectory_on4D

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
subj, g, s, sigma = "HC-fam", 50, 20, 0.022
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
conn.weights = conn.scaled_weights(mode="tract")

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
conn.cortical = np.array([True if roi in cortical_rois else False for roi in conn.region_labels])

## TO USE JUST DMN
useDMN = True
dmn_rois = [  # ROIs not in Gianlucas set, from cingulum bundle description
    'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
    'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal',
    'ctx-lh-insula', 'ctx-rh-insula',
    'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',  # A6. in Gianlucas DMN
    'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',  # A6. in Gianlucas DMN
    'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate',
    'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',  # A7. in Gianlucas DMN
    'ctx-lh-middletemporal', 'ctx-rh-middletemporal',  # A5. in Gianlucas DMN
    'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal',  # A4. in Gianlucas DMN
    'ctx-lh-superiorparietal', 'ctx-rh-superiorparietal',
    'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',  # A3. in Gianlucas DMN
    'ctx-lh-precuneus', 'ctx-rh-precuneus',  # A1. in Gianlucas DMN
    'Left-Hippocampus', 'Right-Hippocampus',  # subcorticals
    'Left-Thalamus', 'Right-Thalamus',
    'Left-Amygdala', 'Right-Amygdala',
    'ctx-lh-entorhinal', 'ctx-rh-entorhinal'  # seed for TAUt
]
if useDMN:
    # load SC labels.
    SClabs = list(conn.region_labels)
    SC_dmn_idx = [SClabs.index(roi) for roi in dmn_rois]

    # #  Load FC labels, transform to SC format; check if match SC.
    # FClabs = list(np.loadtxt(data_folder + "FCavg_matrices/" + subj + "_roi_labels.txt", dtype=str))
    # FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    # FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois_dk]  # find indexes in FClabs that matches cortical_rois

    conn.weights = conn.weights[:, SC_dmn_idx][SC_dmn_idx]
    conn.tract_lengths = conn.tract_lengths[:, SC_dmn_idx][SC_dmn_idx]
    conn.region_labels = conn.region_labels[SC_dmn_idx]
    conn.cortical = conn.cortical[SC_dmn_idx]
    conn.centres = conn.centres[SC_dmn_idx]

#    ADNI PET DATA       ##########
ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)

# Check label order
PETlabs = list(ADNI_AVG.columns[12:])
PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]

#     2.  DEFINE INITIAL CONDITIONS       ######################################

# TODO - check structural matrix damage; tackle rebound (limiting TAUhp?);
#  tackle stability of POWdam and thus AB42;

"""
                A)   Alex Init  
Circular model  -  Alexandersen (2022) initial conditions

Seeding areas rise much faster than the rest of the nodes.
"""

AB_initMap, TAU_initMap = [[1 for roi in conn.region_labels]] * 2

##  REGIONAL SEEDs for toxic proteins

AB_seeds = ["ctx-lh-precuneus", "ctx-lh-isthmuscingulate", "ctx-lh-insula", "ctx-lh-medialorbitofrontal",
            "ctx-lh-lateralorbitofrontal",
            "ctx-rh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-insula", "ctx-rh-medialorbitofrontal",
            "ctx-rh-lateralorbitofrontal"]
TAU_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]

ABt_initMap = [0.05 / len(AB_seeds) if roi in AB_seeds else 0 for roi in conn.region_labels]
TAUt_initMap = [0.005 / len(TAU_seeds) if roi in TAU_seeds else 0 for roi in conn.region_labels]

AB_initdam, TAU_initdam = [[0 for roi in conn.region_labels]] * 2
HA_initdam = [0 for roi in conn.region_labels]

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


#    3. PARAMETERS   and   SIMULATE      ########################################
title, tic = "vCC_cModel_AlexInit_dmn" + str(useDMN), time.time()
prop_simL, prop_dt = 40, 0.25
bnm_simL, transient, bnm_dt = 10, 2, 1  # Units (seconds, seconds, years)

circmodel = CircularADpgModel_vCC(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam, TAU_initdam, HA_initdam,
    TAU_dam2SC=5e-2, HA_damrate=5, maxTAU2SCdam=0.3, # maxHAdam=1.25,  # origins @ ¿?, 0.01, 1.5
    init_He=3.25, init_Cee=108, init_Cie=33.75,  # origins 3.25, 108, 33.75 || Initial values for NMM variable parameters
    rho=50, toxicSynergy=0.4,  # origins 5, 2 || rho as a diffusion constant
    prodAB=3, clearAB=3, transAB2t=3, clearABt=2.4,
    prodTAU=3, clearTAU=3, transTAU2t=3, clearTAUt=2.55,
    cABexc=0.8, cABinh=0.4, cTAUexc=1.8, cTAUinh=1.8)

# 3.2 Define parameter ranges of change
rHe = [0.35, 0.35]
circmodel.init_He["range"] = [circmodel.init_He["value"][0] - rHe[0], circmodel.init_He["value"][0] + rHe[1]]

rCee = [75, 40]  # origins (54, 160) :: (-54+x, x+54)
circmodel.init_Cee["range"] = [circmodel.init_Cee["value"][0] - rCee[0], circmodel.init_Cee["value"][0] + rCee[1]]

rCie = [20.5, 10]  # origins (15, 50) :: (-16.75+x, x+16.25)
circmodel.init_Cie["range"] = [circmodel.init_Cie["value"][0] - rCie[0], circmodel.init_Cie["value"][0] + rCie[1]]


# 3.3 Run
out_circ = circmodel.run(time=prop_simL, dt=prop_dt, sim=[subj, g, s, sigma, bnm_simL, transient], sim_dt=bnm_dt)
                                            # sim=[False] (xor) [subj, model, g, s, time(s), transient(s)](simParams)


##    4.   BUILD SPEC FOLDER and SAVE PARAMETERS    ############
spec_fold = "results/" + title + "_bnm-dt" + str(bnm_dt) + \
            "sL" + str(bnm_simL) + "_" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(spec_fold)

with open(spec_fold + "/.PARAMETERS.txt", "w") as f:
    f.write('Title - %s; simtime: %0.2f(min)\nsubj = %s; useDMN = %s; g = %0.2f; s = %0.2f; sigma = %0.2f\n'
            'prop_simL = %0.2f(y); prop_dt = %0.2f(y)\nbnm_simL = %0.2f(s); transient = %0.2f(s); bnm_dt = %0.2f(y)\n\n'
            % (title, (time.time()-tic)/60, subj, str(useDMN), g, s, sigma, prop_simL, prop_dt, bnm_simL, transient, bnm_dt))
    # add title subj etc
    for key, val in vars(circmodel).items():
        f.write('%s:%s\n' % (key, val))

##   4b.  if needed SAVE DATA    #############
save = True
if save:
    with open(spec_fold + "/.DATA.pkl", "wb") as file:
        pickle.dump([out_circ, conn], file)
        file.close()


##    5.  PLOTTING MACHINERY        #########################################
# 5.1 Default new plot
circApproach(out_circ, conn, title, folder=spec_fold)
braidPlot(out_circ, conn, mode="diagram")

# param_info = "He"+str(circmodel.init_He["value"][0])+str(rHe) + "; Cie" + str(circmodel.init_Cie["value"][0]) + "; Cee" + str(circmodel.init_Cie["value"][0])+ str(rCee) +"; maxTAU2SC" + str(circmodel.maxTAU2SCdam["value"])
# paramtraj_in3D(out_circ, "vH_freq", "PSEmpi_3dFreqCharts4.0-m03d08y2023-t12h.46m.17s", folder=spec_fold, auto_open=True, param_info=param_info)
# paramtraj_in3D(out_circ, "vH_rate", "PSEmpi_3dFreqCharts4.0-m03d08y2023-t12h.46m.17s", folder=spec_fold, auto_open=True, param_info=param_info)


# # ad-hoc PLAYGROUD
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
#
#
# fig = make_subplots(rows=3, cols=1, subplot_titles=["I/E", "He/Cie", "Cee/Cie"])
# time = out_circ[0]
# svars = np.average(np.array(out_circ[1]), axis=2)
# he, cee, cie = svars[:, -4], svars[:, -3], svars[:, -2]
# he = (he - min(he)) / (max(he) - min(he))
# cee = (cee - min(cee)) / (max(cee) - min(cee))
# cie = (cie - min(cie)) / (max(cie) - min(cie))
#
# fig.add_trace(go.Scatter(x=time, y=he + cee - cie), row=1, col=1)
# fig.add_trace(go.Scatter(x=time, y=he - cie), row=2, col=1)
# fig.add_trace(go.Scatter(x=time, y=cee - cie), row=3, col=1)
#
# fig.show("browser")
#
# ####
# shortout = [np.array([out_circ[0][i] for i, out in enumerate(out_circ[2]) if len(out) > 1]),
#             [out for i, out in enumerate(out_circ[2]) if len(out) > 1]]
#
# rate_t0 = shortout[1][0][3]
#
# # plot corr between seeds and baseline rate
# import plotly.express as px
# fig = px.scatter(x=rate_t0, y=ABt_initMap, hover_name=conn.region_labels)
# fig.show("browser")
# plot corr between baseline rate and ADNI "seeding"


# 5.2 Dynamical firing rate, signals and spectra
# short_t = [t for i, t in enumerate(out_circ[0]) if len(out_circ[2][i]) > 1]
# shortout = [simpack[4][::4] for i, simpack in enumerate(out_circ[2]) if len(out_circ[2][i]) > 1]

# timeseries_spectra(shortout, bnm_simL*1000, transient*1000, conn.region_labels,
#                    mode="anim", timescale=short_t, param="Time (years)", folder=spec_fold,
#                    freqRange=[2, 40], opacity=1, title=title, auto_open=True)

# P1. Correlations
corrs = correlations_v2(out_circ, conn, scatter="ABS_simple", band="3-alpha", title=title, folder=spec_fold, auto_open=True)
# corrs_PETrel = np.array(corrs[4])  # ["CN-SMC", "SMC-EMCI", "EMCI-LMCI", "LMCI-AD"] // ["CN", "SMC", "EMCI", "LMCI", "AD"]


# # P2. Original Propagation map
# # animate_propagation_v4(out_circ, corrs_PETrel, ["CN-SMC", "SMC-EMCI", "EMCI-LMCI", "LMCI-AD"],  "PETrel_toxic",
# #                        conn, timeref=True, title=title, folder=spec_fold, auto_open=True)
#
# # P3. Braid diagram
# # braidPlot(out_circ, conn, "diagram", title=params["title"], folder=spec_fold, auto_open=False)
#
# # P4. dynamical signals and spectra
# # skip = 4  # Don't plot all signals as it can get very slow
# # shortout = [np.array([out[7][::skip, :] for i, out in enumerate(out_circ[2]) if len(out) > 1]),
# #                 np.array([out_circ[0][i] for i, out in enumerate(out_circ[2]) if len(out) > 1])]
# # timeseries_spectra(shortout, bnm_simlength*1000, transient=1000, regionLabels=conn.region_labels[::skip],
# #                    mode="animated", folder=spec_fold, freqRange=[2, 40], opacity=1, title=title, auto_open=False)
#
# # Pre5. generate FC surrogate
# # surrogates = surrogatesFC(500, subj, conn, "jr", params["g"], params["s"], params["bnm_simL"],
# #                           params_vCC=[params["i_He"], params["i_Cee"], params["i_Cie"]])
#
# # P5. dynamical FC - we wanna see: 1) posterior increase;  2) posterior decrease and other increse; 3) others decrease
# shortout = [np.array([out_circ[0][i] for i, out in enumerate(out_circ[2]) if len(out) > 1]),
#             [out for i, out in enumerate(out_circ[2]) if len(out) > 1]]
#
# # animateFC(shortout, conn, mode="3Dcortex", threshold=0.05, title="cortex0.05", folder=spec_fold, surrogates=surrogates, auto_open=False)
# # animateFC(shortout, conn, mode="3Ddmn", threshold=0.005, title="dmn0.005", folder=spec_fold, surrogates=surrogates, auto_open=False)
#
# # P6. Transfers reports
# full_outs = [out[1:] for out in out_circ[2] if len(out) > 1]
# TransferOne(out_circ, out_networkSim=shortout, skip=int(params["bnm_dt"] / params["prop_dt"]), mode=params["title"],
#             folder=spec_fold, auto_open=True)
# TransferTwo(out_circ, skip=int(params["bnm_dt"]/params["prop_dt"]), mode=params["title"], folder=spec_fold, auto_open=True)
# # propagationtrajectory_on4D(out_circ, title, PSE3d_tag="PSEmpi_ADpg_PSE3d-m12d12y2022-t22h.00m.48s", folder=spec_fold, auto_open=False)
#
#
# # ##    5.  RHO fit          ###################################
# #
# # # rho_vals = np.logspace(-3, 3, 20)
# # # braid_pse, tic = [], time.time()
# # # for ii, rho in enumerate(rho_vals):
# # #     print("Simulating for rho%0.4f  -  %i/%i\n" % (rho, ii + 1, len(rho_vals)))
# # #
# # #     out_braid = circmodel. \
# # #         run(time=40, dt=prop_dt, sim=[subj, model, g, s, bnm_simlength], sim_dt=bnm_dt)
# # #     # sim=[False] (xor) [subj, model, g, s, time(s)](simParams)
# # #
# # #     braid_pse.append(np.asarray(out_braid[1])[:, 3, :])
# # #     print("     _time %0.2fs\n\n" % (time.time() - tic))
# # #
# # # corresp = braidPlot(np.asarray(braid_pse), conn, "surface", rho_vals, title=title, folder=spec_fold)
# # #
# # #
# # # # Save RHO PSE results and Parameters
# # # with open(spec_fold + "/dataset_" + title + ".pkl", "wb") as f:
# # #     pickle.dump([out_circ, braid_pse, corresp], f)
# #


