
import time
import numpy as np
import pandas as pd
import os
import pickle

from tvb.simulator.lab import connectivity
from ADpg_functions import ProteinSpreadModel, correlations, \
    animate_propagation_v4, g_explore, simulate_v2

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

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

#  REGIONAL SEEDs for toxic proteins
ABt_seeds = ["ctx-lh-precuneus", "ctx-lh-isthmuscingulate", "ctx-lh-insula", "ctx-lh-medialorbitofrontal", "ctx-lh-lateralorbitofrontal",
            "ctx-rh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-insula", "ctx-rh-medialorbitofrontal", "ctx-rh-lateralorbitofrontal"]
TAUt_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]


time, dt = 40, 0.25

# np.arange(0, time, dt)  # Possible times to choose XOR sim_dt
tsel = [0, 16, 20, 24]


#   SETUP and SIMULATE   -   PET dynamics by now  #############
# Following Alexandersen (2022)  - same parameters, same initial conditions
"""
Initial conditions in Alexandersen
AB healthy - 1M per node; 
AB toxic - 0.1M/12nodes=0.0085 (0.85% of nodes AB healthy concentration) in seeds:
 precuneus, isthmus cingulate, insula, medial orbitofrontal, lateral orbitofrontal.
 
Tau healthy - 1M per node; 
Tau toxic - 0.1M/2nodes=0.05 (5% of nodes Tau healthy concentration) in seeds:
 entorhinal cortex.
"""

AB_initMap = [1 for roi in conn.region_labels]
TAU_initMap = [1 for roi in conn.region_labels]
ABt_initMap = [0.1/len(ABt_seeds) if roi in ABt_seeds else 0 for roi in conn.region_labels]
TAUt_initMap = [0.1/len(TAUt_seeds) if roi in TAUt_seeds else 0 for roi in conn.region_labels]

output = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
    init_He=3.25, init_Hi=22, init_taue=10, init_taui=20, rho=0.001, toxicSynergy=12,
    prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
    prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66).\
    run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]
                                                                                        ## [subj, "jr", 3, 4.5, 10]

title = "ALEXparams&init"
# corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETsum", plot="s&c", title=title)
# animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETsum", conn, timeref=True, title=title)
corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic", plot="s&c", title=title)
animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETtoxic", conn, timeref=True, title=title)



#   SETUP and SIMULATE  #############
# Following Alexandersen (2022)  - same parameters, ADNI initial conditions
"""
Initial conditions from ADNI data, assuming the same initial relative concentration of 
toxic proteins used in Alexandersen 2022. 
"""

# Healthy AB/TAU initial protein distribution
AB_initMap = [1 for roi in conn.region_labels]
TAU_initMap = [1 for roi in conn.region_labels]

# Toxic distribution
ABt_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
ABt_initMap = ABt_initMap[PET_idx]  # Sort it
# Distribute the 0.1M (from Alexandersen) between nodes
ABt_initMap = [ab / np.sum(ABt_initMap) * 0.1 for ab in ABt_initMap]


# Healthy TAU initial protein distribution
TAUt_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
TAUt_initMap = TAUt_initMap[PET_idx]
# Distribute the 0.1M (from Alexandersen) between nodes
TAUt_initMap = [tau / np.sum(TAU_initMap) * 0.1 for tau in TAUt_initMap]

output = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
    init_He=3.25, init_Hi=22, init_taue=10, init_taui=20, rho=0.001, toxicSynergy=12,
    prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
    prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66).\
    run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]
                                                                                        ## [subj, "jr", 3, 4.5, 10]

title = "ALEXparamsADNIinitnormal"
# corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PET", plot="s&c", title=title)
# animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PET", conn, timeref=True, title=title)

corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic", plot="s", title=title)
animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETtoxic", conn, timeref=True, title=title)



#   SETUP and SIMULATE  #############
# Following Travis Thompson (2022)  - same parameters, ADNI initial conditions
"""
Initial conditions from ADNI data, assuming the same initial relative concentration of 
toxic proteins used in Alexandersen 2022. 
"""
# TODO
# Healthy AB/TAU initial protein distribution
AB_initMap = [1 for roi in conn.region_labels]
TAU_initMap = [1 for roi in conn.region_labels]

# Toxic distribution
ABt_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
ABt_initMap = ABt_initMap[PET_idx]  # Sort it
# Distribute the 0.1M (from Alexandersen) between nodes
ABt_initMap = [ab / np.sum(ABt_initMap) * 0.1 for ab in ABt_initMap]


# Healthy TAU initial protein distribution
TAUt_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
TAUt_initMap = TAUt_initMap[PET_idx]
# Distribute the 0.1M (from Alexandersen) between nodes
TAUt_initMap = [tau / np.sum(TAU_initMap) * 0.1 for tau in TAUt_initMap]

output = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
    init_He=3.25, init_Hi=22, init_taue=10, init_taui=20, rho=0, toxicSynergy=12,
    prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
    prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66).\
    run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]
                                                                                        ## [subj, "jr", 3, 4.5, 10]

title = "ALEXparamsADNIinitnormal"
# corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PET", plot="s&c", title=title)
# animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PET", conn, timeref=True, title=title)

corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic", plot="s&c", title=title)
animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETtoxic", conn, timeref=True, title=title)



# ## PLV dynamics
# output = ProteinSpreadModel(
#     AB_initMap, TAU_initMap, AB_initMap, TAU_initMap, rho=0.001, toxicSynergy=1,
#     prodAB=0.75, clearAB=1, transAB2t=1, clearABt=1,
#     prodTAU=0.5, clearTAU=1, transTAU2t=1, clearTAUt=1).\
#     run(conn, time=1, dt=0.1, sim=[subj, "jr", 8, 15, 6])  # sim [False; simParams[subj, model, g, s, time(s)]]
#
# corrs_PLV = correlations(output, ["HC-fam", "HC", "QSM", "FAM", "MCI", "MCI-conv"], reftype="PLV", band="3-alpha")
# animate_propagation_v3(output, corrs_PLV, ["HC-fam", "HC", "QSM", "FAM", "MCI", "MCI-conv"],  "PLV", conn, timeref=True)


