
import time
import numpy as np
import pandas as pd
import os

from tvb.simulator.lab import connectivity
from ADpg_functions import ProteinSpreadModel, animate_propagation_v2

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"




subj = "sub-01"

#   STRUCTURAL CONNECTIVITY   #########
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
conn.weights = conn.scaled_weights(mode="tract")  # did they normalize? maybe this affects to the spreading?
conn.speed = np.array([15])

SClabs = list(conn.region_labels)

#      ADNI PET DATA       ##########
ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)
AB_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
TAU_initMap = np.squeeze(
    np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))

# Check label order
PETlabs = list(ADNI_AVG.columns[12:])
PET_idx = [PETlabs.index(roi.lower()) for roi in SClabs]


#####  MATHEMATICAL CHECK - on system's stationary points
## Following Thompson (2020) supplementary

# 1. AB-TAU healthy a0=0.75, b0=0.5
AB_initMap = [0.75 for roi in conn.region_labels]
TAU_initMap = [0.5 for roi in conn.region_labels]
ABt_initMap = [0 for roi in conn.region_labels]
TAUt_initMap = [0 for roi in conn.region_labels]

output = ProteinSpreadModel(
    AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, rho=0.001, toxicSynergy=1,
    prodAB=0.75, clearAB=1, transAB2t=1, clearABt=1,
    prodTAU=0.5, clearTAU=1, transTAU2t=1, clearTAUt=1).run(conn, time=10, dt=0.25)
animate_propagation_v2(output, dyn_mark=True)

# 2. ABt-TAU a0=0.75, b0=0.5, a1t=0.6
AB_initMap = [0.6 for roi in conn.region_labels]
TAU_initMap = [0.5 for roi in conn.region_labels]
ABt_initMap = [0.25 for roi in conn.region_labels]
TAUt_initMap = [0 for roi in conn.region_labels]

output = ProteinSpreadModel(
    AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, rho=0.001, toxicSynergy=1,
    prodAB=0.75, clearAB=1, transAB2t=1, clearABt=0.6,
    prodTAU=0.5, clearTAU=1, transTAU2t=1, clearTAUt=1).run(conn, time=10, dt=0.25)
animate_propagation_v2(output, dyn_mark=True)

# 3.
AB_initMap = [0.75 for roi in conn.region_labels]
TAU_initMap = [0.4 for roi in conn.region_labels]
ABt_initMap = [0 for roi in conn.region_labels]
TAUt_initMap = [0.25 for roi in conn.region_labels]

output = ProteinSpreadModel(
    AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, rho=0.001, toxicSynergy=1,
    prodAB=0.75, clearAB=1, transAB2t=1, clearABt=1,
    prodTAU=0.5, clearTAU=1, transTAU2t=1, clearTAUt=0.4).run(conn, time=10, dt=0.25)
animate_propagation_v2(output, dyn_mark=True)


# 4.
AB_initMap = [0.6 for roi in conn.region_labels]
TAU_initMap = [0.32 for roi in conn.region_labels]
ABt_initMap = [0.25 for roi in conn.region_labels]
TAUt_initMap = [0.45 for roi in conn.region_labels]

output = ProteinSpreadModel(
    AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, rho=0.001, toxicSynergy=1,
    prodAB=0.75, clearAB=1, transAB2t=1, clearABt=0.6,
    prodTAU=0.5, clearTAU=1, transTAU2t=1, clearTAUt=0.4).run(conn, time=10, dt=0.25)
animate_propagation_v2(output, dyn_mark=True)



