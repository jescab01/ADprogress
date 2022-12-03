
import time
import numpy as np
import pandas as pd
import os
import pickle

from tvb.simulator.lab import connectivity
from ADpg_functions import ProteinSpreadModel, correlations, animate_propagation_v3, \
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
AB_seeds = ["ctx-lh-precuneus", "ctx-lh-isthmuscingulate", "ctx-lh-insula", "ctx-lh-medialorbitofrontal", "ctx-lh-lateralorbitofrontal",
            "ctx-rh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-insula", "ctx-rh-medialorbitofrontal", "ctx-rh-lateralorbitofrontal"]
TAU_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]


# TODO 4-implement transfer function inside the model for abeta and tau: look alexandersen.


#   SETUP and SIMULATE   -   PET dynamics by now  #############
# Following Alexandersen (2022)  - same parameters, same initial conditions, wo/ damage functions
"""
Im not adding the damage functions nor implementing the transfers. Rest the same.

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
ABt_initMap = [0.1/len(AB_seeds) if roi in AB_seeds else 0 for roi in conn.region_labels]
TAUt_initMap = [0.1/len(TAU_seeds) if roi in TAU_seeds else 0 for roi in conn.region_labels]

time, dt = 40, 0.25

# np.arange(0, time, dt)  # Possible times to choose XOR sim_dt
tsel = [0, 16, 20, 24]

output = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
    init_He=3.25, init_Hi=22, init_taue=10, rho=0.001, toxicSynergy=12,
    prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
    prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66).\
    run(time=time, dt=dt, sim=[subj, "jr", 48, 4.5, 10], sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]
                                                                                        ## [subj, "jr", 3, 4.5, 10]

corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PET")
# animate_propagation_v3(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PET", conn, timeref=True)
animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PET", conn, timeref=True)



## Explore PSE for the simulated

sim_dt = 4

res = []
for t in range(0, len(output[0]), int(sim_dt/dt)):

    for g in range(0, 10, 1):
        _, _, fftp, _, _, plv_r, _, _, _, _ =\
            simulate_v2(subj, output[2][t][0], "jr", g, 4.5, sv=output[1][t][6:], t=10)

        res.append([t, g, 4.5, plv_r, fftp])


res_df = pd.DataFrame(np.asarray([out[:4] for out in res]), columns=["t", "g", "s", "rPLV"])
res_df.to_csv("res.csv")

import plotly.graph_objects as go
import plotly.express as px

fig = go.Figure(go.Scatter)



fig = px.line(res_df, x="g", y="rPLV", color="t")

fig.show("browser")










# Save the output
with open("output_ALEXparams&initcond-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl", "wb") as file:
    pickle.dump(output, file)
    file.close()

output_sim = [output[2][i] for i in range(len(output[2])) if (len(output[2][i]) > 2) and (i*dt in tsel)]

g_explore(output_sim, tsel, param="year")






#   SETUP and SIMULATE   -   PET dynamics by now  #############
# Following Alexandersen (2022)  - same parameters, ADNI initial conditions, wo/ damage functions
"""
Im not adding the damage functions nor implementing transfers.

Initial conditions from ADNI data, assuming the same relative concentration of 
toxic proteins used in Alexandersen 2022. 
"""

# AB_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
# AB_initMap = AB_initMap[PET_idx]
#
# ABt_initMap = [0.1/len(AB_seeds)*AB_initMap[i] if roi in AB_seeds else 0 for i, roi in enumerate(conn.region_labels)]
#
# TAU_initMap = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == "CN")].iloc[:, 12:]))
# TAU_initMap = TAU_initMap[PET_idx]
#
# TAUt_initMap = [0.1/len(TAU_seeds)*TAU_initMap[i] if roi in TAU_seeds else 0 for i, roi in enumerate(conn.region_labels)]
#
#
# output = ProteinSpreadModel(
#     AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, rho=0.001, toxicSynergy=12,
#     prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
#     prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66).\
#     run(conn, time=40, dt=0.25, sim=False)  # sim [False; simParams[subj, model, g, s, time(s)]]
#
# corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PET")
# animate_propagation_v3(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PET", conn, timeref=True)
#







# ## PLV dynamics
# output = ProteinSpreadModel(
#     AB_initMap, TAU_initMap, AB_initMap, TAU_initMap, rho=0.001, toxicSynergy=1,
#     prodAB=0.75, clearAB=1, transAB2t=1, clearABt=1,
#     prodTAU=0.5, clearTAU=1, transTAU2t=1, clearTAUt=1).\
#     run(conn, time=1, dt=0.1, sim=[subj, "jr", 8, 15, 6])  # sim [False; simParams[subj, model, g, s, time(s)]]
#
# corrs_PLV = correlations(output, ["HC-fam", "HC", "QSM", "FAM", "MCI", "MCI-conv"], reftype="PLV", band="3-alpha")
# animate_propagation_v3(output, corrs_PLV, ["HC-fam", "HC", "QSM", "FAM", "MCI", "MCI-conv"],  "PLV", conn, timeref=True)


