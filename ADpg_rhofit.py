
import time
import numpy as np
import pandas as pd
import os
import pickle

from tvb.simulator.lab import connectivity
from ADpg_functions import ProteinSpreadModel, correlations, \
    animate_propagation_v4, g_explore, simulate_v2, braidPlot

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"



## Testing rho [e-10, e2]
rho_vals = np.logspace(-10, -3, 100)

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

#   Christoffer Alexandersen (2022)   -
#   same parameters, same initial conditions  #############
"""
Initial conditions in Alexandersen
AB healthy - 1M per node;
AB toxic - 0.1M/12nodes=0.0085 (0.85% of nodes AB healthy concentration) in seeds:
 precuneus, isthmus cingulate, insula, medial orbitofrontal, lateral orbitofrontal.

Tau healthy - 1M per node;
Tau toxic - 0.1M/2nodes=0.05 (5% of nodes Tau healthy concentration) in seeds:
 entorhinal cortex.
 
Braids not organized; corrs low; AB seeds leading.
"""

#  REGIONAL SEEDs for toxic proteins
ABt_seeds = ["ctx-lh-precuneus", "ctx-lh-isthmuscingulate", "ctx-lh-insula", "ctx-lh-medialorbitofrontal",
             "ctx-lh-lateralorbitofrontal",
             "ctx-rh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-insula", "ctx-rh-medialorbitofrontal",
             "ctx-rh-lateralorbitofrontal"]
TAUt_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]

AB_initMap = [1 for roi in conn.region_labels]
TAU_initMap = [1 for roi in conn.region_labels]
ABt_initMap = [0.1 / len(ABt_seeds) if roi in ABt_seeds else 0 for roi in conn.region_labels]
TAUt_initMap = [0.1 / len(TAUt_seeds) if roi in TAUt_seeds else 0 for roi in conn.region_labels]

time, dt = 40, 0.25

# pse = []
# for rho in rho_vals:
#     output = ProteinSpreadModel(
#         conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
#         init_He=3.25, init_Hi=22, init_taue=10, rho=rho, toxicSynergy=12,
#         prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
#         prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66). \
#         run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]
#     ## [subj, "jr", 3, 4.5, 10]
#     pse.append(np.asarray(output[1])[:, 3, :])
# pse = np.asarray(pse)

# Plot the result -
title = "ALEXparams&init"
# corresp = braidPlot(pse, conn, "surface", rho_vals, title=title)

# Define rho for further analysis
rho = 0.001
output = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
    init_He=3.25, init_Hi=22, init_taue=10, rho=rho, toxicSynergy=0.5,
    prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
    prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=1.8). \
    run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]

corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic", plot="s", title=title)
animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETtoxic", conn, timeref=True, title=title)



######   Travis Thompson (2020)  - same initial conditions and parameters        #####
"""
PRIMARY TAUOPATHY.
"All nodes healthy, but susceptible primary tauopathy" -
[AB, ABt, TAU, TAUt] = [a0/a1, 0, b0/b1, 0] = [0.75, 0, 0.5, 0];

"temporobasal and frontomedial AB seeding sites (53 nodes) where each seeded with 0.189% of ABt.
Thus brain ABt concentration represents a 1% of healthy concentration."
I can use 0.189% per node (i.e. 0.0014 (M) ABt per roi) XOR [1% brain wide (i.e. 0.105 (M) ABt per roi)]

"locus coeruleus and transentorhinal nodes were seeded with a perturbation of 1%"
Not clear, 1% respect to what? Assuming 1% repect to brain wide healthy concentration: [0.21 (M) TAUt per roi]
"""

AB_initMap = [0.75 for roi in conn.region_labels]
# Toxic seeding AB: temporobasal and frontomedial ABt seeding (1%): ¿57rois? Lausanne multi-resolution parcellation
# Maybe this is inferior temporal and medial frontal?
ABt_seeds = ['ctx-lh-inferiortemporal', 'ctx-lh-superiorfrontal', 'ctx-lh-medialorbitofrontal',
             'ctx-rh-inferiortemporal', 'ctx-rh-superiorfrontal', 'ctx-rh-medialorbitofrontal']
ABt_initMap = [sum(AB_initMap) * 0.01 / len(ABt_seeds) if roi in ABt_seeds else 0 for roi in conn.region_labels]

TAU_initMap = [0.5 for roi in conn.region_labels]
# Toxic seeding TAU: transenthorhinal and locus coeruleus TAUt seeding (1%); LC not in our atlas.
TAUt_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]
TAUt_initMap = [sum(AB_initMap) * 0.01 / len(TAUt_seeds) if roi in TAUt_seeds else 0 for roi in conn.region_labels]

time, dt = 300, 0.25

pse = []
for i, rho in enumerate(rho_vals):
    print(i)
    output = ProteinSpreadModel(
        conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
        init_He=3.25, init_Hi=22, init_taue=10, rho=rho, toxicSynergy=1,
        prodAB=0.75, clearAB=1, transAB2t=1, clearABt=1,
        prodTAU=0.5, clearTAU=1, transTAU2t=1, clearTAUt=1). \
        run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]

    pse.append(np.asarray(output[1])[:, 3, :])
pse = np.asarray(pse)

# Plot the result -
title = "ThompPRIMARY"
corresp = braidPlot(pse, conn, "surface", rho_vals, title=title)

# Define rho for further analysis
time, dt = 300, 0.1
rho = 3
output = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
    init_He=3.25, init_Hi=22, init_taue=10, rho=rho, toxicSynergy=1,
    prodAB=0.75, clearAB=1, transAB2t=1, clearABt=1,
    prodTAU=0.5, clearTAU=1, transTAU2t=1, clearTAUt=1). \
    run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]

corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic",  title=title)
animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETtoxic", conn, timeref=True, title=title)


"""
SECONDARY TAUOPATHY.
"All nodes healthy, but susceptible secondary tauopathy (b2=0.75, b3=3).
Seeding patterns idential to primary."  -
"""

AB_initMap = [0.75 for roi in conn.region_labels]
# Toxic seeding AB: temporobasal and frontomedial ABt seeding (1%): ¿57rois? Lausanne multi-resolution parcellation
# Maybe this is inferior temporal and medial frontal?
ABt_seeds = ['ctx-lh-inferiortemporal', 'ctx-lh-superiorfrontal', 'ctx-lh-medialorbitofrontal',
             'ctx-rh-inferiortemporal', 'ctx-rh-superiorfrontal', 'ctx-rh-medialorbitofrontal']
ABt_initMap = [sum(AB_initMap) * 0.01 / len(ABt_seeds) if roi in ABt_seeds else 0 for roi in conn.region_labels]

TAU_initMap = [0.5 for roi in conn.region_labels]
# Toxic seeding TAU: transenthorhinal and locus coeruleus TAUt seeding (1%); LC not in our atlas.
TAUt_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]
TAUt_initMap = [sum(AB_initMap) * 0.01 / len(TAUt_seeds) if roi in TAUt_seeds else 0 for roi in conn.region_labels]

pse = []
for rho in rho_vals:
    output = ProteinSpreadModel(
        conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
        init_He=3.25, init_Hi=22, init_taue=10, rho=rho, toxicSynergy=3,
        prodAB=0.75, clearAB=1, transAB2t=1, clearABt=1,
        prodTAU=0.5, clearTAU=1, transTAU2t=0.75, clearTAUt=1). \
        run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]
    ## [subj, "jr", 3, 4.5, 10]
    pse.append(np.asarray(output[1])[:, 3, :])
pse = np.asarray(pse)

# Plot the result -
title = "ThompSECONDARY"
corresp = braidPlot(pse, conn, "surface", rho_vals, title=title)

# Define rho for further analysis
rho = 0.001
output = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
    init_He=3.25, init_Hi=22, init_taue=10, rho=rho, toxicSynergy=3,
    prodAB=0.75, clearAB=1, transAB2t=1, clearABt=1,
    prodTAU=0.5, clearTAU=1, transTAU2t=0.75, clearTAUt=1). \
    run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]

corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic", plot="s", title=title)
animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"],  "PETtoxic", conn, timeref=True, title=title)


"""
HYBRID MODEL.
Heterogeneous parameters.
"Seeding patterns idential to primary."  - 
"""

AB_initMap = [0.75 for roi in conn.region_labels]
# Toxic seeding AB: temporobasal and frontomedial ABt seeding (1%): ¿57rois? Lausanne multi-resolution parcellation
# Maybe this is inferior temporal and medial frontal?
ABt_seeds = ['ctx-lh-inferiortemporal', 'ctx-lh-superiorfrontal', 'ctx-lh-medialorbitofrontal',
             'ctx-rh-inferiortemporal', 'ctx-rh-superiorfrontal', 'ctx-rh-medialorbitofrontal']
ABt_initMap = [sum(AB_initMap) * 0.01 / len(ABt_seeds) if roi in ABt_seeds else 0 for roi in conn.region_labels]

TAU_initMap = [0.5 for roi in conn.region_labels]
# Toxic seeding TAU: transenthorhinal and locus coeruleus TAUt seeding (1%); LC not in our atlas.
TAUt_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]
TAUt_initMap = [sum(AB_initMap) * 0.01 / len(TAUt_seeds) if roi in TAUt_seeds else 0 for roi in conn.region_labels]

## Parameter heterogeneity

b2_hetero = \
    np.array([('ctx-lh-parsopercularis', 7.452),
     ('ctx-lh-superiorfrontal', 7.542),
     ('ctx-lh-precentral', 5.589),
     ('ctx-lh-lateralorbitofrontal', 6.486),
     ('ctx-lh-parstriangularis', 5.52e-06),
     ('ctx-lh-posteriorcingulate', 3.45),
     ('ctx-lh-middletemporal', 11.04),
     ('ctx-lh-superiortemporal', 8.28),
     ('ctx-lh-cuneus', 13.8),
     ('ctx-lh-inferiorparietal', 11.73),
     ('ctx-lh-lingual', 13.8),
     ('ctx-lh-parahippocampal', 11.04),
     ('ctx-lh-rostralmiddlefrontal', 6.707),
     ('ctx-lh-caudalmiddlefrontal', 7.452),
     ('ctx-lh-postcentral', 3.726),
     ('ctx-lh-medialorbitofrontal', 6.486),
     ('ctx-lh-rostralanteriorcingulate', 6.21e-06),
     ('ctx-lh-inferiortemporal', 13.11),
     ('ctx-lh-superiortemporal', 8.97),
     ('ctx-lh-superiorparietal', 12.42),
     ('ctx-lh-pericalcarine', 13.8),
     ('ctx-lh-lateraloccipital', 15.18),
     ('ctx-lh-fusiform', 7.59),
     ('ctx-lh-temporalpole', 1.104e-05),
     ('ctx-lh-entorhinal', 3.125),
     ('Left-Pallidum', 2.76),
     ('Left-Putamen', 3.795),
     ('ctx-lh-precuneus', 3.105),
     ('ctx-rh-parsopercularis', 7.452),
     ('ctx-rh-superiorfrontal', 7.542),
     ('ctx-rh-precentral', 5.589),
     ('ctx-rh-lateralorbitofrontal', 6.486),
     ('ctx-rh-parstriangularis', 5.52e-06),
     ('ctx-rh-posteriorcingulate', 3.45),
     ('ctx-rh-middletemporal', 11.04),
     ('ctx-rh-superiortemporal', 8.28),
     ('ctx-rh-cuneus', 13.8),
     ('ctx-rh-inferiorparietal', 11.73),
     ('ctx-rh-lingual', 13.8),
     ('ctx-rh-parahippocampal', 11.04),
     ('ctx-rh-rostralmiddlefrontal', 6.707),
     ('ctx-rh-caudalmiddlefrontal', 7.452),
     ('ctx-rh-postcentral', 3.726),
     ('ctx-rh-medialorbitofrontal', 6.486),
     ('ctx-rh-rostralanteriorcingulate', 6.21e-06),
     ('ctx-rh-inferiortemporal', 13.11),
     ('ctx-rh-superiortemporal', 8.97),
     ('ctx-rh-superiorparietal', 12.42),
     ('ctx-rh-pericalcarine', 13.8),
     ('ctx-rh-lateraloccipital', 15.18),
     ('ctx-rh-fusiform', 7.59),
     ('ctx-rh-temporalpole', 1.104e-05),
     ('ctx-rh-entorhinal', 3.125),
     ('Right-Pallidum', 2.76),
     ('Right-Putamen', 3.795),
     ('ctx-rh-precuneus', 3.105)])

b3_hetero = \
    np.array([('ctx-lh-entorhinal', 1.104e-05),
     ('Left-Pallidum', 2.76),
     ('Left-Putamen', 3.795),
     ('ctx-lh-precuneus', 3.1),
     ('ctx-rh-entorhinal', 1.104e-05),
     ('Right-Pallidum', 2.76),
     ('Right-Putamen', 3.795),
     ('ctx-rh-precuneus', 3.1)])


b2 = [1.035 if roi not in b2_hetero[:, 0] else float(b2_hetero[list(b2_hetero[:, 0]).index(roi), 1]) for roi in conn.region_labels]

b3 = [4.14 if roi not in b3_hetero[:, 0] else float(b3_hetero[list(b3_hetero[:, 0]).index(roi), 1]) for roi in conn.region_labels]

# 1) First, try out their rho values;
output = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
    init_He=3.25, init_Hi=22, init_taue=10, rho=[1.38, 0.138, 1.38, 0.014], toxicSynergy=b3,
    prodAB=1.035, clearAB=1.38, transAB2t=1.38, clearABt=0.828,
    prodTAU=0.69, clearTAU=1.38, transTAU2t=b2, clearTAUt=0.552). \
    run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]

# Plot the result -
title = "ThompHYBRID"
if not np.isnan(np.asarray(output[1])):
    braidPlot(output, conn, "diagram", rho_vals, title=title)

# 2) Then, compute braid surface.
pse = []
for rho in rho_vals:
    output = ProteinSpreadModel(
        conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
        init_He=3.25, init_Hi=22, init_taue=10, rho=0, toxicSynergy=b3,
        prodAB=1.035, clearAB=1.38, transAB2t=1.38, clearABt=0.828,
        prodTAU=0.69, clearTAU=1.38, transTAU2t=b2, clearTAUt=0.552). \
        run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]
    ## [subj, "jr", 3, 4.5, 10]
    pse.append(np.asarray(output[1])[:, 3, :])
pse = np.asarray(pse)

# Plot the result -
corresp = braidPlot(pse, conn, "surface", rho_vals, title=title)

# Define rho for further analysis
rho = 0.001
output = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam=0, TAU_initdam=0,
    init_He=3.25, init_Hi=22, init_taue=10, rho=0, toxicSynergy=b3,
    prodAB=1.035, clearAB=1.38, transAB2t=1.38, clearABt=0.828,
    prodTAU=0.69, clearTAU=1.38, transTAU2t=b2, clearTAUt=0.552). \
    run(time=time, dt=dt, sim=False, sim_dt=4)  # sim [False; simParams[subj, model, g, s, time(s)]]

corrs_PET = correlations(output, ["CN", "SMC", "EMCI", "LMCI", "AD"], reftype="PETtoxic", plot="s", title=title)
animate_propagation_v4(output, corrs_PET, ["CN", "SMC", "EMCI", "LMCI", "AD"], "PETtoxic", conn, timeref=True,
                       title=title)
