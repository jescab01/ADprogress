
import numpy as np
import pandas as pd
from zipfile import ZipFile
from collections import Counter

from tvb.simulator.lab import connectivity

"""
Correlate full empirical matrices per condition
"""

data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

##   1. C3N dataset    #########################

groups = ["HC-fam", "FAM", "HC", "QSM", "MCI", "MCI-conv"]

####  1.1 SC // FC AVERAGE - Loop over conditions to correlate

corrSC = np.zeros((len(groups), len(groups)))
corrFC = np.zeros((len(groups), len(groups)))

for i, group1 in enumerate(groups):
    for j, group2 in enumerate(groups):
        for mode in ["SC", "FC"]:

            if mode == "SC":

                # Load SC data
                conn1 = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + group1 + "_aparc_aseg-mni_09c.zip")
                conn2 = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + group2 + "_aparc_aseg-mni_09c.zip")

                # Correlate
                t1 = np.zeros(shape=(2, len(conn1.weights) ** 2 // 2 - len(conn1.weights) // 2))
                t1[0, :] = conn1.weights[np.triu_indices(len(conn1.weights), 1)]
                t1[1, :] = conn2.weights[np.triu_indices(len(conn1.weights), 1)]

                corrSC[i, j] = np.corrcoef(t1)[0, 1]

            elif mode == "FC":

                # Load SC data
                conn1 = np.loadtxt(data_folder + "FC_matrices/" + group1 + "_3-alpha_plv_rms.txt", delimiter=',')
                conn2 = np.loadtxt(data_folder + "FC_matrices/" + group2 + "_3-alpha_plv_rms.txt", delimiter=',')

                # Correlate
                t1 = np.zeros(shape=(2, len(conn1) ** 2 // 2 - len(conn1) // 2))
                t1[0, :] = conn1[np.triu_indices(len(conn1), 1)]
                t1[1, :] = conn2[np.triu_indices(len(conn1), 1)]

                corrFC[i, j] = np.corrcoef(t1)[0, 1]


##     2. ADNI dataset        ###########################

# Load data
ADNI_AVG = pd.read_csv(data_folder + 'ADNI/.PET_AVx_GroupAVERAGED.csv', index_col=0)

groups = ["CN", "SMC", "EMCI", "LMCI", "AD"]

corrAV45 = np.zeros((len(groups), len(groups)))
corrAV1451 = np.zeros((len(groups), len(groups)))

for x, mode in enumerate(["AV45", "AV1451"]):
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):

            # Select data
            pet1 = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == mode) & (ADNI_AVG["Group"] == group1)].iloc[:, 12:]))
            pet2 = np.squeeze(np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == mode) & (ADNI_AVG["Group"] == group2)].iloc[:, 12:]))

            # Correlate
            if mode == "AV45":
                corrAV45[i, j] = np.corrcoef(pet1, pet2)[0, 1]
            elif mode == "AV1451":
                corrAV1451[i, j] = np.corrcoef(pet1, pet2)[0, 1]

