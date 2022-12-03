
import numpy as np
import pandas as pd
from zipfile import ZipFile
from collections import Counter

from tvb.simulator.lab import connectivity

"""
This script is intended to be executed once to save the needed averaged matrices. 
"""

data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

##   1. C3N dataset    #########################

# Read participants.csv to define subj-condition relationship
c3n_participants = pd.read_csv(data_folder + "participants.csv")


####  1.1 SC AVERAGE - Loop over conditions to make average them out
for group in list(set(c3n_participants.Joint_sample)):
    print('Working on SC - %s group  ::  ' % group)
    subset = c3n_participants.loc[c3n_participants["Joint_sample"] == group]

    # loop over this subset of subjects.
    temp_w, temp_tl = [], []
    for subj in list(set(subset.ID_BIDS)):
        print(subj, end="\r")
        conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
        temp_w.append(conn.weights)
        temp_tl.append(conn.tract_lengths)
    print(subj)
    temp_w = np.average(np.asarray(temp_w), axis=0)
    temp_tl = np.average(np.asarray(temp_tl), axis=0)

    np.savetxt(data_folder + 'SC_matrices/temp/avg-weights.txt', temp_w)
    np.savetxt(data_folder + 'SC_matrices/temp/avg-tract_lengths.txt', temp_tl)

    zip_name = group + '_aparc_aseg-mni_09c.zip'
    zipObj = ZipFile(data_folder + 'SC_matrices/' + zip_name, 'w')
    # Add multiple files to the zip
    zipObj.write(data_folder + 'SC_matrices/temp/avg-weights.txt', 'weights.txt')
    zipObj.write(data_folder + 'SC_matrices/temp/avg-tract_lengths.txt', 'tract_lengths.txt')
    zipObj.write(data_folder + 'SC_matrices/temp/centres.txt', 'centres.txt')
    zipObj.close()



####   1.2 FC AVERAGE - Loop over conditions to make average them out

for group in list(set(c3n_participants.Joint_sample)):
    subset = c3n_participants.loc[c3n_participants["Joint_sample"] == group]

    # loop over subjects and bands.
    for band in ["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"]:
        temp = []
        for subj in list(set(subset.ID_BIDS)):
            temp.append(np.loadtxt(data_folder + "FCavg_matrices/" + subj + "_" + band + "_plv_avg.txt", delimiter=','))

        temp = np.average(np.asarray(temp), axis=0)

        np.savetxt(data_folder + 'FCavg_matrices/' + group + '_' + band + '_plv_avg.txt', temp, delimiter=",")



##     2. ADNI dataset        ###########################

ADNI_database = pd.read_csv(data_folder + '/ADNI/ADNIMERGE_database.csv')
PET_AV45 = pd.read_csv(data_folder + '/ADNI/PET_AV45.csv', index_col=0)
PET_AV1451 = pd.read_csv(data_folder + '/ADNI/PET_AV1451.csv', index_col=0)

AVERAGEdf = pd.DataFrame()
for group in list(set(ADNI_database.DX_bl)):

    print('Working on AV45 PET - %s group:' % group)

    subset_adni = pd.DataFrame(columns=ADNI_database.columns)
    subset_av45 = pd.DataFrame(columns=PET_AV45.columns)
    for i, row in PET_AV45.iterrows():
        print('     .     %i/%i' % (i+1, len(PET_AV45)), end="\r")
        if any(ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (
                ADNI_database["vis_month"] == row.vis_month)].DX_bl.values):
            if ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)].DX_bl.values[0] == group:
                # Use just one recording per subject. Avoid sample biases.
                if row.subject_id not in subset_adni.subject_id.values:
                    subset_adni = subset_adni.append(ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)])
                    subset_av45 = subset_av45.append(row)
    print('     .     %i/%i' % (i + 1, len(PET_AV45)))
    # Add new row with averaged data to Dataframe
    data = {"Group": group, "PET":"AV45", "mAge": round(subset_adni.AGE.mean(), 4),
            "nMale": Counter(subset_adni.PTGENDER)["Male"], "nFemale": Counter(subset_adni.PTGENDER)["Female"],
            "mVisMonth_adni": round(subset_adni.vis_month.mean(), 2),
            "mVisMonth_PET": round(subset_av45.vis_month.mean(), 2)} | dict(subset_av45.iloc[:, 2:].mean())

    AVERAGEdf = AVERAGEdf.append(pd.DataFrame(data, index=[0]))

    print("   .    DONE\n")


    print('Working on AV1451 PET - %s group...' % group)

    subset_adni = pd.DataFrame(columns=ADNI_database.columns)
    subset_av1451 = pd.DataFrame(columns=PET_AV1451.columns)
    for i, row in PET_AV1451.iterrows():
        print('     .     %i/%i' % (i+1, len(PET_AV1451)), end="\r")
        if any(ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)].DX_bl.values):
            if ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)].DX_bl.values[0] == group:
                # Use just one recording per subject. Avoid sample biases.
                if row.subject_id not in subset_adni.subject_id.values:
                    subset_adni = subset_adni.append(ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)])
                    subset_av1451 = subset_av1451.append(row)
    print('     .     %i/%i' % (i + 1, len(PET_AV1451)))
    # Add new row with averaged data to Dataframe
    data = {"Group": group, "PET": "AV1451", "mAge": round(subset_adni.AGE.mean(), 4),
            "nMale": Counter(subset_adni.PTGENDER)["Male"], "nFemale": Counter(subset_adni.PTGENDER)["Female"],
            "mVisMonth_adni": round(subset_adni.vis_month.mean(), 2),
            "mVisMonth_PET": round(subset_av1451.vis_month.mean(), 2)} | dict(subset_av1451.iloc[:, 2:].mean())

    AVERAGEdf = AVERAGEdf.append(pd.DataFrame(data, index=[0]))

    print("   .    DONE\n")

AVERAGEdf_ = AVERAGEdf.dropna()
AVERAGEdf_.to_csv(data_folder + 'ADNI/.PET_AVx_GroupAVERAGED.csv')


