
import time
import numpy as np
import pandas as pd
import pingouin as pg
from zipfile import ZipFile
from collections import Counter

from tvb.simulator.lab import connectivity

"""
Statistically testing specific ROI differences through ADpg
"""

data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

# Load data
ADNI_database = pd.read_csv(data_folder + '/ADNI/ADNIMERGE_database.csv')
PET_AV45 = pd.read_csv(data_folder + '/ADNI/PET_AV45.csv', index_col=0)
PET_AV1451 = pd.read_csv(data_folder + '/ADNI/PET_AV1451.csv', index_col=0)

ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)


##     2. ADNI dataset        ###########################

groups = ["CN", "SMC", "EMCI", "LMCI", "AD"]

### 2.1. PREPARE a long dataframe for pingouin tests
# DFlong = pd.DataFrame()
#
# for group in groups:
#
#     print('Working on AV45 PET - %s group:' % group)
#     subset_adni = pd.DataFrame(columns=ADNI_database.columns)  # Not to repeat subjects in sample
#     for i, row in PET_AV45.iterrows():
#         print('     .     %i/%i' % (i+1, len(PET_AV45)), end="\r")
#         if any(ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)].DX_bl.values):
#             if ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)].DX_bl.values[0] == group:
#                 # Use just one recording per subject. Avoid sample biases.
#                 if row.subject_id not in subset_adni.subject_id.values:
#                     subset_adni = subset_adni.append(ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)])
#
#                     ## Add info to long dataframe
#                     petvals = row.iloc[12:].reset_index()
#                     petvals.columns = ["roi", "value"]
#                     temp = pd.DataFrame([[group]*len(petvals), [row.subject_id] * len(petvals), ["AV45"]*len(petvals), petvals.roi.values, petvals.value.values])
#                     DFlong = DFlong.append(temp.transpose())
#     print('     .     %i/%i' % (i + 1, len(PET_AV45)))
#     print("   .    DONE\n")
#
#
#     print('Working on AV1451 PET - %s group...' % group)
#     subset_adni = pd.DataFrame(columns=ADNI_database.columns)
#     for i, row in PET_AV1451.iterrows():
#         print('     .     %i/%i' % (i+1, len(PET_AV1451)), end="\r")
#         if any(ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)].DX_bl.values):
#             if ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)].DX_bl.values[0] == group:
#                 # Use just one recording per subject. Avoid sample biases.
#                 if row.subject_id not in subset_adni.subject_id.values:
#                     subset_adni = subset_adni.append(ADNI_database.loc[(ADNI_database["subject_id"] == row.subject_id) & (ADNI_database["vis_month"] == row.vis_month)])
#
#                     ## Add info to long dataframe
#                     petvals = row.iloc[12:].reset_index()
#                     petvals.columns = ["roi", "value"]
#                     temp = pd.DataFrame([[group]*len(petvals), [row.subject_id] * len(petvals), ["AV1451"]*len(petvals), petvals.roi.values, petvals.value.values])
#                     DFlong = DFlong.append(temp.transpose())
#     print('     .     %i/%i' % (i + 1, len(PET_AV1451)))
#     print("   .    DONE\n")

# # Save the DFlong
# DFlong.to_csv(data_folder + '/ADNI/.1PET_longdf_perROI.csv')
DFlong = pd.read_csv(data_folder + '/ADNI/.PET_1longdf_perROI.csv', index_col=0)
DFlong.columns = ["cond", "subj", "pet", "roi", "value"]
DFlong = DFlong.astype({"value": float})


###  2.2 PET - ANOVA/Kruskal comparisons over conditions per connection
# anova_pet = pd.DataFrame()
#
# ## Structural Connectivity tests
# tic = time.time()
# # ANOVA per roi with condition as factor
# for i, roi in enumerate(set(DFlong.roi.values)):
#     for pet in ["AV45", "AV1451"]:
#         print("PET %s Tests  -  %i/%i  - %s" % (pet, i+1, len(set(DFlong.roi.values)), roi), end="\r")
#         subset = DFlong.loc[(DFlong["roi"] == roi) & (DFlong["cond"].isin(groups) & (DFlong["pet"]==pet))]
#
#         # TODO check for ANOVA assumptions
#
#         test = pg.kruskal(data=subset, dv="value", between="cond")
#         test["roi"] = roi
#         test["pet"] = pet
#         anova_pet = anova_pet.append(test)
#
# print("PET %s Tests  -  %i/%i  - %s         . time %0.2fm" % (pet, i + 1, len(set(DFlong.roi.values)), roi, (time.time()-tic)/60))
#
#
# # Correct for multiple comparisons
# anova_pet = anova_pet.reset_index()
# anova_pet["p-adj"] = pg.multicomp(anova_pet["p-unc"].values, method="holm")[1]
#
# # Save & load
# anova_pet.to_csv(data_folder + "ADNI/.2PET_anovas_perROI.csv")
anova_pet = pd.read_csv(data_folder + "ADNI/.PET_2anovas_perROI.csv")


### 2.3 POSTHOC tests : Follow to multiple comparisons in each significant ANOVA
# posthoc = pd.DataFrame()
#
# tic = time.time()
# for i, roi in enumerate(set(anova_pet.roi.values)):
#     for pet in ["AV45", "AV1451"]:
#         subset = DFlong.loc[(DFlong["roi"] == roi) & (DFlong["pet"] == pet)]
#         print("PET Tests  -  %i/%i  - %s " % (i + 1, len(set(anova_pet.roi.values)), roi), end="\r")
#
#         test = pg.pairwise_tukey(data=subset, dv="value", between="cond")
#         test["roi"] = [roi] * len(test)
#         test["pet"] = [pet] * len(test)
#         posthoc = posthoc.append(test)
#
# print("PET Tests  -  %i/%i  - %s         . time %0.2fm" % (i + 1, len(set(anova_pet.roi.values)), roi, (time.time()-tic)/60))
#
# posthoc.to_csv(data_folder + "ADNI/.PET_3posthocs_perConn.csv")
posthoc=pd.read_csv(data_folder + "ADNI/.PET_3posthocs_perConn.csv")


