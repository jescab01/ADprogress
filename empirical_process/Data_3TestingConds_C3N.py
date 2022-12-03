
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


##   1. C3N dataset    #########################

groups = ["HC-fam", "FAM", "HC", "QSM", "MCI", "MCI-conv"]

# Read participants.csv to define subj-condition relationship
c3n_participants = pd.read_csv(data_folder + "participants.csv")


### 1.1 Structural Connectivity
"""
Nothing here. Indeed, no changes should be expected in terms of number of streamlines: 
with tau what it is altered is synaptic terminals (dendritic spines) not the tracts per se.
"""

# a. Prepare dataset for statistical testing: DataFrame
# data_sc = pd.DataFrame()
#
# for group in groups:
#     print('Working on SC - %s group  ::  ' % group)
#     subset = c3n_participants.loc[c3n_participants["Joint_sample"] == group]
#
#     for subject in subset.ID_BIDS.values:
#
#         conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subject + "_aparc_aseg-mni_09c.zip")
#         array = conn.weights[np.triu_indices(len(conn.weights), 1)]
#         labelconn = np.asarray([[roi1 + " || " + roi2 for roi2 in conn.region_labels] for roi1 in conn.region_labels])[np.triu_indices(len(conn.weights), 1)]
#
#         data_sc = data_sc.append(pd.DataFrame([[group]*len(labelconn), [subject]*len(labelconn), labelconn, array]).transpose())
#
# data_sc.to_csv(data_folder + "SC_matrices/.SC_dataframe_perConn.csv")
data_sc = pd.read_csv(data_folder + "SC_matrices/.SC_dataframe_perConn.csv", index_col=0)
data_sc.columns = ["cond", "subj", "conn", "value"]
data_sc = data_sc.astype({"value": float})

# b. Statistical testing SC - ANOVA/Kruskal comparisons over conditions per connection
# anova_sc = pd.DataFrame()
#
# ## Structural Connectivity tests
# tic = time.time()
# # ANOVA per roi with condition as factor
# for i, conn in enumerate(set(data_sc.conn.values)):
#
#     print("SC Tests  -  %i/%i  - %s" % (i+1, len(set(data_sc.conn.values)), conn), end="\r")
#     subset = data_sc.loc[data_sc["conn"] == conn]
#
#     # TODO check for ANOVA assumptions
#     test = pg.kruskal(data=subset, dv="value", between="cond")
#     test["conn"] = conn
#     anova_sc = anova_sc.append(test)
#
# print("SC Tests  -  %i/%i  - %s         . time %0.2fm" % (i + 1, len(set(data_sc.conn.values)), conn, (time.time()-tic)/60))
#
# # Correct for multiple comparisons
# anova_sc = anova_sc.reset_index()
# anova_sc["p-adj"] = pg.multicomp(anova_sc["p-unc"].values, method="holm")[1]
anova_sc = pd.read_csv(data_folder + "SC_matrices/.SC_anovas_perConn.csv")



### 1.2 Functional Connectivity
"""
TODO whats here? Differences between FAM and the rest. :/
"""
# a. Prepare dataset for statistical testing: DataFrame
# data_fc = pd.DataFrame()
#
# for group in groups:
#     print('Working on FC - %s group  ::  ' % group)
#     subset = c3n_participants.loc[c3n_participants["Joint_sample"] == group]
#
#     for band in ["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"]:
#
#         for subject in subset.ID_BIDS.values:
#             conn = np.loadtxt(data_folder + "FCavg_matrices/" + subject + "_" + band + "_plv_avg.txt", delimiter=',')
#             array = conn[np.triu_indices(len(conn), 1)]
#
#             labs = np.loadtxt(data_folder + "FCavg_matrices/" + subject + "_roi_labels.txt", delimiter=',', dtype=str)
#             labelconn = np.asarray([[roi1 + " || " + roi2 for roi2 in labs] for roi1 in labs])[np.triu_indices(len(conn), 1)]
#
#             data_fc = data_fc.append(pd.DataFrame([[group]*len(labelconn), [subject]*len(labelconn), [band]*len(labelconn), labelconn, array]).transpose())
#
# # The process is slow: save and load
# data_fc.to_csv(data_folder + "FCavg_matrices/.FC_dataframe_perConn.csv")
data_fc = pd.read_csv(data_folder + "FCavg_matrices/.FC_dataframe_perConn.csv", index_col=0)
data_fc.columns = ["cond", "subj", "band", "conn", "value"]
data_fc = data_fc.astype({"value": float})

# b. Statistical testing FC - ANOVA/Kruskal comparisons over conditions per connection
# anova_fc = pd.DataFrame()
#
# tic = time.time()
# for i, conn in enumerate(set(data_fc.conn.values)):
#     for band in ["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"]:
#         subset = data_fc.loc[(data_fc["conn"] == conn) & (data_fc["band"] == band)]
#         print("FC Tests  -  %i/%i  - %s   | band: %s" % (i + 1, len(set(data_fc.conn.values)), conn, band), end="\r")
#
#         # TODO check for ANOVA assumptions and adjust bands ([only alpha] or all?)
#
#         test = pg.kruskal(data=subset, dv="value", between="cond")
#         test["band"] = band
#         test["conn"] = conn
#         anova_fc = anova_fc.append(test)
#
# print("FC Tests  -  %i/%i  - %s         . time %0.2fm" % (i + 1, len(set(data_fc.conn.values)), conn, (time.time()-tic)/60))
#
# # Correct for multiple comparisons
# anova_fc = anova_fc.reset_index()
# anova_fc["p-adj"] = pg.multicomp(anova_fc["p-unc"].values, method="holm")[1]
#
# # Save and load
# anova_fc.to_csv(data_folder + "FCavg_matrices/.FC_anovas_perConn.csv")
anova_fc = pd.read_csv(data_folder + "FCavg_matrices/.FC_anovas_perConn.csv")

# c. Tukey tests - Follow to multiple comparisons
# posthoc = pd.DataFrame()
#
# tic = time.time()
# for i, conn in enumerate(set(anova_fc.conn.values)):
#     for band in ["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"]:
#         subset = data_fc.loc[(data_fc["conn"] == conn) & (data_fc["band"] == band)]
#         print("FC Tests  -  %i/%i  - %s   | band: %s" % (i + 1, len(set(anova_fc.conn.values)), conn, band), end="\r")
#
#         test = pg.pairwise_tukey(data=subset, dv="value", between="cond")
#         test["band"] = [band] * len(test)
#         test["conn"] = [conn] * len(test)
#         posthoc = posthoc.append(test)
#
# print("FC Tests  -  %i/%i  - %s         . time %0.2fm" % (i + 1, len(set(anova_fc.conn.values)), conn, (time.time()-tic)/60))
#
# # Save and load posthoc tests
# posthoc.to_csv(data_folder + "FCavg_matrices/.FC_posthocs_perConn.csv")
posthoc = pd.read_csv(data_folder + "FCavg_matrices/.FC_posthocs_perConn.csv")



### 1.3 Spectral Power
"""
What happens here? Sth strange: everything is significant in ANOVA, nothing is significant in Tukey. wtf.
"""

# a. Prepare data for testing
# data_psd = pd.DataFrame()
#
# freqs = np.loadtxt(data_folder + "SpectralPower/FREQS_spectra_dpss.txt", delimiter=",")
# labs = np.loadtxt(data_folder + "SpectralPower/LABELS_spectra_dpss.txt", delimiter=',', dtype=str)
#
# for group in groups:
#     print('Working on Spectral Power - %s group  ::  ' % group)
#     subset = c3n_participants.loc[c3n_participants["Joint_sample"] == group]
#
#     for subject in subset.ID_BIDS.values:
#         spectra = np.loadtxt(data_folder + "SpectralPower/" + subject + "_spectra_dpss.txt", delimiter=",")
#
#         for bandname, bandrange in zip(["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]):
#             auc = np.trapz(spectra[:, (bandrange[0] <= freqs) & (freqs <= bandrange[1])])
#             data_psd = data_psd.append(pd.DataFrame([[group] * len(auc), [subject] * len(auc), [bandname] * len(auc), labs, auc]).transpose())
#
# # The process is slow: save and load
# data_psd.to_csv(data_folder + "SpectralPower/.PSD_dataframe_perROI.csv")
data_psd = pd.read_csv(data_folder + "SpectralPower/.PSD_dataframe_perROI.csv", index_col=0)
data_psd.columns = ["cond", "subj", "band", "roi", "value"]
data_psd = data_psd.astype({"value": float})

# b. Statistical testing - ANOVA/Kruskal over spectral power
# anova_psd = pd.DataFrame()
#
# tic = time.time()
# for i, roi in enumerate(set(data_psd.roi.values)):
#     for band in ["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"]:
#         subset = data_psd.loc[(data_psd["roi"] == roi) & (data_psd["band"] == band)]
#         print("PSD Tests  -  %i/%i  - %s   | band: %s" % (i + 1, len(set(data_psd.roi.values)), roi, band), end="\r")
#
#         # TODO check for ANOVA assumptions
#
#         test = pg.kruskal(data=subset, dv="value", between="cond")
#         test["band"] = band
#         test["roi"] = roi
#         anova_psd = anova_psd.append(test)
#
# print("PSD Kruskal Tests  -  %i/%i  - %s         . time %0.2fm" % (i + 1, len(set(data_psd.roi.values)), roi, (time.time()-tic)/60))
#
# # Correct for multiple comparisons
# anova_psd = anova_psd.reset_index()
# anova_psd["p-adj"] = pg.multicomp(anova_psd["p-unc"].values, method="holm")[1]
#
# # Save and load
# anova_psd.to_csv(data_folder + "SpectralPower/.PSD_anovas_perROI.csv")
anova_psd = pd.read_csv(data_folder + "SpectralPower/.PSD_anovas_perROI.csv", index_col=0)

# c. Tukey tests - Follow to multiple comparisons
# posthoc = pd.DataFrame()
#
# tic = time.time()
# for i, roi in enumerate(set(anova_psd.roi.values)):
#     for band in ["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"]:
#         subset = data_psd.loc[(data_psd["roi"] == roi) & (data_psd["band"] == band)]
#         print("PSD Tukey Tests  -  %i/%i  - %s   | band: %s" % (i + 1, len(set(anova_psd.roi.values)), roi, band), end="\r")
#
#         test = pg.pairwise_tukey(data=subset, dv="value", between="cond")
#         test["band"] = [band] * len(test)
#         test["roi"] = [roi] * len(test)
#         posthoc = posthoc.append(test)
#
# print("PSD Tukey Tests  -  %i/%i  - %s         . time %0.2fm" % (i + 1, len(set(anova_psd.roi.values)), roi, (time.time()-tic)/60))
#
# # Save and load posthoc tests
# posthoc.to_csv(data_folder + "SpectralPower/.PSD_posthocs_perROI.csv")
posthoc = pd.read_csv(data_folder + "SpectralPower/.PSD_posthocs_perROI.csv", index_col=0)


posthoc[posthoc["p-tukey"]<0.05]