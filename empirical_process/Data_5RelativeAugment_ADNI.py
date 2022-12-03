
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

"""
Make the PET averages relative to healthy concentration
"""

data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"


##    ADNI dataset        ###########################
# Load data
ADNI_AVG = pd.read_csv(data_folder + 'ADNI/.PET_AVx_GroupAVERAGED.csv', index_col=0)

groups = ["CN", "SMC", "EMCI", "LMCI", "AD"]


## ABSOLUTE values
df_abs = pd.DataFrame([["ABS", group, pet, roi, ADNI_AVG.loc[(ADNI_AVG["PET"] == pet) & (ADNI_AVG["Group"] == group), roi].values[0]]
                       for group in groups for pet in ["AV45", "AV1451"] for roi in ADNI_AVG.columns[12:]],
                       columns=["mode", "Group", "PET", "roi", "value"])


## RELATIVE to CONTROL GROUP
ADNI_AVGrel = ADNI_AVG.copy()
for x, mode in enumerate(["AV45", "AV1451"]):
    ref = ADNI_AVG.loc[(ADNI_AVG["PET"] == mode) & (ADNI_AVG["Group"] == "CN")].iloc[:, 7:].values[0]

    for i, group in enumerate(groups):
        g = ADNI_AVG.loc[(ADNI_AVG["PET"] == mode) & (ADNI_AVG["Group"] == group)].iloc[:, 7:].values[0]

        ADNI_AVGrel.loc[(ADNI_AVGrel["PET"] == mode) & (ADNI_AVGrel["Group"] == group), ADNI_AVGrel.columns[7:]] = g - ref

ADNI_AVGrel.to_csv(data_folder + 'ADNI/.PET_AVx_GroupREL_2CN.csv')


df_rel_2cn = pd.DataFrame([["REL_2CN", group, pet, roi, ADNI_AVGrel.loc[(ADNI_AVGrel["PET"] == pet) & (ADNI_AVGrel["Group"] == group), roi].values[0]]
                       for group in groups for pet in ["AV45", "AV1451"] for roi in ADNI_AVGrel.columns[12:]],
                      columns=["mode", "Group", "PET", "roi", "value"])


## RELATIVE to PREVIOUS STAGE
ADNI_AVGrel_2PS = []
for x, mode in enumerate(["AV45", "AV1451"]):

    for i, group1 in enumerate(groups[:-1]):
        ref = ADNI_AVG.loc[(ADNI_AVG["PET"] == mode) & (ADNI_AVG["Group"] == group1)].iloc[:, 7:].values[0]
        group2 = groups[i+1]
        g2 = ADNI_AVG.loc[(ADNI_AVG["PET"] == mode) & (ADNI_AVG["Group"] == group2)].iloc[:, 7:].values[0]
        ADNI_AVGrel_2PS.append([group1 + "-" + group2, mode] + list(g2-ref))

ADNI_AVGrel_2PS = pd.DataFrame(ADNI_AVGrel_2PS, columns=["Transition", "PET"] + list(ADNI_AVG.columns[7:]))
ADNI_AVGrel_2PS.to_csv(data_folder + 'ADNI/.PET_AVx_GroupREL_2PrevStage.csv')


df_rel_2ps = pd.DataFrame([["REL_2PS", group, pet, roi, ADNI_AVGrel_2PS.loc[(ADNI_AVGrel_2PS["PET"] == pet) & (ADNI_AVGrel_2PS["Transition"] == group), roi].values[0]]
                       for group in ["CN-SMC", "SMC-EMCI", "EMCI-LMCI", "LMCI-AD"] for pet in ["AV45", "AV1451"] for roi in ADNI_AVGrel_2PS.columns[7:]],
                      columns=["mode", "Group", "PET", "roi", "value"])


df = pd.concat([df_abs, df_rel_2cn, df_rel_2ps])
fig = px.line(df, x="Group", y="value", facet_col="PET", color="roi", facet_row="mode", facet_col_spacing=0.1,
              title="PET rises: Absolute values; relative to CN; and relative to previous disease stage")
fig.update_yaxes(matches=None)
fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True, matches=None))
pio.write_html(fig, "empirical_process/figures/ADNI_PETRises.html", auto_open=True)
