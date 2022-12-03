
import time
import numpy as np
import pandas as pd
import pingouin as pg
from zipfile import ZipFile
from collections import Counter

from tvb.simulator.lab import connectivity

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

"""
Aquí tiene que venir una funcion que me permita visualizar los resultados de varias maneras:

- por bandas y una de all
- con solo significativas o con top 10
-

"""

def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

## Prepare brain volume for plotting
# Idea from Matteo Mancini: https://neurosnippets.com/posts/interactive-network/
def obj_data_to_mesh3d(ofile):

    with open(ofile, "r") as f:
        odata = f.read()

    # odata is the string read from an obj file
    vertices = []
    faces = []
    lines = odata.splitlines()

    for line in lines:
        slist = line.split()
        if slist:
            if slist[0] == 'v':
                vertex = np.array(slist[1:], dtype=float)
                vertices.append(vertex)
            elif slist[0] == 'f':
                face = []
                for k in range(1, len(slist)):
                    face.append([int(s) for s in slist[k].replace('//', '/').split('/')])
                if len(face) > 3:  # triangulate the n-polyonal face, n>3
                    faces.extend(
                        [[face[0][0] - 1, face[k][0] - 1, face[k + 1][0] - 1] for k in range(1, len(face) - 1)])
                else:
                    faces.append([face[j][0] - 1 for j in range(len(face))])
            else:
                pass

    return np.array(vertices), np.array(faces)

vertices, faces = obj_data_to_mesh3d("E:\OneDrive - Universidad Complutense de Madrid (UCM)\TheoryOfMind\Figures\lh.pial_simplified.obj")

vert_x, vert_y, vert_z = vertices[:, :3].T
face_i, face_j, face_k = faces.T


##  PLOT results
data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/HC-fam_aparc_aseg-mni_09c.zip")


#### 1. ADNI - Two rows - SIGNIFICANT
"""
Works well. Showing first amiloyd-beta to rise concentration and tau after. 
"""
anova_pet = pd.read_csv(data_folder + "ADNI/.PET_2anovas_perROI.csv", index_col=0)
posthoc=pd.read_csv(data_folder + "ADNI/.PET_3posthocs_perConn.csv", index_col=0)
posthoc["roi_lower"] = [roi.lower() for roi in posthoc["roi"].values]

groups = ["CN", "SMC", "EMCI", "LMCI", "AD"]

# As SIGNIFICANT in mode: filter anovas and then posthocs
sub_anovas = anova_pet[anova_pet["p-adj"] <= 0.05]
sub_posthocs = posthoc[(posthoc["roi"].isin(sub_anovas.roi.values)) & (posthoc["p-tukey"] <= 0.05) & (posthoc["roi_lower"].isin(conn.region_labels))].copy()

# Pre-define SIZE
range_size = [5, 50]
sub_posthocs["size"] = abs(sub_posthocs["diff"].values) * (range_size[1] - range_size[0]) + (range_size[0] - np.min(abs(sub_posthocs["diff"].values)))

# pre-define COLOUR per difference
sub_posthocs["color_out"] = ["royalblue" if np.sign(row["diff"])<0 else "indianred" for i, row in sub_posthocs.iterrows()]
width = 20

# # Pre-define COLOUR per node
# cmap = px.colors.qualitative.Light24  # Alphabet, Dark24, Light24 have the wider range of colours.
# cmap = (cmap * 10)[:len(conn.region_labels)//2]
# cmap = cmap[0:7] + cmap[0:7] + cmap[7:] + cmap[7:]
# sub_posthocs["color_in"] =  [cmap[list(conn.region_labels).index(roi)] for roi in sub_posthocs.roi_lower.values]

sp_cols = [group+"->"+groups[i+1] for i, group in enumerate(groups[:-1])]
n_rows = 2

specs = [[{"type": "scene"}] * len(sp_cols)] * n_rows

fig = make_subplots(rows=n_rows, cols=len(sp_cols), horizontal_spacing=0, row_titles=["AV45", "AV1451"],
                    subplot_titles=sp_cols, specs=specs, shared_xaxes=True, shared_yaxes=True)

for j, sp in enumerate(sp_cols):
    group1, group2 = sp.split("->")[0], sp.split("->")[1]
    sub_sig = sub_posthocs.loc[
        (sub_posthocs["A"].isin([group1, group2])) & (sub_posthocs["B"].isin([group1, group2]))]

    if not sub_sig.empty:
        if sub_sig["A"].values[0] != group2:
            sub_sig = sub_sig.copy()
            sub_sig["diff"] = -sub_sig["diff"]
    # work on first row: AV45
    sub_av = sub_sig[(sub_sig["pet"]=="AV45") & (sub_sig["roi"].isin(conn.region_labels))]

    roi_idx = [list(conn.region_labels).index(roi) for roi in sub_av.roi.values]
    hovertext3d = [
        "<b>" + row["roi"] + "</b><br>" + "(g2-g1) diff " + str(round(row["diff"], 5))  for i, row in sub_av.iterrows()]

    fig.add_trace(go.Scatter3d(x=[conn.centres[roi_id, 0] for roi_id in roi_idx],
                               y=[conn.centres[roi_id, 1] for roi_id in roi_idx],
                               z=[conn.centres[roi_id, 2] for roi_id in roi_idx],
                               showlegend=False, hovertext=hovertext3d,
                               mode="markers", marker=dict(size=sub_av["size"], color=sub_av["diff"], colorscale="RdBu",
                                                           reversescale=True, cmin=-0.2, cmax=0.2),
                               ), row=1, col=j + 1)
    # work on second row: AV1451
    sub_av = sub_sig[(sub_sig["pet"] == "AV1451") & (sub_sig["roi"].isin(conn.region_labels))]

    roi_idx = [list(conn.region_labels).index(roi) for roi in sub_av.roi.values]
    hovertext3d = [
        "<b>" + row["roi"] + "</b><br>" + "(g2-g1) diff " + str(round(row["diff"], 5)) for i, row in sub_av.iterrows()]

    fig.add_trace(go.Scatter3d(x=[conn.centres[roi_id, 0] for roi_id in roi_idx],
                               y=[conn.centres[roi_id, 1] for roi_id in roi_idx],
                               z=[conn.centres[roi_id, 2] for roi_id in roi_idx],
                               showlegend=False, hovertext=hovertext3d,
                               mode="markers", marker=dict(size=sub_av["size"], color=sub_av["diff"], colorscale="RdBu",
                                                           reversescale=True, cmin=-0.2, cmax=0.2),
                               ), row=2, col=j + 1)

    # Add 3d brains
    fig.add_trace(go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
                            color='silver', opacity=0.2, showscale=False, hoverinfo='none'), row=1, col=j+1)
    fig.add_trace(go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
                            color='silver', opacity=0.2, showscale=False, hoverinfo='none'), row=2, col=j+1)

fig.update_layout(
         title='Significant changes in Protein deposition: ADNI dataset', template="plotly_white")

print("Saving images: may take a while")
# pio.write_image(fig, file='figures/ADpg_PETchanges.svg', height=800, width=1100)
pio.write_html(fig, file='figures/ADpg_PET-SIGNIFICANTchanges.html', auto_open=True)


#### 2.a. C3N (FC) - One row - SIGNIFICANT
"""
It does not work. The difference between FAM data and other is too large: 
probably the differences in the data preprocessing (before segments) makes a big deal in the final FC matrices.
However, do it and comment with Fernando what could be happening.
"""

anova_fc = pd.read_csv(data_folder + "FCavg_matrices/.FC_anovas_perConn.csv", index_col=0)
posthoc = pd.read_csv(data_folder + "FCavg_matrices/.FC_posthocs_perConn.csv", index_col=0)

groups = ["HC-fam", "FAM", "QSM", "MCI", "MCI-conv"]

# As SIGNIFICANT in mode: filter anovas and then posthocs
sub_anovas = anova_fc[anova_fc["p-adj"] <= 0.05]
sub_posthocs = posthoc[(posthoc["conn"].isin(sub_anovas.conn.values)) & (posthoc["p-tukey"] <= 0.05)].copy()

## Subset interstage significant changes
sub_posthocs["A_B"] = sub_posthocs.A.values + "_" + sub_posthocs.B.values

# set(sub_posthocs["A_B"])
# ['FAM_HC-fam', 'FAM_QSM', ]

# disgregate rois
sub_posthocs["roi1"] = [conn.split(" || ")[0] for conn in sub_posthocs["conn"].values]
sub_posthocs["roi1"] = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in sub_posthocs["roi1"].values]

sub_posthocs["roi2"] = [conn.split(" || ")[1] for conn in sub_posthocs["conn"].values]
sub_posthocs["roi2"] = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in sub_posthocs["roi2"].values]

# identify roi ids
sub_posthocs["roi1_sc-id"] = [list(conn.region_labels).index(roi) for roi in sub_posthocs["roi1"].values]
sub_posthocs["roi2_sc-id"] = [list(conn.region_labels).index(roi) for roi in sub_posthocs["roi2"].values]

# pre-define SIZE by node strength
weights = conn.scaled_weights(mode="tract")
degree = np.sum(weights, axis=1)
range_size = [3, 8]
size = degree * (range_size[1]-range_size[0]) + (range_size[0] - np.min(degree))

# TODO pre-define COLOUR by PSD diff?

# pre-define LINE WIDTH by FC diff
range_width = [2, 40]
sub_posthocs["line_width"] = abs(sub_posthocs["diff"].values) * (range_width[1]-range_width[0]) + (range_width[0] - np.min(abs(sub_posthocs["diff"].values)))


sp_cols = [group+"->"+groups[i+1] for i, group in enumerate(groups[:-1])]
n_rows = 1

specs = [[{"type": "scene"}] * len(sp_cols)] * n_rows

fig = make_subplots(rows=n_rows, cols=len(sp_cols), horizontal_spacing=0,
                    subplot_titles=sp_cols, specs=specs, shared_xaxes=True, shared_yaxes=True)

for j, sp in enumerate(sp_cols):
    group1, group2 = sp.split("->")[0], sp.split("->")[1]
    sub_sig = sub_posthocs.loc[
        (sub_posthocs["A"].isin([group1, group2])) & (sub_posthocs["B"].isin([group1, group2]))]

    if not sub_sig.empty:
        if sub_sig["A"].values[0] != group2:
            sub_sig = sub_sig.copy()
            sub_sig["diff"] = -sub_sig["diff"]

        for _, row in sub_sig.iterrows():
            roi1_id = list(conn.region_labels).index(row["roi1"])
            roi2_id = list(conn.region_labels).index(row["roi2"])

            fig.add_trace(go.Scatter3d(x=[conn.centres[roi1_id, 0], conn.centres[roi2_id, 0]],
                                       y=[conn.centres[roi1_id, 1], conn.centres[roi2_id, 1]],
                                       z=[conn.centres[roi1_id, 2], conn.centres[roi2_id, 2]],
                                       name=row["conn"], legendgroup=row["conn"], mode="markers+lines", showlegend=False,
                                       marker=dict(size=size[[roi1_id, roi2_id]]),#, color=node_colours[[roi1_id, roi2_id]]),
                                       line=dict(width=row["line_width"], color=row["diff"], colorscale="Cividis")
                                       ), row=1, col=j + 1)

    # Add 3d brains
    fig.add_trace(go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
                            color='silver', opacity=0.2, showscale=False, hoverinfo='none'), row=1, col=j+1)

xe, ye, ze = -1.25, 2, 0.5
fig.update_layout(
            scene1=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))),
            scene2=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))),
            scene3=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))),
            scene4=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))
fig.update_layout(
         title='Significant changes in FC: C3N dataset', template="plotly_white", )

print("Saving images: may take a while")
# pio.write_image(fig, file='figures/ADpg_PETchanges.svg', height=800, width=1100)
pio.write_html(fig, file='figures/ADpg_FC-SIGNIFICANTchanges.html', auto_open=True)



#### 2.b. C3N (FC) - One row - TOP10    #################
"""
Was not finished.
"""

anova_fc = pd.read_csv(data_folder + "FCavg_matrices/.FC_anovas_perConn.csv", index_col=0)
posthoc = pd.read_csv(data_folder + "FCavg_matrices/.FC_posthocs_perConn.csv", index_col=0)

groups = ["HC-fam", "FAM", "QSM", "MCI", "MCI-conv"]

# As SIGNIFICANT in mode: filter anovas and then posthocs
sub_posthocs = posthoc.copy()

# disgregate rois
sub_posthocs["roi1"] = [conn.split(" || ")[0] for conn in sub_posthocs["conn"].values]
sub_posthocs["roi1"] = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in sub_posthocs["roi1"].values]

sub_posthocs["roi2"] = [conn.split(" || ")[1] for conn in sub_posthocs["conn"].values]
sub_posthocs["roi2"] = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in sub_posthocs["roi2"].values]

# identify roi ids
sub_posthocs["roi1_sc-id"] = [list(conn.region_labels).index(roi) for roi in sub_posthocs["roi1"].values]
sub_posthocs["roi2_sc-id"] = [list(conn.region_labels).index(roi) for roi in sub_posthocs["roi2"].values]

# pre-define SIZE by node strength
weights = conn.scaled_weights(mode="tract")
degree = np.sum(weights, axis=1)
range_size = [5, 15]
size = degree * (range_size[1]-range_size[0]) + (range_size[0] - np.min(degree))

#TODO pre-define COLOUR by PSD diff?

# TODO pre-define connection


sp_cols = [group+"->"+groups[i+1] for i, group in enumerate(groups[:-1])]
n_rows = 1

specs = [[{"type": "scene"}] * len(sp_cols)] * n_rows

fig = make_subplots(rows=n_rows, cols=len(sp_cols), horizontal_spacing=0,
                    subplot_titles=sp_cols, specs=specs, shared_xaxes=True, shared_yaxes=True)

for j, sp in enumerate(sp_cols):
    group1, group2 = sp.split("->")[0], sp.split("->")[1]
    sub_sig = sub_posthocs.loc[
        (sub_posthocs["A"].isin([group1, group2])) & (sub_posthocs["B"].isin([group1, group2]))]

    if not sub_sig.empty:
        if sub_sig["A"].values[0] != group2:
            sub_sig = sub_sig.copy()
            sub_sig["diff"] = -sub_sig["diff"]

        sub_sig = sub_sig.sort_values(by=["diff"], ascending=False).head(10)

        # pre-define LINE WIDTH by FC diff
        range_width = [1, 40]
        sub_sig["line_width"] = abs(sub_sig["diff"].values) * (range_width[1] - range_width[0]) + (
                    range_width[0] - np.min(abs(sub_sig["diff"].values)))

        for _, row in sub_sig.iterrows():
            roi1_id = list(conn.region_labels).index(row["roi1"])
            roi2_id = list(conn.region_labels).index(row["roi2"])

            fig.add_trace(go.Scatter3d(x=[conn.centres[roi1_id, 0], conn.centres[roi2_id, 0]],
                                       y=[conn.centres[roi1_id, 1], conn.centres[roi2_id, 1]],
                                       z=[conn.centres[roi1_id, 2], conn.centres[roi2_id, 2]],
                                       name=row["conn"], legendgroup=row["conn"], mode="markers+lines", showlegend=False,
                                       marker=dict(size=size[[roi1_id, roi2_id]]),#, color=node_colours[[roi1_id, roi2_id]]),
                                       line=dict(width=row["line_width"], color=row["diff"], colorscale="Cividis")
                                       ), row=1, col=j + 1)

    # Add 3d brains
    fig.add_trace(go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
                            color='silver', opacity=0.2, showscale=False, hoverinfo='none'), row=1, col=j+1)

xe, ye, ze = -1.25, 2, 0.5
fig.update_layout(title='Significant changes in FC: C3N dataset', template="plotly_white",
            scene1=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))),
            scene2=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))),
            scene3=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))),
            scene4=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))

print("Saving images: may take a while")
# pio.write_image(fig, file='figures/ADpg_PETchanges.svg', height=800, width=1100)
pio.write_html(fig, file='figures/ADpg_FC-TOP10changes.html', auto_open=True)





#### 2.c. C3N (FC) - Multiple rows (bands) - TOP10
bands = ["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"]

