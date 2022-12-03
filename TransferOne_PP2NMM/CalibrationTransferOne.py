
import time
import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from tvb.simulator.lab import connectivity
from ADpg_functions import ProteinSpreadModel
from Calibration_functions import simulations, TransferOne, propagationtrajectory_on4D

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTpeaks, FFTplot
    from toolbox.signals import epochingTool, timeseriesPlot
    from toolbox.fc import PLV
    from toolbox.mixes import timeseries_spectra

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"



#  0. STRUCTURAL CONNECTIVITY   #########
#  Define structure through which the proteins will spread;
#  Not necessarily the same than the one used to simulate activity.
subj = "HC-fam"
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")

#    ADNI PET DATA       ##########
ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)

# Check label order
PETlabs = list(ADNI_AVG.columns[12:])
PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]



mode = "classic_WORKINGfit_short"
print(mode)

#  1. SETUP and SIMULATE protein dynamics  #############
# Following Alexandersen (2022)  - same parameters, same initial conditions
"""
Compute Christoffer protein propagation.
TUNE NMM parameters range -
"""
#  REGIONAL SEEDs for toxic proteins
AB_seeds = ["ctx-lh-precuneus", "ctx-lh-isthmuscingulate", "ctx-lh-insula", "ctx-lh-medialorbitofrontal", "ctx-lh-lateralorbitofrontal",
            "ctx-rh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-insula", "ctx-rh-medialorbitofrontal", "ctx-rh-lateralorbitofrontal"]
TAU_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]

AB_initMap, TAU_initMap = [[1 for roi in conn.region_labels]]*2
ABt_initMap = [0.1 / len(AB_seeds) if roi in AB_seeds else 0 for roi in conn.region_labels]
TAUt_initMap = [0.1 / len(TAU_seeds) if roi in TAU_seeds else 0 for roi in conn.region_labels]

AB_initdam, TAU_initdam = [[0 for roi in conn.region_labels]]*2

## TUNNING
He, Hi, taue, taui = 3.5, 22, 10, 16

protpropmodel = ProteinSpreadModel(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam, TAU_initdam,
    init_He=He, init_Hi=Hi, init_taue=taue, init_taui=taui, rho=0.001, toxicSynergy=12,
    prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
    prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66)

protpropmodel.init_He["range"] = [He-1, He+2]  # origins (2.6, 9.75) :: (-0.65+x, x+6.5)
protpropmodel.init_Hi["range"] = [Hi-3, Hi+18]  # origins (17.6, 40) :: (-4.4+x, x+18)
protpropmodel.init_taue["range"] = [taue-4, taue+10]  # origins (6, 12) :: (-4+x, x+10)
protpropmodel.init_taui["range"] = [taui-8, taui+20]  # origins (12, 40) :: (-8+x, x+20)

out_prop = protpropmodel.run(time=40, dt=0.25, sim=False)
timepoints = out_prop[0]


## 3. Simulations with protein timepoints SINGLE NODE  / TESTing TRANSFER FUNCTION _protprop->NMM   #####
# out_avg = np.average(out_prop[1], axis=2).transpose()[6:]
# conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
#
# out_simpleSim, t0, skip = [], time.time(), 10
# for i, t in enumerate(timepoints):
#     if i % skip == 0:
#         print("SIMPLE SIMULATIONs  t%0.2f of propagation dyn  -  %i/%i     | time: %0.2fs" %
#               (t, i/skip, len(timepoints)/skip, time.time()-t0), end="\r")
#         out_simpleSim.append(simulations(out_avg[:, i], conn, mode=mode, rois="pair"))
# print("SIMPLE SIMULATIONs  t%0.2f of propagation dyn  -  %i/%i     | time: %0.2fm" % (t, i/skip, len(timepoints)/skip, (time.time()-t0)/60))
#
# # Plot results
# TransferOne(out_prop, out_simpleSim, skip=skip, mode=mode)


##  4. DEEPEN into specific points of the temporal domain
# t = 18
# id = list(out_prop[0]).index(t)
# raw_data = simulations(out_avg[:, id], conn, out="signals", mode=mode, rois="pair")
#
# timeseries_spectra(raw_data, 5000, 1000, ["cx", "th"], mode="html", folder="figures",
#                        freqRange=[2, 40], opacity=1, title=None, auto_open=True)

print("here")
##  5. TEST whole network TRANSFER FUNCTION _protprop->NMM    #####
out_avg = np.moveaxis(np.array(out_prop[1]), 0, 1)[6:]
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")

out_netSim, t0, skip = [], time.time(), 10
for i, t in enumerate(timepoints):
    if i % skip == 0:
        print("BNM SIMULATIONs  t%0.2f of propagation dyn  -  %i/%i     | time: %0.2fs" %
              (t, i / skip, len(timepoints) / skip, time.time() - t0), end="\r")
        out_netSim.append(simulations(out_avg[:, i, :], conn, mode=mode, rois="bnm"))
print("BNM SIMULATIONs  t%0.2f of propagation dyn  -  %i/%i     | time: %0.2fm" % (
t, i / skip, len(timepoints) / skip, (time.time() - t0) / 60))
# Plot results
TransferOne(out_prop, out_networkSim=out_netSim, skip=skip, mode=mode)

##  4. DEEPEN into specific points of the temporal domain
# t = 25
# id = list(out_prop[0]).index(t)
# raw_data = simulations(out_avg[:, id], conn, out="signals", mode=mode, rois="bnm")
#
# timeseries_spectra(raw_data, 5000, 1000, conn.region_labels, mode="html", folder="figures",
#                        freqRange=[2, 40], opacity=1, title=None, auto_open=True)


## 6. Propagation trajectory on a 4D parameter space animation
propagationtrajectory_on4D(out_prop, mode, PSE3d_tag="PSEmpi_ADpg_PSE3d-m11d10y2022-t17h.09m.44s")














#
#
#
#
#
#
# #   STRUCTURAL CONNECTIVITY   #########
# #  Define structure through which the proteins will spread;
# #  Not necessarily the same than the one used to simulate activity.
# subj = "HC-fam"
# conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
#
# conn.weights = conn.weights[:2][:, :2]
# conn.tract_lengths = conn.tract_lengths[:2][:, :2]
# conn.region_labels = conn.region_labels[:2]
#
# conn.weights = conn.scaled_weights(mode="tract")  # did they normalize? maybe this affects to the spreading?
# conn.speed = np.array([15])
#
# # Prepare simulation parameters
# model, g, g_wc = "jr", 3, None
# simLength = 10 * 1000  # ms
# samplingFreq = 1000  # Hz
# transient = 2000  # ms
#
#
#
# tic = time.time()
#
# #   NEURAL MASS MODEL  &  COUPLING FUNCTION   #########################################################
# sigma_array = np.asarray([0 if 'Thal' in roi else 0 for roi in conn.region_labels])
# p_array = np.asarray([0.22 if 'Thal' in roi else 0.09 for roi in conn.region_labels])
#
# if model == "jrd":  # JANSEN-RIT-DAVID
#     # Parameters edited from David and Friston (2003).
#     m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
#                            tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
#                            He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
#                            tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),
#
#                            w=np.array([0.8]), c=np.array([135.0]),
#                            c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
#                            c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
#                            v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
#                            p=np.array([p_array]), sigma=np.array([sigma_array]))
#
#     # Remember to hold tau*H constant.
#     m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
#     m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])
#
#     coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=np.array([0.8]), e0=np.array([0.005]),
#                                             v0=np.array([6.0]), r=np.array([0.56]))
#
# elif model == "jrwc":  # JANSEN-RIT(cx) + WILSON-COWAN(th)
#
#     jrMask_wc = [[False] if 'Thal' in roi else [True] for roi in conn.region_labels]
#
#     m = JansenRit_WilsonCowan(
#         # Jansen-Rit nodes parameters. From Stefanovski et al. (2019)
#         He=np.array([3.25]), Hi=np.array([22]),
#         tau_e=np.array([10]), tau_i=np.array([20]),
#         c=np.array([135.0]), p=np.array([p_array]),
#         c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
#         c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
#         v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
#         # Wilson-Cowan nodes parameters. From Abeysuriya et al. (2018)
#         P=np.array([0.31]), sigma=np.array([sigma_array]), Q=np.array([0]),
#         c_ee=np.array([3.25]), c_ei=np.array([2.5]),
#         c_ie=np.array([3.75]), c_ii=np.array([0]),
#         tau_e_wc=np.array([10]), tau_i_wc=np.array([20]),
#         a_e=np.array([4]), a_i=np.array([4]),
#         b_e=np.array([1]), b_i=np.array([1]),
#         c_e=np.array([1]), c_i=np.array([1]),
#         k_e=np.array([1]), k_i=np.array([1]),
#         r_e=np.array([0]), r_i=np.array([0]),
#         theta_e=np.array([0]), theta_i=np.array([0]),
#         alpha_e=np.array([1]), alpha_i=np.array([1]),
#         # JR mask | WC mask
#         jrMask_wc=np.asarray(jrMask_wc))
#
#     m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])
#
#     coup = coupling.SigmoidalJansenRit_Linear(
#         a=np.array([g]), e0=np.array([0.005]), v0=np.array([6]), r=np.array([0.56]),
#         # Jansen-Rit Sigmoidal coupling
#         a_linear=np.asarray([g_wc]),  # Wilson-Cowan Linear coupling
#         jrMask_wc=np.asarray(jrMask_wc))  # JR mask | WC mask
#
# else:  # JANSEN-RIT
#     # Parameters from Stefanovski 2019.
#     m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
#                       tau_e=np.array([10]), tau_i=np.array([20]),
#                       c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
#                       c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
#                       p=np.array([0.1085]), sigma=np.array([0]),
#                       e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))
#
#     coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
#                                        r=np.array([0.56]))
#
# # OTHER PARAMETERS   ###
# # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
# integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)
#
# mon = (monitors.Raw(),)
#
# # print("Simulating %s (%is)  ||  PARAMS: g%i sigma%0.2f" % (model, simLength / 1000, g, 0.022))
#
# # Run simulation
# sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
# sim.configure()
# output = sim.run(simulation_length=simLength)
#
# # Extract data: "output[a][b][:,0,:,0].T" where:
# # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
# if model == "jrd":
#     raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
#                (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
# else:
#     raw_data = output[0][1][transient:, 0, :, 0].T
# raw_time = output[0][0][transient:]
#
# # PLOTs :: Signals and spectra
# timeseries_spectra(raw_data[:], simLength, transient, conn.region_labels, mode="html", freqRange=[2, 40], opacity=1)
#
# print("SIMULATION REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))
#
#
#
#
#
