import time
import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats
from mne import filter
import statsmodels.api as sm
from itertools import combinations

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from tvb.simulator.models.JansenRit_WilsonCowan import JansenRit_WilsonCowan

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys

    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTpeaks
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order

## Folder structure - CLUSTER
else:
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order

    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"


class ProteinSpreadModel:
    # Spread model variables following Alexandersen 2022.
    # M is an arbitrary unit of concentration

    def __init__(self, initConn, AB_initMap, TAU_initMAp, ABt_initMap, TAUt_initMap, AB_initdam, TAU_initdam,
                 init_He, init_Hi, init_taue, init_taui, rho=0.001, toxicSynergy=2,
                 prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
                 prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66,
                 AB_damrate=1, TAU_damrate=1, TAU_dam2SC=0.2,
                 cABexc=0.8, cABinh=0.4, cTAU=1.8, th_exc=False):

        self.rho = {"label": "rho", "value": np.array([rho]), "doc": "effective diffusion constant (cm/year)"}

        self.prodAB = {"label": ["k0", "a0"], "value": np.array([prodAB]), "doc": "production rate for a-beta (M/year)"}
        self.clearAB = {"label": ["k1", "a1"], "value": np.array([clearAB]),
                        "doc": "clearance rate for a-beta (1/M*year)"}
        self.transAB2t = {"label": ["k2", "a2"], "value": np.array([transAB2t]),
                          "doc": "transformation of a-beta into its toxic variant (M/year)"}
        self.clearABt = {"label": ["k1t", "a1t"], "value": np.array([clearABt]),
                         "doc": "clearance rate for toxic a-beta (1/M*year)"}

        self.prodTAU = {"label": ["k3", "b0"], "value": np.array([prodTAU]),
                        "doc": "production rate for p-tau (M/year)"}
        self.clearTAU = {"label": ["k4", "b1"], "value": np.array([clearTAU]),
                         "doc": "clearance rate for p-tau (1/M*year)"}
        self.transTAU2t = {"label": ["k5", "b2"], "value": np.array([transTAU2t]),
                           "doc": "transformation of p-tau into its toxic variant (M/year)"} \
            if len(np.array([transTAU2t]).shape) == 1 else \
            {"label": ["k5", "b2"], "value": np.array([transTAU2t]).squeeze(),
             "doc": "transformation of p-tau into its toxic variant (M/year)"}

        self.clearTAUt = {"label": ["k4t", "b1t"], "value": np.array([clearTAUt]),
                          "doc": "clearance rate for toxic p-tau (1/M*year)"}

        self.toxicSynergy = {"label": ["k6", "b3"], "value": np.array([toxicSynergy]),
                             "doc": "synergistic effect between toxic a-beta and toxic p-tau production (1/M^2*year)"} \
            if len(np.array([toxicSynergy]).shape) == 1 else \
            {"label": ["k5", "b2"], "value": np.array([toxicSynergy]).squeeze(),
             "doc": "transformation of p-tau into its toxic variant (M/year)"}

        self.AB_initMap = {"label": "", "value": AB_initMap, "doc": "mapping of initial roi concentration of AB"}
        self.TAU_initMap = {"label": "", "value": TAU_initMAp, "doc": "mapping of initial roi concentration of TAU"}

        self.ABt_initMap = {"label": "", "value": ABt_initMap,
                            "doc": "mapping of initial roi concentration of AB toxic"}
        self.TAUt_initMap = {"label": "", "value": TAUt_initMap,
                             "doc": "mapping of initial roi concentration of TAU toxic"}

        AB_initdam = AB_initdam if type(AB_initdam) == list else [AB_initdam for roi in initConn.region_labels]
        self.AB_initdam = {"label": "q(AB)", "value": AB_initdam, "doc": "initial damage/impact variable of AB"}

        TAU_initdam = TAU_initdam if type(TAU_initdam) == list else [TAU_initdam for roi in initConn.region_labels]
        self.TAU_initdam = {"label": "q(TAU)", "value": TAU_initdam,
                            "doc": "initial damage/impact of hyperphosphorilated TAU"}

        self.AB_damrate = {"label": "k(AB)", "value": np.array([AB_damrate]),
                           "doc": "rate of damage/impact for AB (M/year)"}
        self.TAU_damrate = {"label": "K(TAU)", "value": np.array([TAU_damrate]),
                            "doc": "rate of damage/impact for hyperphosphorilated TAU (M/year)"}
        self.TAU_dam2SC = {"label": "gamma", "value": np.array([TAU_dam2SC]),
                           "doc": "constant for the damage of structural connectivity by hpTAU (cm/year)"}

        self.initConn = {"label": "SC", "value": initConn, "doc": "Initial state for structural connectivity"}

        init_He = init_He if type(init_He) == list else [init_He for roi in initConn.region_labels]
        self.init_He = {"label": "He", "value": init_He, "range": [2.6, 9.75],
                        "doc": "Initial state for excitation. def3.25"}
        init_Hi = init_Hi if type(init_Hi) == list else [init_Hi for roi in initConn.region_labels]
        self.init_Hi = {"label": "Hi", "value": init_Hi, "range": [17.6, 40],
                        "doc": "Initial state for inhibition. def22"}
        init_taue = init_taue if type(init_taue) == list else [init_taue for roi in initConn.region_labels]
        self.init_taue = {"label": "tau_e", "value": init_taue, "range": [6, 20],
                          "doc": "Initial state for delays (exc). def10"}
        init_taui = init_taui if type(init_taui) == list else [init_taui for roi in initConn.region_labels]
        self.init_taui = {"label": "tau_e", "value": init_taui, "range": [12, 40],
                          "doc": "Initial state for delays (inh). def16"}

        self.cABexc = {"label": "c_beta", "value": np.array([cABexc]),
                       "doc": "constant for the effect of AB on excitation"}
        self.cABinh = {"label": "c_beta2", "value": np.array([cABinh]),
                       "doc": "constant for the effect of AB on inhibition"}
        self.cTAU = {"label": "c_tau", "value": np.array([cTAU]), "doc": "constant for the effect of pTau on delays"}

        self.th_exc = {"value": th_exc, "doc": "Decide whether updating thalamus values."}

    def run(self, time, dt, sim=False, sim_dt=1):

        ## 1. Initiate state variables
        state_variables = np.asarray([self.AB_initMap["value"],
                                      self.ABt_initMap["value"],
                                      self.TAU_initMap["value"],
                                      self.TAUt_initMap["value"],

                                      self.AB_initdam["value"],
                                      self.TAU_initdam["value"],

                                      self.init_He["value"],
                                      self.init_Hi["value"],
                                      self.init_taue["value"],
                                      self.init_taui["value"]])

        weights = self.initConn["value"].weights

        evolution_sv = [state_variables.copy()]

        print("Simulating protein spread  . for %0.2fts (dt=%0.2f)   _simulate: %s" % (time, dt, sim))

        if (type(sim_dt) == int) | (type(sim_dt) == float):
            tsel = np.arange(0, time, sim_dt)
        else:
            tsel = sim_dt

        if (sim) and (0 in tsel):
            subj, model, g, s, simLength = sim
            raw_data, raw_time, fftp, plv, plv_emp, plv_r, regionLabels, _, transient, reqtime \
                = simulate_v2(subj, weights, model, g, s, p_th=0.1085, sigma=0, sv=state_variables[6:],
                              t=simLength)
            evolution_net = [
                [weights, raw_data, raw_time, fftp, plv, plv_emp, plv_r, regionLabels, simLength, transient]]
            print("   . ts%0.2f/%0.2f  _  SIMULATION REQUIRED %0.2f seconds  -  rPLV(%0.2f)" % (
                0, time, reqtime, plv_r))

        else:
            evolution_net = [[weights]]
            print("   . ts%0.2f/%0.2f" % (0, time), end="\r")

        ## 2. loop over time
        for t in np.arange(dt, time, dt):
            deriv = self.dfun(state_variables, self.Laplacian(weights))

            state_variables = state_variables + dt * deriv

            # if type(self.th_exc["value"]) == list:
            #     mask = self.th_exc["value"]
            #     state_variables[:, mask] = state_variables[:, mask] + dt * deriv[:, mask]
            # else:

            ## Update weights by damage function
            TAUdam = state_variables[5]
            dWeights = -self.TAU_dam2SC["value"] * (
                    np.tile(TAUdam, (len(TAUdam), 1)).transpose() + np.tile(TAUdam, (len(TAUdam), 1)))

            # TODO weights cannot be less than 0
            weights = weights + dt * dWeights
            weights[weights < 0] = 0

            if sim and (t in tsel):
                subj, model, g, s, simLength = sim
                raw_data, raw_time, fftp, plv, plv_emp, plv_r, regionLabels, _, transient, reqtime \
                    = simulate_v2(subj, weights, model, g, s, p_th=0.1085, sigma=0, sv=state_variables[6:], t=simLength)

                evolution_net.append(
                    [weights, raw_data, raw_time, fftp, plv, plv_emp, plv_r, regionLabels, simLength, transient])
                evolution_sv.append(state_variables)
                print("   . ts%0.2f/%0.2f  _  SIMULATION REQUIRED %0.2f seconds  -  rPLV(%0.2f)" % (
                    t, time, reqtime, plv_r))

            else:
                evolution_sv.append(state_variables.copy())
                evolution_net.append([weights])
                print("   . ts%0.2f/%0.2f" % (t, time), end="\r")

        return [np.arange(0, time, dt), evolution_sv, evolution_net]

    def Laplacian(self, weights):
        # Weighted adjacency, Diagonal and Laplacian matrices
        Wij = np.divide(weights, np.square(self.initConn["value"].tract_lengths),
                        where=np.square(self.initConn["value"].tract_lengths) != 0,
                        # Where to compute division; else out
                        out=np.zeros_like(weights))  # array allocation
        Dii = np.eye(len(Wij)) * np.sum(Wij, axis=0)
        Lij = Dii - Wij

        return Lij

    def dfun(self, state_variables, Lij):
        # Here we want to model the spread of proteinopathies.
        # Approach without activity dependent spread/generation. Following Alexandersen 2022.

        AB = state_variables[0]
        ABt = state_variables[1]
        TAU = state_variables[2]
        TAUt = state_variables[3]

        ABdam = state_variables[4]
        TAUdam = state_variables[5]

        He_ = state_variables[6]
        Hi_ = state_variables[7]
        taue_ = state_variables[8]
        taui_ = state_variables[9]

        # Unpack heterogeneous rho
        [rho_AB, rho_ABt, rho_TAU, rho_TAUt] = self.rho["value"][0] \
            if len(self.rho["value"].shape) == 2 else self.rho["value"].repeat(4)

        # Derivatives
        ###  Amyloid-beta
        dAB = -rho_AB * np.sum(Lij * AB, axis=1) + self.prodAB["value"] - self.clearAB["value"] * AB - \
              self.transAB2t["value"] * AB * ABt
        dABt = -rho_ABt * np.sum(Lij * ABt, axis=1) - self.clearABt["value"] * ABt + self.transAB2t[
            "value"] * AB * ABt

        ###  (hyperphosphorilated) Tau
        dTAU = -rho_TAU * np.sum(Lij * TAU, axis=1) + self.prodTAU["value"] - self.clearTAU["value"] * TAU - \
               self.transTAU2t["value"] * TAU * TAUt - self.toxicSynergy["value"] * ABt * TAU * TAUt
        dTAUt = -rho_TAUt * np.sum(Lij * TAUt, axis=1) - self.clearTAUt["value"] * TAUt + self.transTAU2t[
            "value"] * TAU * TAUt + self.toxicSynergy["value"] * ABt * TAU * TAUt

        dABdam = self.AB_damrate["value"] * ABt * (1 - ABdam)
        dTAUdam = self.TAU_damrate["value"] * TAUt * (1 - TAUdam)

        ## ACTIVATION transfers: a(exc), b(inhi)
        dHe = (self.cABexc["value"] * ABdam * ((self.init_He["range"][1] - He_) - self.cTAU["value"] * TAUdam) * (
                    He_ - self.init_He["range"][0]))
        dHi = - self.cABinh["value"] * ABdam * (Hi_ - self.init_Hi["range"][0])

        ## FREQUENCY transfers: c(delays)
        dtaue = self.cTAU["value"] * TAUdam * (self.init_taue["range"][1] - taue_)
        dtaui = taui_ - taui_  # By now, taui does not change

        derivative = np.array([dAB, dABt, dTAU, dTAUt, dABdam, dTAUdam, dHe, dHi, dtaue, dtaui])

        return derivative


class CircularADpgModel:
    # Spread model variables following Alexandersen 2022.
    # M is an arbitrary unit of concentration

    def __init__(self, initConn, AB_initMap, TAU_initMAp, ABt_initMap, TAUt_initMap,
                 AB_initdam, TAU_initdam, POW_initdam,
                 init_He, init_Hi, init_taue, init_taui, rho=0.001, toxicSynergy=2,
                 prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
                 prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66,
                 AB_damrate=1, TAU_damrate=1, TAU_dam2SC=0.2, POW_damrate=1, maxPOWdam=2,
                 cABexc=0.8, cABinh=0.4, cTAU=1.8):

        self.rho = {"label": "rho", "value": np.array([rho]), "doc": "effective diffusion constant (cm/year)"}

        self.prodAB = {"label": ["k0", "a0"], "value": np.array([prodAB]), "doc": "production rate for a-beta (M/year)"}
        self.clearAB = {"label": ["k1", "a1"], "value": np.array([clearAB]),
                        "doc": "clearance rate for a-beta (1/M*year)"}
        self.transAB2t = {"label": ["k2", "a2"], "value": np.array([transAB2t]),
                          "doc": "transformation of a-beta into its toxic variant (M/year)"}
        self.clearABt = {"label": ["k1t", "a1t"], "value": np.array([clearABt]),
                         "doc": "clearance rate for toxic a-beta (1/M*year)"}

        self.prodTAU = {"label": ["k3", "b0"], "value": np.array([prodTAU]),
                        "doc": "production rate for p-tau (M/year)"}
        self.clearTAU = {"label": ["k4", "b1"], "value": np.array([clearTAU]),
                         "doc": "clearance rate for p-tau (1/M*year)"}
        self.transTAU2t = {"label": ["k5", "b2"], "value": np.array([transTAU2t]),
                           "doc": "transformation of p-tau into its toxic variant (M/year)"} \
            if len(np.array([transTAU2t]).shape) == 1 else \
            {"label": ["k5", "b2"], "value": np.array([transTAU2t]).squeeze(),
             "doc": "transformation of p-tau into its toxic variant (M/year)"}

        self.clearTAUt = {"label": ["k4t", "b1t"], "value": np.array([clearTAUt]),
                          "doc": "clearance rate for toxic p-tau (1/M*year)"}

        self.toxicSynergy = {"label": ["k6", "b3"], "value": np.array([toxicSynergy]),
                             "doc": "synergistic effect between toxic a-beta and toxic p-tau production (1/M^2*year)"} \
            if len(np.array([toxicSynergy]).shape) == 1 else \
            {"label": ["k5", "b2"], "value": np.array([toxicSynergy]).squeeze(),
             "doc": "transformation of p-tau into its toxic variant (M/year)"}

        self.AB_initMap = {"label": "", "value": AB_initMap, "doc": "mapping of initial roi concentration of AB"}
        self.TAU_initMap = {"label": "", "value": TAU_initMAp, "doc": "mapping of initial roi concentration of TAU"}

        self.ABt_initMap = {"label": "", "value": ABt_initMap,
                            "doc": "mapping of initial roi concentration of AB toxic"}
        self.TAUt_initMap = {"label": "", "value": TAUt_initMap,
                             "doc": "mapping of initial roi concentration of TAU toxic"}

        AB_initdam = AB_initdam if type(AB_initdam) == list else [AB_initdam for roi in initConn.region_labels]
        self.AB_initdam = {"label": "q(AB)", "value": AB_initdam, "doc": "initial damage/impact variable of AB"}
        TAU_initdam = TAU_initdam if type(TAU_initdam) == list else [TAU_initdam for roi in initConn.region_labels]
        self.TAU_initdam = {"label": "q(TAU)", "value": TAU_initdam,
                            "doc": "initial damage/impact of hyperphosphorilated TAU"}
        POW_initdam = POW_initdam if type(POW_initdam) == list else [POW_initdam for roi in initConn.region_labels]
        self.POW_initdam = {"label": "q(POW)", "value": POW_initdam, "doc": "initial damage/impact variable of POWER"}
        self.maxPOWdam = {"label": "q(POW)", "value": np.array([maxPOWdam]),
                          "doc": "max damage/impact variable of POWER"}

        self.AB_damrate = {"label": "k(AB)", "value": np.array([AB_damrate]),
                           "doc": "rate of damage/impact for AB (M/year)"}
        self.TAU_damrate = {"label": "K(TAU)", "value": np.array([TAU_damrate]),
                            "doc": "rate of damage/impact for hyperphosphorilated TAU (M/year)"}
        self.TAU_dam2SC = {"label": "gamma", "value": np.array([TAU_dam2SC]),
                           "doc": "constant for the damage of structural connectivity by hpTAU (cm/year)"}
        self.POW_damrate = {"label": "", "value": np.array([POW_damrate]),
                            "doc": "rate of damage/impact for AB (M/year)"}

        self.initConn = {"label": "SC", "value": initConn, "doc": "Initial state for structural connectivity"}

        init_He = init_He if type(init_He) == list else [init_He for roi in initConn.region_labels]
        self.init_He = {"label": "He", "value": init_He, "range": [2.6, 9.75],
                        "doc": "Initial state for excitation. def3.25"}
        init_Hi = init_Hi if type(init_Hi) == list else [init_Hi for roi in initConn.region_labels]
        self.init_Hi = {"label": "Hi", "value": init_Hi, "range": [17.6, 40],
                        "doc": "Initial state for inhibition. def22"}
        init_taue = init_taue if type(init_taue) == list else [init_taue for roi in initConn.region_labels]
        self.init_taue = {"label": "tau_e", "value": init_taue, "range": [6, 20],
                          "doc": "Initial state for delays (exc). def10"}
        init_taui = init_taui if type(init_taui) == list else [init_taui for roi in initConn.region_labels]
        self.init_taui = {"label": "tau_e", "value": init_taui, "range": [12, 40],
                          "doc": "Initial state for delays (inh). def16"}

        self.cABexc = {"label": "c_beta", "value": np.array([cABexc]),
                       "doc": "constant for the effect of AB on excitation"}
        self.cABinh = {"label": "c_beta2", "value": np.array([cABinh]),
                       "doc": "constant for the effect of AB on inhibition"}
        self.cTAU = {"label": "c_tau", "value": np.array([cTAU]), "doc": "constant for the effect of pTau on delays"}

    def run(self, time, dt, sim=False, sim_dt=1):

        ## 1. Initiate state variables
        state_variables = np.asarray([self.AB_initMap["value"],
                                      self.ABt_initMap["value"],
                                      self.TAU_initMap["value"],
                                      self.TAUt_initMap["value"],

                                      self.AB_initdam["value"],
                                      self.TAU_initdam["value"],

                                      self.init_He["value"],
                                      self.init_Hi["value"],
                                      self.init_taue["value"],
                                      self.init_taui["value"],

                                      self.POW_initdam["value"]])

        weights = self.initConn["value"].weights

        evolution_sv = [state_variables.copy()]

        print("Simulating protein spread  . for %0.2fts (dt=%0.2f)   _simulate: %s" % (time, dt, sim))

        if (type(sim_dt) == int) | (type(sim_dt) == float):
            tsel = np.arange(0, time, sim_dt)
        else:
            tsel = sim_dt

        if (sim) and (0 in tsel):
            subj, model, g, s, simLength = sim
            raw_data, _, fftp, plv, plv_emp, plv_r, regionLabels, _, _, reqtime \
                = simulate_v2(subj, weights, model, g, s, sv=state_variables[6:-1], t=simLength)

            baseline_fftp = fftp[1]
            evolution_net = [[weights, fftp[0], fftp[1], plv, plv_emp, plv_r, regionLabels, raw_data]]
            print("   . ts%0.2f/%0.2f  _  SIMULATION REQUIRED %0.2f seconds  -  rPLV(%0.2f)" % (
                0, time, reqtime, plv_r))


        else:
            evolution_net = [[weights]]
            print("   . ts%0.2f/%0.2f" % (0, time), end="\r")

        ## 2. loop over time
        for t in np.arange(dt, time, dt):

            # POW_effect
            delta_fftp = fftp[1] / baseline_fftp

            deriv = self.dfun(state_variables, self.Laplacian(weights), delta_fftp)

            state_variables = state_variables + dt * deriv

            ## Update weights by damage function
            TAUdam = state_variables[5]
            dWeights = -self.TAU_dam2SC["value"] * (
                    np.tile(TAUdam, (len(TAUdam), 1)).transpose() + np.tile(TAUdam, (len(TAUdam), 1)))

            weights = weights + dt * dWeights
            weights[weights < 0] = 0  # weights cannot be negative

            if sim and (t in tsel):
                subj, model, g, s, simLength = sim
                raw_data, _, fftp, plv, plv_emp, plv_r, regionLabels, _, _, reqtime \
                    = simulate_v2(subj, weights, model, g, s, sv=state_variables[6:-1], t=simLength)

                evolution_net.append(
                    [weights, fftp[0], fftp[1], plv, plv_emp, plv_r, regionLabels, raw_data])
                evolution_sv.append(state_variables)
                print("   . ts%0.2f/%0.2f  _  SIMULATION REQUIRED %0.2f seconds  -  rPLV(%0.2f)" % (
                    t, time, reqtime, plv_r))

            else:
                evolution_sv.append(state_variables.copy())
                evolution_net.append([weights])
                print("   . ts%0.2f/%0.2f" % (t, time), end="\r")

        return [np.arange(0, time, dt), evolution_sv, evolution_net]

    def Laplacian(self, weights):
        # Weighted adjacency, Diagonal and Laplacian matrices
        Wij = np.divide(weights, np.square(self.initConn["value"].tract_lengths),
                        where=np.square(self.initConn["value"].tract_lengths) != 0,
                        # Where to compute division; else out
                        out=np.zeros_like(weights))  # array allocation
        Dii = np.eye(len(Wij)) * np.sum(Wij, axis=0)
        Lij = (Dii - Wij)

        return Lij

    def dfun(self, state_variables, Lij, d_fftp):
        # Here we want to model the spread of proteinopathies.
        # Approach without activity dependent spread/generation. Following Alexandersen 2022.

        AB = state_variables[0]
        ABt = state_variables[1]
        TAU = state_variables[2]
        TAUt = state_variables[3]

        ABdam = state_variables[4]
        TAUdam = state_variables[5]

        He_ = state_variables[6]
        Hi_ = state_variables[7]
        taue_ = state_variables[8]
        taui_ = state_variables[9]

        POWdam = state_variables[10]

        # Unpack heterogeneous rho
        [rho_AB, rho_ABt, rho_TAU, rho_TAUt] = self.rho["value"][0] \
            if len(self.rho["value"].shape) == 2 else self.rho["value"].repeat(4)

        # Derivatives
        ###  Amyloid-beta
        dAB = -rho_AB * np.sum(Lij * AB, axis=1) + self.prodAB["value"] * POWdam - self.clearAB["value"] * AB - \
              self.transAB2t["value"] * AB * ABt
        dABt = -rho_ABt * np.sum(Lij * ABt, axis=1) - self.clearABt["value"] * ABt + self.transAB2t[
            "value"] * AB * ABt

        ###  (hyperphosphorilated) Tau
        dTAU = -rho_TAU * np.sum(Lij * TAU, axis=1) + self.prodTAU["value"] - self.clearTAU["value"] * TAU - \
               self.transTAU2t["value"] * TAU * TAUt
        dTAUt = -rho_TAUt * np.sum((Lij * POWdam).transpose() * TAUt, axis=1) - self.clearTAUt["value"] * TAUt + \
                self.transTAU2t["value"] * TAU * TAUt

        dABdam = self.AB_damrate["value"] * ABt * (1 - ABdam)
        dTAUdam = self.TAU_damrate["value"] * TAUt * (1 - TAUdam)

        ## POWER impact
        dPOWdam = self.POW_damrate["value"] * d_fftp * (self.maxPOWdam["value"] - POWdam)

        ## ACTIVATION transfers: a(exc), b(inhi)
        dHe = (self.cABexc["value"] * ABdam * ((self.init_He["range"][1] - He_) - self.cTAU["value"] * TAUdam) * (
                    He_ - self.init_He["range"][0]))
        dHi = - self.cABinh["value"] * ABdam * (Hi_ - self.init_Hi["range"][0])

        ## FREQUENCY transfers: c(delays)
        dtaue = self.cTAU["value"] * TAUdam * (self.init_taue["range"][1] - taue_)
        dtaui = taui_ - taui_  # By now, taui does not change

        derivative = np.array([dAB, dABt, dTAU, dTAUt, dABdam, dTAUdam, dHe, dHi, dtaue, dtaui, dPOWdam])

        return derivative


def simulate_v2(subj, weights, model, g, s, g_wc=None, p_th=0.12, sigma=0.022, sv=None, t=10):
    # Prepare simulation parameters
    simLength = t * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 1000  # ms

    tic = time.time()

    # STRUCTURAL CONNECTIVITY      #########################################

    conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
    conn.weights = weights
    conn.weights = conn.scaled_weights(mode="tract")  # did they normalize? maybe this affects to the spreading?

    conn.speed = np.array([s])

    # Define regions implicated in Functional analysis: not considering subcortical ROIs
    cortical_rois = ['ctx-lh-bankssts', 'ctx-rh-bankssts', 'ctx-lh-caudalanteriorcingulate',
                     'ctx-rh-caudalanteriorcingulate',
                     'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-rh-cuneus',
                     'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
                     'ctx-lh-fusiform', 'ctx-rh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
                     'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal', 'ctx-lh-insula', 'ctx-rh-insula',
                     'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                     'ctx-rh-lateraloccipital',
                     'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-rh-lingual',
                     'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-middletemporal',
                     'ctx-rh-middletemporal',
                     'ctx-lh-paracentral', 'ctx-rh-paracentral', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
                     'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-rh-parsorbitalis',
                     'ctx-lh-parstriangularis', 'ctx-rh-parstriangularis', 'ctx-lh-pericalcarine',
                     'ctx-rh-pericalcarine',
                     'ctx-lh-postcentral', 'ctx-rh-postcentral', 'ctx-lh-posteriorcingulate',
                     'ctx-rh-posteriorcingulate',
                     'ctx-lh-precentral', 'ctx-rh-precentral', 'ctx-lh-precuneus', 'ctx-rh-precuneus',
                     'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
                     'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
                     'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal', 'ctx-lh-superiorparietal',
                     'ctx-rh-superiorparietal',
                     'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'ctx-lh-supramarginal',
                     'ctx-rh-supramarginal',
                     'ctx-lh-temporalpole', 'ctx-rh-temporalpole', 'ctx-lh-transversetemporal',
                     'ctx-rh-transversetemporal']

    #  Load FC labels, transform to SC format; check if match SC.
    FClabs = list(np.loadtxt(data_folder + "FCavg_matrices/" + subj + "_roi_labels.txt", dtype=str))
    FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    FC_cortex_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois

    # load SC labels.
    SClabs = list(conn.region_labels)
    SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

    #   NEURAL MASS MODEL  &  COUPLING FUNCTION   #########################################################
    sigma_array = np.asarray([sigma if 'Thal' in roi else 0 for roi in conn.region_labels])
    p_array = np.asarray([p_th if 'Thal' in roi else 0.09 for roi in conn.region_labels])

    if model == "jrd":  # JANSEN-RIT-DAVID
        # Parameters edited from David and Friston (2003).
        m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                               tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                               He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                               tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                               w=np.array([0.8]), c=np.array([135.0]),
                               c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                               c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                               v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                               p=np.array([p_array]), sigma=np.array([sigma_array]))

        # Remember to hold tau*H constant.
        m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
        m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

        coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=np.array([0.8]), e0=np.array([0.005]),
                                                v0=np.array([6.0]), r=np.array([0.56]))

    elif model == "jrwc":  # JANSEN-RIT(cx) + WILSON-COWAN(th)

        jrMask_wc = [[False] if 'Thal' in roi else [True] for roi in conn.region_labels]

        m = JansenRit_WilsonCowan(
            # Jansen-Rit nodes parameters. From Stefanovski et al. (2019)
            He=np.array([3.25]), Hi=np.array([22]),
            tau_e=np.array([10]), tau_i=np.array([20]),
            c=np.array([135.0]), p=np.array([p_array]),
            c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
            c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
            v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
            # Wilson-Cowan nodes parameters. From Abeysuriya et al. (2018)
            P=np.array([0.31]), sigma=np.array([sigma_array]), Q=np.array([0]),
            c_ee=np.array([3.25]), c_ei=np.array([2.5]),
            c_ie=np.array([3.75]), c_ii=np.array([0]),
            tau_e_wc=np.array([10]), tau_i_wc=np.array([20]),
            a_e=np.array([4]), a_i=np.array([4]),
            b_e=np.array([1]), b_i=np.array([1]),
            c_e=np.array([1]), c_i=np.array([1]),
            k_e=np.array([1]), k_i=np.array([1]),
            r_e=np.array([0]), r_i=np.array([0]),
            theta_e=np.array([0]), theta_i=np.array([0]),
            alpha_e=np.array([1]), alpha_i=np.array([1]),
            # JR mask | WC mask
            jrMask_wc=np.asarray(jrMask_wc))

        m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])

        coup = coupling.SigmoidalJansenRit_Linear(
            a=np.array([g]), e0=np.array([0.005]), v0=np.array([6]), r=np.array([0.56]),
            # Jansen-Rit Sigmoidal coupling
            a_linear=np.asarray([g_wc]),  # Wilson-Cowan Linear coupling
            jrMask_wc=np.asarray(jrMask_wc))  # JR mask | WC mask

    else:  # JANSEN-RIT
        # Parameters from Stefanovski 2019.
        m = JansenRit1995(He=np.array(sv[0]), Hi=np.array(sv[1]),
                          tau_e=np.array(sv[2]), tau_i=np.array([sv[3]]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([0.09]), sigma=np.array([0]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))

    # OTHER PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    # print("Simulating %s (%is)  ||  PARAMS: g%i sigma%0.2f" % (model, simLength / 1000, g, 0.022))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    if model == "jrd":
        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
    else:
        raw_data = output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T
    raw_time = output[0][0][transient:]

    fftp = FFTpeaks(raw_data, simLength - transient)

    # Further analysis on Cortical signals
    cx_data = raw_data[SC_cortex_idx, :]

    # PLOTs :: Signals and spectra
    # timeseries_spectra(raw_data[:], simLength, transient, regionLabels, mode="inline", freqRange=[2, 40], opacity=1)

    bands = [["3-alpha"], [(8, 12)]]
    # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

    for b in range(len(bands[0])):
        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(cx_data, samplingFreq, lowcut, highcut, verbose=False)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals", verbose=False)

        # Obtain Analytical signal
        efPhase = list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.angle(analyticalSignal))
            efEnvelope.append(np.abs(analyticalSignal))

        # Check point
        # from toolbox import timeseriesPlot, plotConversions
        # regionLabels = conn.region_labels
        # timeseriesPlot(raw_data, raw_time, regionLabels)
        # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

        # CONNECTIVITY MEASURES
        ## PLV and plot
        plv = PLV(efPhase, verbose=False)

        # Load empirical data to make simple comparisons
        plv_emp = \
            np.loadtxt(data_folder + "FCavg_matrices/" + subj + "_" + bands[0][b] + "_plv_avg.txt", delimiter=',')[:,
            FC_cortex_idx][
                FC_cortex_idx]

        # Comparisons
        t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
        t1[0, :] = plv[np.triu_indices(len(plv), 1)]
        t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
        plv_r = np.corrcoef(t1)[0, 1]

    return raw_data, raw_time, fftp, plv, plv_emp, plv_r, conn.region_labels, simLength, transient, time.time() - tic


def correlations_v2(output, scatter="REL_simple", band="3-alpha", title="new", folder="figures"):
    """
    Calculate and return both absolute and relative correlations
    for all the interesting variables (PET-tau, PET-ab, FC, FC-dmn, FC-theory, SC).
    PET correlations are calculated against toxic simulated burden.

    Then plot everything if asked for.

    :param output:
    :param scatter:
    :param band:
    :param title:
    :return:
    """

    # Here we wanna get a simplify array with (n_refcond, time) containing correlation values
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

    adni_refgroups = ["CN", "SMC", "EMCI", "LMCI", "AD"]
    c3n_refgroups = ["HC-fam", "FAM", "QSM", "MCI", "MCI-conv"]

    CORRs = []
    ## A. Compute Correlations
    ## 1. ABSOLUTE CORRELATIONS (for PET cumulative change; for FC & SC full matrices comparisons
    print("Working on ABSOLUTE correlations: PET", end=", ")
    # 1.1 PET
    #    ADNI PET DATA       ##########
    ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupREL_2CN.csv", index_col=0)

    # Check label order
    conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/HC-fam_aparc_aseg-mni_09c.zip")
    PETlabs = list(ADNI_AVG.columns[12:])
    PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]

    # loop over refgroups: ["CN", "SMC", "EMCI", "LMCI", "AD"]

    corr_groups = []
    df_corr = pd.DataFrame()
    for j, group in enumerate(adni_refgroups):

        transition = adni_refgroups[j] + "_rel2CN"

        AB_emp = np.squeeze(
            np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == group)].iloc[:, 12:]))
        AB_emp = AB_emp[PET_idx]

        TAU_emp = np.squeeze(
            np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == group)].iloc[:, 12:]))
        TAU_emp = TAU_emp[PET_idx]

        # Calculate the derivatives on the simulated data
        dABt = np.asarray(
            [np.asarray(output[1])[i, 1, :] - np.asarray(output[1])[0, 1, :] for i in range(len(output[1]) - 1)])
        dTAUt = np.asarray(
            [np.asarray(output[1])[i, 3, :] - np.asarray(output[1])[0, 3, :] for i in range(len(output[1]) - 1)])

        corr_group_t = []
        for i in range(len(dABt)):
            # Correlate increase in empirical with derivatives in simulated
            corr_group_t.append([np.corrcoef(AB_emp, dABt[i, :])[0, 1], np.corrcoef(TAU_emp, dTAUt[i, :])[0, 1]])

            if "ABS" in scatter:
                # Create dataframe to plot
                df_corr = df_corr.append(
                    pd.DataFrame(
                        [["rel2CN"] * len(dABt[i, :]), [output[0][i]] * len(dABt[i, :]), [transition] * len(dABt[i, :]),
                         ["ABt"] * len(dABt[i, :]), dABt[i, :], AB_emp, conn.region_labels]).transpose())

                df_corr = df_corr.append(
                    pd.DataFrame(
                        [["rel2CN"] * len(dTAUt[i, :]), [output[0][i]] * len(dTAUt[i, :]), [transition] * len(dTAUt[i, :]),
                         ["TAUt"] * len(dTAUt[i, :]), dTAUt[i, :], TAU_emp, conn.region_labels]).transpose())

        corr_groups.append(corr_group_t)
    CORRs.append(corr_groups)

    print("FC", end=", ")
    # 1.2 FC
    # Define regions implicated in Functional analysis: not considering subcortical ROIs
    cortical_rois = ['ctx-lh-bankssts', 'ctx-rh-bankssts',
                     'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',
                     'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal',
                     'ctx-lh-cuneus', 'ctx-rh-cuneus',
                     'ctx-lh-entorhinal', 'ctx-rh-entorhinal',
                     'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
                     'ctx-lh-fusiform', 'ctx-rh-fusiform',
                     'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
                     'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal',
                     'ctx-lh-insula', 'ctx-rh-insula',
                     'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate',
                     'ctx-lh-lateraloccipital', 'ctx-rh-lateraloccipital',
                     'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal',
                     'ctx-lh-lingual', 'ctx-rh-lingual',
                     'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal',
                     'ctx-lh-middletemporal', 'ctx-rh-middletemporal',
                     'ctx-lh-paracentral', 'ctx-rh-paracentral',
                     'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
                     'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis',
                     'ctx-lh-parsorbitalis', 'ctx-rh-parsorbitalis',
                     'ctx-lh-parstriangularis', 'ctx-rh-parstriangularis',
                     'ctx-lh-pericalcarine', 'ctx-rh-pericalcarine',
                     'ctx-lh-postcentral', 'ctx-rh-postcentral',
                     'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate',
                     'ctx-lh-precentral', 'ctx-rh-precentral',
                     'ctx-lh-precuneus', 'ctx-rh-precuneus',
                     'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
                     'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
                     'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal',
                     'ctx-lh-superiorparietal', 'ctx-rh-superiorparietal',
                     'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal',
                     'ctx-lh-supramarginal', 'ctx-rh-supramarginal',
                     'ctx-lh-temporalpole', 'ctx-rh-temporalpole',
                     'ctx-lh-transversetemporal', 'ctx-rh-transversetemporal']
    dmn_rois = [ # ROIs not in Gianlucas set, from cingulum bundle description
        'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
        'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal',
        'ctx-lh-insula', 'ctx-rh-insula',
        'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',  # A6. in Gianlucas DMN
        'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',  # A6. in Gianlucas DMN
        'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate',
        'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',  # A7. in Gianlucas DMN
        'ctx-lh-middletemporal', 'ctx-rh-middletemporal',  # A5. in Gianlucas DMN
        'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal',  # A4. in Gianlucas DMN
        'ctx-lh-superiorparietal', 'ctx-rh-superiorparietal',
        'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',  # A3. in Gianlucas DMN
        'ctx-lh-precuneus', 'ctx-rh-precuneus'  # A1. in Gianlucas DMN
    ]

    #  Load FC labels, transform to SC format; check if match SC.
    FClabs = list(np.loadtxt(data_folder + "FCavg_matrices/" + c3n_refgroups[0] + "_roi_labels.txt", dtype=str))
    FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    FC_cortex_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois
    FC_dmn_idx = [FClabs.index(roi) for roi in dmn_rois]

    shortout = [np.array([out[3] for i, out in enumerate(output[2]) if len(out) > 1]),
                np.array([output[0][i] for i, out in enumerate(output[2]) if len(out) > 1])]

    corr_groups = []
    for group in c3n_refgroups:
        plv_emp = np.loadtxt(data_folder + "FCavg_matrices/" + group + "_" + band + "_plv_avg.txt", delimiter=',')[:,
                  FC_cortex_idx][
            FC_cortex_idx]

        t1 = np.zeros(shape=(2, len(plv_emp) ** 2 // 2 - len(plv_emp) // 2))
        t1[0, :] = plv_emp[np.triu_indices(len(plv_emp), 1)]

        corr_group_t = []
        for plv_sim in shortout[0]:
            t1[1, :] = plv_sim[np.triu_indices(len(plv_emp), 1)]
            corr_group_t.append(np.corrcoef(t1)[0, 1])

        corr_groups.append(corr_group_t)

    CORRs.append([corr_groups, shortout[1]])

    # Do it for DMN
    corr_groups = []
    for group in c3n_refgroups:
        plv_emp = np.loadtxt(data_folder + "FCavg_matrices/" + group + "_" + band + "_plv_avg.txt", delimiter=',')[:,
                  FC_dmn_idx][
            FC_dmn_idx]

        t1 = np.zeros(shape=(2, len(plv_emp) ** 2 // 2 - len(plv_emp) // 2))
        t1[0, :] = plv_emp[np.triu_indices(len(plv_emp), 1)]

        corr_group_t = []
        for plv_sim in shortout[0]:
            plv_sim = plv_sim[:, FC_dmn_idx][FC_dmn_idx]
            t1[1, :] = plv_sim[np.triu_indices(len(plv_emp), 1)]
            corr_group_t.append(np.corrcoef(t1)[0, 1])

        corr_groups.append(corr_group_t)

    CORRs.append([corr_groups, shortout[1]])


    print("SC", end=".\n\n")
    # 1.3 SC
    corr_groups = []
    for group in c3n_refgroups:
        conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + group + "_aparc_aseg-mni_09c.zip")
        conn.weights = conn.scaled_weights(mode="tract")

        t1 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
        t1[0, :] = conn.weights[np.triu_indices(len(conn.region_labels), 1)]

        # correlate and save
        corr_group_t = []
        for i in range(len(output[2])):
            t1[1, :] = output[2][i][0][np.triu_indices(len(conn.region_labels), 1)]
            corr_group_t.append(np.corrcoef(t1)[0, 1])

        corr_groups.append(corr_group_t)

    CORRs.append(corr_groups)


    ## 2. RELATIVE CORRELATIONS
    print("Working on RELATIVE correlations: PET", end=", ")

    # 2.1 PET
    #    ADNI PET DATA       ##########
    ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupREL_2PrevStage.csv", index_col=0)

    # Check label order
    conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/HC-fam_aparc_aseg-mni_09c.zip")
    PETlabs = list(ADNI_AVG.columns[7:])
    PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]

    corr_groups = []
    for j, group in enumerate(adni_refgroups[:-1]):

        transition = group + "-" + adni_refgroups[j + 1]

        AB_emp = np.squeeze(
            np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Transition"] == transition)].iloc[:, 7:]))
        AB_emp = AB_emp[PET_idx]

        TAU_emp = np.squeeze(np.asarray(
            ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Transition"] == transition)].iloc[:, 7:]))
        TAU_emp = TAU_emp[PET_idx]

        # Calculate the derivatives on the simulated data
        dABt = np.asarray(
            [np.asarray(output[1])[i + 1, 1, :] - np.asarray(output[1])[i, 1, :] for i in range(len(output[1]) - 1)])
        dTAUt = np.asarray(
            [np.asarray(output[1])[i + 1, 3, :] - np.asarray(output[1])[i, 3, :] for i in range(len(output[1]) - 1)])

        corr_group_t = []
        for i in range(len(dABt)):
            # Correlate increase in empirical with derivatives in simulated
            corr_group_t.append([np.corrcoef(AB_emp, dABt[i, :])[0, 1], np.corrcoef(TAU_emp, dTAUt[i, :])[0, 1]])

            if "REL" in scatter:
                # Create dataframe to plot
                df_corr = df_corr.append(
                    pd.DataFrame(
                        [["rel2PS"] * len(dABt[i, :]), [output[0][i]] * len(dABt[i, :]), [transition] * len(dABt[i, :]),
                         ["ABt"] * len(dABt[i, :]), dABt[i, :], AB_emp, conn.region_labels]).transpose())

                df_corr = df_corr.append(
                    pd.DataFrame(
                        [["rel2PS"] * len(dTAUt[i, :]), [output[0][i]] * len(dTAUt[i, :]), [transition] * len(dTAUt[i, :]),
                         ["TAUt"] * len(dTAUt[i, :]), dTAUt[i, :], TAU_emp, conn.region_labels]).transpose())

        corr_groups.append(corr_group_t)

    CORRs.append(corr_groups)


    print("FC", end=", ")
    # 2.2 FC
    #  Load FC labels, transform to SC format; check if match SC.
    FClabs = list(np.loadtxt(data_folder + "FCavg_matrices/" + c3n_refgroups[0] + "_roi_labels.txt", dtype=str))
    FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    FC_cortex_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois

    shortout_rel = [np.array([out[3] for i, out in enumerate(output[2]) if len(out) > 1]),
                    np.array([output[0][i] for i, out in enumerate(output[2]) if len(out) > 1])]

    corr_groups = []
    for j, group in enumerate(c3n_refgroups[:-1]):

        plv_emp0 = np.loadtxt(data_folder + "FCavg_matrices/" + group + "_" + band + "_plv_avg.txt", delimiter=',')[:,
                   FC_cortex_idx][
            FC_cortex_idx]

        plv_emp1 = \
        np.loadtxt(data_folder + "FCavg_matrices/" + c3n_refgroups[j + 1] + "_" + band + "_plv_avg.txt", delimiter=',')[
        :,
        FC_cortex_idx][
            FC_cortex_idx]

        dplv_emp = plv_emp1 - plv_emp0

        t1 = np.zeros(shape=(2, len(dplv_emp) ** 2 // 2 - len(dplv_emp) // 2))
        t1[0, :] = dplv_emp[np.triu_indices(len(dplv_emp), 1)]

        # TODO compare with derivatives
        corr_group_t = []
        for i, plv_sim0 in enumerate(shortout_rel[0][:-1]):
            dplv_sim = shortout_rel[0][i + 1] - plv_sim0

            t1[1, :] = dplv_sim[np.triu_indices(len(dplv_emp), 1)]
            corr_group_t.append(np.corrcoef(t1)[0, 1])

        corr_groups.append(corr_group_t)

    CORRs.append([corr_groups, shortout[1]])

    # Do it for DMN
    corr_groups = []
    for j, group in enumerate(c3n_refgroups[:-1]):

        plv_emp0 = np.loadtxt(data_folder + "FCavg_matrices/" + group + "_" + band + "_plv_avg.txt", delimiter=',')[:,
                   FC_dmn_idx][
            FC_dmn_idx]

        plv_emp1 = \
        np.loadtxt(data_folder + "FCavg_matrices/" + c3n_refgroups[j + 1] + "_" + band + "_plv_avg.txt", delimiter=',')[
        :,
        FC_dmn_idx][
            FC_dmn_idx]

        dplv_emp = plv_emp1 - plv_emp0

        t1 = np.zeros(shape=(2, len(dplv_emp) ** 2 // 2 - len(dplv_emp) // 2))
        t1[0, :] = dplv_emp[np.triu_indices(len(dplv_emp), 1)]

        # TODO compare with derivatives
        corr_group_t = []
        for i, plv_sim0 in enumerate(shortout_rel[0][:-1]):
            plv_sim0 = plv_sim0[:, FC_dmn_idx][FC_dmn_idx]
            plv_sim1 = shortout_rel[0][i + 1][:, FC_dmn_idx][FC_dmn_idx]
            dplv_sim = plv_sim1 - plv_sim0

            t1[1, :] = dplv_sim[np.triu_indices(len(dplv_emp), 1)]
            corr_group_t.append(np.corrcoef(t1)[0, 1])

        corr_groups.append(corr_group_t)

    CORRs.append([corr_groups, shortout[1]])



    print("SC", end=".\n\n")
    # 2.3 SC
    corr_groups = []
    for i, group in enumerate(c3n_refgroups[:-1]):
        conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + group + "_aparc_aseg-mni_09c.zip")
        weights0 = conn.scaled_weights(mode="tract")

        conn = connectivity.Connectivity.from_file(
            data_folder + "SC_matrices/" + c3n_refgroups[i + 1] + "_aparc_aseg-mni_09c.zip")
        weights1 = conn.scaled_weights(mode="tract")

        dweights_emp = weights1 - weights0

        t1 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
        t1[0, :] = dweights_emp[np.triu_indices(len(conn.region_labels), 1)]

        # correlate and save
        corr_group_t = []
        for i in range(len(output[2]))[:-1]:
            dweights_sim = output[2][i + 1][0] - output[2][i][0]

            t1[1, :] = dweights_sim[np.triu_indices(len(conn.region_labels), 1)]
            corr_group_t.append(np.corrcoef(t1)[0, 1])

        corr_groups.append(corr_group_t)

    CORRs.append(corr_groups)

    ## B. PLOTTING (if needed)
    cmap_adni = px.colors.sample_colorscale("Phase", np.arange(0, 1, 1/len(adni_refgroups)))
    cmap_c3n = px.colors.sample_colorscale("Phase", np.insert(np.arange(0, 1, 1/len(adni_refgroups)), 1, 0.1))
    op_p, op_s = 0.4, 0.9

    fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.14,
                        column_titles=["ADNI PET", "C3N FC<br>(alpha [8-12Hz])", "C3N SC"])


    # 3.1 Add Absolute lines
    for i, group in enumerate(adni_refgroups):
        fig.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[0])[i, :, 0], mode="lines", name=group + " _AB",
                                 legendgroup="corradni", opacity=op_p,
                                 line=dict(color=cmap_adni[i], width=3), showlegend=True), row=1, col=1)

        fig.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[0])[i, :, 1], mode="lines", name=group + " _TAU",
                                 legendgroup="corradni", opacity=op_s,
                                 line=dict(color=cmap_adni[i], width=2, dash="dash"),  # visible="legendonly",
                                 showlegend=True), row=1, col=1)

    for i, group in enumerate(c3n_refgroups):
        fig.add_trace(go.Scatter(x=CORRs[1][1], y=np.array(CORRs[1][0][i]), mode="lines", name=group + " _FC",
                                 legendgroup="corrc3n", opacity=op_p,
                                 line=dict(color=cmap_c3n[i], width=3), showlegend=True), row=1, col=2)

        fig.add_trace(go.Scatter(x=CORRs[1][1], y=np.array(CORRs[2][0][i]), mode="lines", name=group + " _FCdmn",
                                 legendgroup="corrc3n", opacity=op_s,
                                 line=dict(color=cmap_c3n[i], width=2, dash="dash"), showlegend=True), row=1, col=2)

        fig.add_trace(
            go.Scatter(x=output[0], y=np.array(CORRs[3][i]), mode="lines", name=group + " _SC", legendgroup="corrc3n",
                       line=dict(color=cmap_c3n[i], width=3), opacity=op_p, showlegend=False), row=1, col=3)


    # 3.2 Add  Relative lines
    for i, group in enumerate(adni_refgroups[:-1]):
        transition = group + "-" + adni_refgroups[i + 1]
        fig.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[4])[i, :, 0], mode="lines", name=transition + " _AB",
                                 legendgroup="corradnirel", opacity=op_p,
                                 line=dict(color=cmap_adni[i+1], width=3), showlegend=True), row=2, col=1)

        fig.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[4])[i, :, 1], mode="lines", name=transition + " _TAU",
                                 legendgroup="corradnirel", opacity=op_s,
                                 line=dict(color=cmap_adni[i+1], width=2, dash="dash"),  # visible="legendonly",
                                 showlegend=False), row=2, col=1)

    for i, group in enumerate(c3n_refgroups[:-1]):
        transition = group + "-" + c3n_refgroups[i + 1]
        fig.add_trace(go.Scatter(x=CORRs[1][1], y=np.array(CORRs[5][0][i]), mode="lines", name=transition + " _FC",
                                 legendgroup="corrc3nrel", opacity=op_p,
                                 line=dict(color=cmap_c3n[i+1], width=3), showlegend=True), row=2, col=2)

        fig.add_trace(go.Scatter(x=CORRs[1][1], y=np.array(CORRs[6][0][i]), mode="lines", name=transition + " _FCdmn",
                                 legendgroup="corrc3nrel", opacity=op_s,
                                 line=dict(color=cmap_c3n[i+1], width=2, dash="dash"), showlegend=True), row=2, col=2)

        fig.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[7][i]), mode="lines", name=transition + " _SC",
                                 legendgroup="corrc3nrel", opacity=op_p,
                                 line=dict(color=cmap_c3n[i+1], width=3), showlegend=False), row=2, col=3)

    fig.update_layout(template="plotly_white", legend=dict(groupclick="toggleitem"),
                      yaxis1=dict(title="PET Cumulative change (from CN)<br>Pearson's r", range=[-1, 1]),
                      yaxis2=dict(title="FC matrices<br>Pearson's r", range=[-1, 1]),
                      yaxis3=dict(title="SC matrices<br>Pearson's r", range=[-1, 1]),
                      yaxis4=dict(title="PET Relative change (from prev. stage)<br>Pearson's r", range=[-1, 1]), xaxis5=dict(title="Time (years)"),
                      yaxis5=dict(title="FC Relative changes (from prev. stage)<br>Pearson's r", range=[-1, 1]), xaxis6=dict(title="Time (years)"),
                      yaxis6=dict(title="SC Relative changes (from prev. stage)<br>Pearson's r", range=[-1, 1]), xaxis7=dict(title="Time (years)"))

    pio.write_html(fig, folder + "/CORRELATIONS_" + title + ".html", auto_open=True)


    # plot scatter
    if "ABS" in scatter:

        add_space = 0.005
        print("CORRELATIONS  _Plotting animation - wait patiently")
        df_corr.columns = ["mode", "time", "group", "pet", "sim", "emp", "roi"]

        df_sub = df_corr.loc[df_corr["mode"] == "rel2CN"]

        if "mult" in scatter:
            fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", facet_col="group", animation_frame="time",
                             hover_name="roi")
            fig.update_layout(title=title + " _ABS " + scatter.split("_")[1], template="plotly_white",
                              yaxis1=dict(
                                  range=[min(df_sub["sim"].values) - add_space, max(df_sub["sim"].values) + add_space]))


        elif "color" in scatter:
            df_sub = df_sub.loc[df_sub["group"] == "LMCI-AD"]
            fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time",
                             hover_name="roi", color="roi")
            fig.update_layout(title=title + " _ABS " + scatter.split("_")[1], template="plotly_white",
                              yaxis1=dict(
                                  range=[min(df_sub["sim"].values) - add_space, max(df_sub["sim"].values) + add_space]))


        elif "simple" in scatter:
            df_sub = df_sub.loc[df_sub["group"] == "LMCI-AD"]
            fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time", hover_name="roi")
            fig.update_layout(title=title + " _ABS " + scatter.split("_")[1], template="plotly_white",
                              yaxis1=dict(
                                  range=[min(df_sub["sim"].values) - add_space, max(df_sub["sim"].values) + add_space]))

        pio.write_html(fig, folder + "/ScatterCorr-" + scatter + "-" + title + ".html",
                       auto_open=True)

    if "REL" in scatter:

        add_space = 0.001
        print("CORRELATIONS  _Plotting animation - wait patiently")
        df_corr.columns = ["mode", "time", "transition", "pet", "sim", "emp", "roi"]

        df_sub = df_corr.loc[df_corr["mode"] == "rel2PS"]

        if "mult" in scatter:
            fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", facet_col="transition", animation_frame="time",
                             hover_name="roi")
            fig.update_layout(title=title + " _REL " + scatter.split("_")[1], template="plotly_white", yaxis1=dict(
                range=[min(df_sub["sim"].values) - add_space, max(df_sub["sim"].values) + add_space]))


        elif "color" in scatter:
            df_sub = df_sub.loc[df_sub["transition"] == "LMCI-AD"]
            fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time",
                             hover_name="roi", color="roi", )
            fig.update_layout(title=title + " _REL " + scatter.split("_")[1], template="plotly_white", yaxis1=dict(
                range=[min(df_sub["sim"].values) - add_space, max(df_sub["sim"].values) + add_space]))


        elif "simple" in scatter:
            df_sub = df_sub.loc[df_sub["transition"] == "LMCI-AD"]
            fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time", hover_name="roi")
            fig.update_layout(title=title + " _REL " + scatter.split("_")[1], template="plotly_white", yaxis1=dict(
                range=[min(df_sub["sim"].values) - add_space, max(df_sub["sim"].values) + add_space]))

        pio.write_html(fig, folder + "/ScatterCorr-" + scatter + "-" + title + ".html",
                       auto_open=True)

    return CORRs


def animate_propagation_v4(output, corrs, refgroups, reftype, conn, timeref=True, title="", folder="figures"):

    adni_refgroups = ["CN", "SMC", "EMCI", "LMCI", "AD"]
    c3n_refgroups = ["HC-fam", "FAM", "QSM", "MCI", "MCI-conv"]

    # Create text labels per ROI
    hovertext3d = [["<b>" + roi + "</b><br>"
                    + str(round(output[1][ii][0, i], 5)) + "(M) a-beta<br>"
                    + str(round(output[1][ii][1, i], 5)) + "(M) a-beta toxic <br>"
                    + str(round(output[1][ii][2, i], 5)) + "(M) pTau <br>"
                    + str(round(output[1][ii][3, i], 5)) + "(M) pTau toxic <br>"
                    for i, roi in enumerate(conn.region_labels)] for ii, t in enumerate(output[0])]

    sz_ab, sz_t = 25, 10  # Different sizes for AB and pT nodes

    if any(len(out) > 1 for out in output[2]):

        shortout = [np.array([np.average(out[1]) for i, out in enumerate(output[2]) if len(out) > 1]),
                    np.array([np.average(out[2]) for i, out in enumerate(output[2]) if len(out) > 1]),
                    np.array([output[0][i] for i, out in enumerate(output[2]) if len(out) > 1])]

        ## ADD INITIAL TRACE for 3dBrain - t0
        fig = make_subplots(rows=2, cols=3,
                            specs=[[{"rowspan": 2, "type": "surface"}, {}, {}], [{}, {}, {"secondary_y": True}]],
                            column_widths=[0.5, 0.25, 0.25], shared_xaxes=True, horizontal_spacing=0.075,
                            subplot_titles=(
                            ['<b>Protein accumulation dynamics</b>', '', '', '', 'Correlations (emp-sim)', '']))

        # Add trace for AB + ABt
        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="AB", showlegend=True,
                                   legendgroup="AB",
                                   marker=dict(size=(np.abs(output[1][0][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                               cmax=0.5, cmin=-0.25,
                                               color=np.abs(output[1][0][1, :]) / np.abs(output[1][0][0, :]),
                                               opacity=0.5,
                                               line=dict(color="grey", width=1), colorscale="YlOrBr")), row=1, col=1)

        # Add trace for TAU + TAUt
        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="TAU", showlegend=True,
                                   legendgroup="TAU",
                                   marker=dict(size=(np.abs(output[1][0][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                               cmax=0.5, cmin=-0.25,
                                               color=np.abs(output[1][0][3, :]) / np.abs(output[1][0][2, :]), opacity=1,
                                               line=dict(color="grey", width=1), colorscale="BuPu", symbol="diamond")),
                      row=1, col=1)

        ## ADD INITIAL TRACE for lines
        sim_pet_avg = np.average(np.asarray(output[1]), axis=2)

        if timeref:
            # Add dynamic reference - t0
            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[-0.15, 1.15], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=1, col=2)

            min_r, max_r = np.min(corrs) - 0.15, np.max(corrs) + 0.15
            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[min_r, max_r], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=2, col=2)

            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[0, 25], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=1, col=3)

            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[0, max(shortout[1])], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=2, col=3)

        # Add static lines - PET proteins concentrations
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 0], mode="lines", name="AB", legendgroup="AB",
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7,
                                 showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 1], mode="lines", name="AB toxic", legendgroup="AB",
                                 line=dict(color=px.colors.sequential.YlOrBr[5], width=3), opacity=0.7,
                                 showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 2], mode="lines", name="TAU", legendgroup="TAU",
                                 line=dict(color=px.colors.sequential.BuPu[3], width=3), opacity=0.7, showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 3], mode="lines", name="TAU toxic", legendgroup="TAU",
                                 line=dict(color=px.colors.sequential.BuPu[5], width=3), opacity=0.7, showlegend=True),
                      row=1, col=2)

        # Add static lines - data correlations
        cmap_adni = px.colors.sample_colorscale("Phase", np.arange(0, 1, 1 / len(adni_refgroups)))
        cmap_c3n = px.colors.sample_colorscale("Phase", np.insert(np.arange(0, 1, 1 / len(adni_refgroups)), 1, 0.1))
        op_p, op_s = 0.4, 0.9
        for ii, group in enumerate(refgroups):
            if "PET" in reftype:
                c = ii + 1 if "rel" in reftype else ii
                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 0], mode="lines", name=group + " - AB",
                                         legendgroup="corrAB", opacity=op_p,
                                         line=dict(color=cmap_adni[c], width=3), showlegend=True), row=2, col=2)

                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 1], mode="lines", name=group + " - rTAU",
                                         legendgroup="corrTAU", opacity=op_s,
                                         line=dict(color=cmap_adni[c], width=2, dash="dash"),  # visible="legendonly",
                                         showlegend=True), row=2, col=2)

            else:
                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :], mode="lines", name=group + " - r" + reftype,
                                         legendgroup="corr", opacity=op_s,
                                         line=dict(color=cmap_c3n[c], width=3), showlegend=True), row=2, col=2)

        # Add static lines - parameter values
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 6], mode="lines", name="He", legendgroup="params",
                                 line=dict(width=3), opacity=0.7, showlegend=True),
                      row=1, col=3)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 7], mode="lines", name="Hi", legendgroup="params",
                                 line=dict(width=3), opacity=0.7, showlegend=True),
                      row=1, col=3)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 8], mode="lines", name="tau_e", legendgroup="params",
                                 line=dict(width=3), opacity=0.7, showlegend=True),
                      row=1, col=3)

        # Add static lines - average spectral properties
        fig.add_trace(go.Scatter(x=shortout[2], y=shortout[1], mode="lines", name="Power", legendgroup="spectra",
                                 line=dict(width=4, color="lawngreen"), opacity=0.7, showlegend=True),
                      row=2, col=3)

        fig.add_trace(go.Scatter(x=shortout[2], y=shortout[0], mode="lines", name="Frequency", legendgroup="spectra",
                                 line=dict(width=2, color="mediumvioletred"), opacity=0.7, showlegend=True),
                      row=2, col=3, secondary_y=True)

        ## ADD FRAMES - t[1:end]
        if timeref:

            fig.update(frames=[go.Frame(data=[
                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=(np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                         color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=(np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                         color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),

                go.Scatter(x=[output[0][i], output[0][i]]),
                go.Scatter(x=[output[0][i], output[0][i]]),
                go.Scatter(x=[output[0][i], output[0][i]]),
                go.Scatter(x=[output[0][i], output[0][i]])
            ],
                traces=[0, 1, 2, 3, 4, 5], name=str(i)) for i, t in enumerate(output[0])])
        else:
            fig.update(frames=[go.Frame(data=[
                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :]) * sz_ab,
                                         color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :]) * sz_t,
                                         color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),
            ],
                traces=[0, 1], name=str(i)) for i, t in enumerate(output[0])])

        # CONTROLS : Add sliders and buttons
        fig.update_layout(
            template="plotly_white", legend=dict(x=1, y=0.55, tracegroupgap=10, groupclick="toggleitem"),
            scene=dict(xaxis=dict(title="Sagital axis<br>(L-R)"),
                       yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                       zaxis=dict(title="Horizontal axis<br>(inf-sup)")),
            xaxis1=dict(title="Time (Years)"), xaxis2=dict(title="Time (Years)"),
            xaxis4=dict(title="Time (Years)"), xaxis5=dict(title="Time (Years)"),
            yaxis1=dict(title="Concentration (M)"), yaxis2=dict(title="param value"),
            yaxis4=dict(title="Pearson's r"), yaxis5=dict(title="Power (dB)"),
            yaxis6=dict(title="Frequency (Hz)", range=[0, 14]),

            sliders=[dict(
                steps=[
                    dict(method='animate', args=[[str(i)], dict(mode="immediate", frame=dict(duration=250, redraw=True,
                                                                                             easing="cubic-in-out"),
                                                                transition=dict(duration=0))], label=str(t)) for i, t
                    in enumerate(output[0])],
                transition=dict(duration=0), x=0.15, xanchor="left", y=1.4,
                currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
                len=0.8, tickcolor="white")],

            updatemenus=[dict(type="buttons", showactive=False, y=1.35, x=0, xanchor="left",
                              buttons=[
                                  dict(label="Play", method="animate",
                                       args=[None,
                                             dict(frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                                  transition=dict(duration=0),
                                                  fromcurrent=True, mode='immediate')]),
                                  dict(label="Pause", method="animate",
                                       args=[[None],
                                             dict(frame=dict(duration=250, redraw=False, easing="cubic-in-out"),
                                                  transition=dict(duration=0),
                                                  mode="immediate")])])])

        pio.write_html(fig, file=folder + "/ProteinPropagation_&corr" + reftype + "_" + title + ".html", auto_open=True,
                       auto_play=False)

    else:

        ## ADD INITIAL TRACE for 3dBrain - t0
        fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2, "type": "surface"}, {}], [{}, {}]],
                            column_widths=[0.6, 0.4], shared_xaxes=True,
                            subplot_titles=(
                                ['<b>Protein accumulation dynamics</b> ' + title + " - ref: " + reftype, '', '',
                                 'Correlations (emp-sim)', ]))

        # Add trace for AB + ABt
        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="AB", showlegend=True,
                                   legendgroup="AB",
                                   marker=dict(size=(np.abs(output[1][0][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                               cmax=0.5, cmin=-0.25,
                                               color=np.abs(output[1][0][1, :]) / np.abs(output[1][0][0, :]),
                                               opacity=0.5,
                                               line=dict(color="grey", width=1), colorscale="YlOrBr")), row=1, col=1)

        # Add trace for TAU + TAUt
        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="TAU", showlegend=True,
                                   legendgroup="TAU",
                                   marker=dict(size=(np.abs(output[1][0][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                               cmax=0.5, cmin=-0.25,
                                               color=np.abs(output[1][0][3, :]) / np.abs(output[1][0][2, :]), opacity=1,
                                               line=dict(color="grey", width=1), colorscale="BuPu", symbol="diamond")),
                      row=1, col=1)

        ## ADD INITIAL TRACE for lines
        sim_pet_avg = np.average(np.asarray(output[1]), axis=2)

        if timeref:
            # Add dynamic reference - t0
            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[-0.15, 1.15], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=1, col=2)

            min_r, max_r = np.min(corrs) - 0.15, np.max(corrs) + 0.15
            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[min_r, max_r], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=2, col=2)

        # Add static lines - PET proteins concentrations
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 0], mode="lines", name="AB", legendgroup="AB",
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7,
                                 showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 1], mode="lines", name="AB toxic", legendgroup="AB",
                                 line=dict(color=px.colors.sequential.YlOrBr[5], width=3), opacity=0.7,
                                 showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 2], mode="lines", name="TAU", legendgroup="TAU",
                                 line=dict(color=px.colors.sequential.BuPu[3], width=3), opacity=0.7, showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 3], mode="lines", name="TAU toxic", legendgroup="TAU",
                                 line=dict(color=px.colors.sequential.BuPu[5], width=3), opacity=0.7, showlegend=True),
                      row=1, col=2)

        # Add static lines - data correlations
        cmap_p = px.colors.qualitative.Pastel2
        cmap_s = px.colors.qualitative.Set2
        for ii, group in enumerate(refgroups):
            if "PET" in reftype:
                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 0], mode="lines", name=group + " - AB",
                                         legendgroup="corrAB",
                                         line=dict(color=cmap_p[ii], width=3), showlegend=True), row=2, col=2)

                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 1], mode="lines", name=group + " - rTAU",
                                         legendgroup="corrTAU",
                                         line=dict(color=cmap_s[ii], width=2, dash="dash"),  # visible="legendonly",
                                         showlegend=True), row=2, col=2)

            else:
                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :], mode="lines", name=group + " - r" + reftype,
                                         legendgroup="corr",
                                         line=dict(color=cmap_p[ii], width=3), showlegend=True), row=2, col=2)

        # fig.show("browser")

        ## ADD FRAMES - t[1:end]
        if timeref:

            fig.update(frames=[go.Frame(data=[
                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=(np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                         color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=(np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                         color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),

                go.Scatter(x=[output[0][i], output[0][i]]),
                go.Scatter(x=[output[0][i], output[0][i]])
            ],
                traces=[0, 1, 2, 3], name=str(i)) for i, t in enumerate(output[0])])
        else:
            fig.update(frames=[go.Frame(data=[
                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :]) * sz_ab,
                                         color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :]) * sz_t,
                                         color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),
            ],
                traces=[0, 1], name=str(i)) for i, t in enumerate(output[0])])

        # CONTROLS : Add sliders and buttons
        fig.update_layout(
            template="plotly_white", legend=dict(x=1.05, y=0.55, tracegroupgap=40, groupclick="toggleitem"),
            scene=dict(xaxis=dict(title="Sagital axis<br>(L-R)"),
                       yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                       zaxis=dict(title="Horizontal axis<br>(inf-sup)")),
            xaxis1=dict(title="Time (Years)"), xaxis3=dict(title="Time (Years)"),
            yaxis1=dict(title="Concentration (M)"), yaxis3=dict(title="Pearson's corr"),
            sliders=[dict(
                steps=[
                    dict(method='animate', args=[[str(i)], dict(mode="immediate", frame=dict(duration=100, redraw=True,
                                                                                             easing="cubic-in-out"),
                                                                transition=dict(duration=300))], label=str(t)) for i, t
                    in enumerate(output[0])],
                transition=dict(duration=100), x=0.15, xanchor="left", y=1.4,
                currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
                len=0.8, tickcolor="white")],
            updatemenus=[dict(type="buttons", showactive=False, y=1.35, x=0, xanchor="left",
                              buttons=[
                                  dict(label="Play", method="animate",
                                       args=[None,
                                             dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                  transition=dict(duration=300),
                                                  fromcurrent=True, mode='immediate')]),
                                  dict(label="Pause", method="animate",
                                       args=[[None],
                                             dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                  transition=dict(duration=300),
                                                  mode="immediate")])])])

        pio.write_html(fig, file="figures/ProtProp_&corr" + reftype + "_" + title + ".html", auto_open=True)


def g_explore(output, g_sel, param="g", mode="html", folder="figures"):
    n_g = len(g_sel)
    col_titles = [""] + [param + "==" + str(g) for g in g_sel]
    specs = [[{} for g in range(n_g + 1)]] * 3
    id_emp = (n_g + 1) * 2
    sp_titles = ["Empirical" if i == id_emp else "" for i in range((n_g + 1) * 3)]
    fig = make_subplots(rows=3, cols=n_g + 1, specs=specs, row_titles=["signals", "FFT", "FC"],
                        column_titles=col_titles, shared_yaxes=True, subplot_titles=sp_titles)

    for i, g in enumerate(g_sel):

        sl = True if i < 1 else False

        # Unpack output
        _, signals, timepoints, plv, plv_emp, r_plv, regionLabels, simLength, transient = output[i]

        freqs = np.arange(len(timepoints) / 2)
        freqs = freqs / (simLength - transient / 1000)

        cmap = px.colors.qualitative.Plotly
        for ii, signal in enumerate(signals):
            # Timeseries
            fig.add_trace(go.Scatter(x=timepoints[:5000] / 1000, y=signal[:5000], name=regionLabels[ii],
                                     legendgroup=regionLabels[ii],
                                     showlegend=sl, marker_color=cmap[ii % len(cmap)]), row=1, col=i + 2)
            # Spectra
            freqRange = [2, 40]
            fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
            fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
            fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
            fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                     marker_color=cmap[ii % len(cmap)], name=regionLabels[ii],
                                     legendgroup=regionLabels[ii], showlegend=False), row=2, col=i + 2)

        # Functional Connectivity
        fig.add_trace(go.Heatmap(z=plv, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4),
                                 colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=i + 2)

    # empirical FC matrices
    fig.add_trace(go.Heatmap(z=plv_emp, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4), legendgroup="",
                             colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=1)

    w_ = 800 if n_g < 3 else 1000
    fig.update_layout(legend=dict(yanchor="top", y=1.05, tracegroupgap=1),
                      template="plotly_white", height=900, width=w_)

    # Update layout
    for col in range(n_g + 1):  # +1 empirical column
        # first row
        idx = col + 1  # +1 to avoid 0 indexing in python
        if idx > 1:
            fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Time (s)"}
            if idx == 2:
                fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Voltage (mV)"}

        # second row
        idx = 1 * (n_g + 1) + (col + 1)  # +1 to avoid 0 indexing in python
        if idx > 1 + n_g:
            fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Frequency (Hz)"}
            if idx == 3 + n_g:
                fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Power (dB)"}

        # third row
        # idx = 2 * n_g+1 + (col+1)  # +1 to avoid 0 indexing in python
        # fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
        # fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}

    if mode == "html":
        pio.write_html(fig, file=folder + "/g_explore.html", auto_open=True)
    elif mode == "png":
        pio.write_image(fig, file=folder + "/g_explore" + str(time.time()) + ".png", engine="kaleido")
    elif mode == "svg":
        pio.write_image(fig, file=folder + "/g_explore.svg", engine="kaleido")

    elif mode == "inline":
        plotly.offline.iplot(fig)


def braidPlot(data, conn, mode="surface", rho_vals=None, title="new", folder="figures"):
    # Regions in Braak stages for TAU
    rI = ["ctx-rh-entorhinal", "ctx-lh-entorhinal"]
    rII = ["Right-Hippocampus", "Left-Hippocampus"]
    rIII = ["ctx-rh-parahippocampal", "ctx-lh-parahippocampal"]
    rIV = ["ctx-rh-caudalanteriorcingulate", "ctx-rh-rostralanteriorcingulate",
           "ctx-lh-caudalanteriorcingulate", "ctx-lh-rostralanteriorcingulate"]
    rV = ["ctx-rh-cuneus", 'ctx-rh-pericalcarine', 'ctx-rh-lateraloccipital', 'ctx-rh-lingual',
          "ctx-lh-cuneus", 'ctx-lh-pericalcarine', 'ctx-lh-lateraloccipital', 'ctx-lh-lingual']

    if mode == "diagram":

        tau_dyn = np.asarray(data[1])[:, 3, :]
        tau_dyn_perc = tau_dyn / tau_dyn[-1, :] * 100

        rxs = [[tau_dyn_perc[:, list(conn.region_labels).index(roi)] for roi in rx] for rx in [rI, rII, rIII, rIV, rV]]
        rxs_avg = np.asarray([np.average(np.asarray(rx), axis=0) for rx in rxs])
        # For each percentage; tell me in what time it was rebased.
        percxRx_TIME = np.asarray([[np.min(np.argwhere(rx >= perc)) for rx in rxs_avg] for perc in range(1, 100)])
        # columns=["rI", "rII", "rIII", "rIV", "rV"])
        # Tell me the rx order based on perc_time
        percxRx_ORDER_braiddiag = np.asarray([np.argsort(np.argsort(row)) for row in percxRx_TIME])
        percxRx_ORDER_braiddiag = percxRx_ORDER_braiddiag.astype(float)

        # Take care with equalities in perc_time that are sorted randomly
        for i, row in enumerate(percxRx_TIME):
            if len(set(row)) < 5:
                for val in set(row):
                    if len(row[row == val]) > 1:
                        percxRx_ORDER_braiddiag[i, row == val] = \
                            np.tile(np.average(percxRx_ORDER_braiddiag[i, row == val]), (len(row[row == val])))

        # Plotting
        cmap = px.colors.qualitative.Set2
        fig = make_subplots(rows=2, cols=1, subplot_titles=["Temporal dynamics: first time surpassing percentage",
                                                            "Braid diagram"])
        for i in range(percxRx_ORDER_braiddiag.shape[1]):
            fig.add_trace(go.Scatter(x=percxRx_TIME[:, i], y=list(range(1, 100)), legendgroup="r" + str(i + 1),
                                     showlegend=False, line=dict(color=cmap[i])), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(1, 100)), y=percxRx_ORDER_braiddiag[:, i], name="r" + str(i + 1),
                                     legendgroup="r" + str(i + 1), showlegend=True, line=dict(color=cmap[i])), row=2,
                          col=1)

        fig.update_layout(xaxis1=dict(title="Timestep"), yaxis1=dict(title="% Concentration (M)"),
                          xaxis2=dict(title="Percentage (%)"), yaxis2=dict(title="ORDER<br>of % reaching"))
        pio.write_html(fig, file=folder + "/BraidDiagram_" + title + ".html", auto_open=True)

    elif mode == "surface":
        rnd_id = np.random.randint(0, len(rho_vals))
        patterns = []
        # for each value of rho
        for i, rho in enumerate(rho_vals):
            # compute the diagram
            tau_dyn = data[i, :, :]
            tau_dyn_perc = tau_dyn / tau_dyn[-1, :] * 100

            rxs = [[tau_dyn_perc[:, list(conn.region_labels).index(roi)] for roi in rx] for rx in
                   [rI, rII, rIII, rIV, rV]]
            rxs_avg = np.asarray([np.average(np.asarray(rx), axis=0) for rx in rxs])
            # For each percentage; tell me in what time it was rebased.
            percxRx_TIME = np.asarray([[np.min(np.argwhere(rx >= perc)) for rx in rxs_avg] for perc in range(1, 100)])
            # columns=["rI", "rII", "rIII", "rIV", "rV"])
            # Tell me the rx order based on perc_time
            percxRx_ORDER_braiddiag = np.asarray([np.argsort(np.argsort(row)) for row in percxRx_TIME])
            percxRx_ORDER_braiddiag = percxRx_ORDER_braiddiag.astype(float)

            if rnd_id == i:
                diag = [percxRx_TIME, percxRx_ORDER_braiddiag]

            # Take care with equalities in perc_time that are sorted randomly
            for i, row in enumerate(percxRx_TIME):
                if len(set(row)) < 5:
                    for val in set(row):
                        if len(row[row == val]) > 1:
                            percxRx_ORDER_braiddiag[i, row == val] = \
                                np.tile(np.average(percxRx_ORDER_braiddiag[i, row == val]), (len(row[row == val])))

            # compact the diagram in a line with categorical patters
            patterns.append([str(row) for row in percxRx_ORDER_braiddiag])

        rho_patts = sorted(list(set([p for patt in patterns for p in set(patt)])))

        patt_int = np.asarray(patterns)
        corresp = []
        for i, pattern in enumerate(rho_patts):
            patt_int[np.where(patt_int == rho_patts[i])] = i
            corresp.append([i, pattern])

        patt_int = patt_int.astype(int)

        # Plotting
        cmap = px.colors.qualitative.Set2
        fig = make_subplots(rows=2, cols=2, specs=[[{}, {"rowspan": 2}], [{}, {}]],
                            subplot_titles=["Temporal dynamics - RANDOM rho (" + str(rho_vals[rnd_id]) + ")",
                                            "Braid surface",
                                            "Braid diagram- RANDOM rho (" + str(rho_vals[rnd_id]) + ")"])

        fig.add_trace(go.Heatmap(x=list(range(1, 100)), y=rho_vals, z=patt_int, colorscale="Turbo"), row=1, col=2)

        ## Add a random diagram as example
        percxRx_TIME, percxRx_ORDER_braiddiag = diag
        for i in range(percxRx_ORDER_braiddiag.shape[1]):
            fig.add_trace(go.Scatter(x=percxRx_TIME[:, i], y=list(range(1, 100)), legendgroup="r" + str(i + 1),
                                     showlegend=False, line=dict(color=cmap[i])), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(1, 100)), y=percxRx_ORDER_braiddiag[:, i], name="r" + str(i + 1),
                                     legendgroup="r" + str(i + 1), showlegend=True, line=dict(color=cmap[i])), row=2,
                          col=1)

        fig.update_layout(xaxis1=dict(title="Timestep"), yaxis1=dict(title="% Concentration (M)"),
                          xaxis2=dict(title="Percentage (%)"), yaxis2=dict(title="rho (diffusion factor)", type="log"),
                          xaxis3=dict(title="Percentage (%)"), yaxis3=dict(title="ORDER<br>of % reaching"),
                          legend=dict(orientation="h"))
        pio.write_html(fig, file=folder + "/BraidSurface_" + title + ".html", auto_open=True)

        corresp = pd.DataFrame(corresp, columns=["id", "pattern"])

        return corresp


def animateFC(data, conn, mode="3Dcortex", threshold=0.05, title="new", folder="figures"):

    # Define regions implicated in Functional analysis: not considering subcortical ROIs
    cortical_rois = ['ctx-lh-bankssts', 'ctx-rh-bankssts', 'ctx-lh-caudalanteriorcingulate',
                     'ctx-rh-caudalanteriorcingulate',
                     'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-rh-cuneus',
                     'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
                     'ctx-lh-fusiform', 'ctx-rh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
                     'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal', 'ctx-lh-insula', 'ctx-rh-insula',
                     'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                     'ctx-rh-lateraloccipital',
                     'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-rh-lingual',
                     'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-middletemporal',
                     'ctx-rh-middletemporal',
                     'ctx-lh-paracentral', 'ctx-rh-paracentral', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
                     'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-rh-parsorbitalis',
                     'ctx-lh-parstriangularis', 'ctx-rh-parstriangularis', 'ctx-lh-pericalcarine',
                     'ctx-rh-pericalcarine',
                     'ctx-lh-postcentral', 'ctx-rh-postcentral', 'ctx-lh-posteriorcingulate',
                     'ctx-rh-posteriorcingulate',
                     'ctx-lh-precentral', 'ctx-rh-precentral', 'ctx-lh-precuneus', 'ctx-rh-precuneus',
                     'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
                     'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
                     'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal', 'ctx-lh-superiorparietal',
                     'ctx-rh-superiorparietal',
                     'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'ctx-lh-supramarginal',
                     'ctx-rh-supramarginal',
                     'ctx-lh-temporalpole', 'ctx-rh-temporalpole', 'ctx-lh-transversetemporal',
                     'ctx-rh-transversetemporal']
    dmn_rois = [  # ROIs not in Gianlucas set, from cingulum bundle description
        'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
        'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal',
        'ctx-lh-insula', 'ctx-rh-insula',
        'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',  # A6. in Gianlucas DMN
        'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',  # A6. in Gianlucas DMN
        'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate',
        'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',  # A7. in Gianlucas DMN
        'ctx-lh-middletemporal', 'ctx-rh-middletemporal',  # A5. in Gianlucas DMN
        'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal',  # A4. in Gianlucas DMN
        'ctx-lh-superiorparietal', 'ctx-rh-superiorparietal',
        'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',  # A3. in Gianlucas DMN
        'ctx-lh-precuneus', 'ctx-rh-precuneus'  # A1. in Gianlucas DMN
    ]

    # Load SC labels.
    SClabs = list(conn.region_labels)
    if "cortex" in mode:
        SC_idx = [SClabs.index(roi) for roi in cortical_rois]
    elif "dmn" in mode:
        SC_idx = [SClabs.index(roi) for roi in dmn_rois]

    # 3d mode
    if "3D" in mode:

        regionLabels = conn.region_labels[SC_idx]
        weights = conn.weights[:, SC_idx][SC_idx]
        centres = conn.centres[SC_idx, :]

        # fig = make_subplots(rows=2, cols=1, specs=[[{"type": "surface"}],[{}]])
        fig = go.Figure()

        # Edges trace
        ## Filter edges to show: remove low connected nodes via thresholding
        edges_ids = list(combinations([i for i, roi in enumerate(regionLabels)], r=2))
        edges_ids = [(i, j) for i, j in edges_ids if weights[i, j] > threshold]

        ## Define [start, end, None] per coordinate and connection
        edges_x = [elem for sublist in [[centres[i, 0]] + [centres[j, 0]] + [None] for i, j in edges_ids] for elem in sublist]
        edges_y = [elem for sublist in [[centres[i, 1]] + [centres[j, 1]] + [None] for i, j in edges_ids] for elem in sublist]
        edges_z = [elem for sublist in [[centres[i, 2]] + [centres[j, 2]] + [None] for i, j in edges_ids] for elem in sublist]

        ## Define color per connection based on FC changes
        increaseFC = [data[1][i][3] - data[1][0][3] for i, t in enumerate(data[0])]
        increaseFC_norm = [[(increaseFC[ii][i, j] + 1)/2 for i, j in edges_ids] for ii, t in enumerate(data[0])]
        edges_color = [[px.colors.sample_colorscale("Jet", e)[0] for e in inc for i in range(3)] for inc in increaseFC_norm]

        fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip",
                                   line=dict(color=edges_color[0], width=3), opacity=0.6, name="Edges"))

        # Nodes trace
        ## Define size per degree
        degree, size_range = np.sum(weights, axis=1), [8, 25]
        size = ((degree - np.min(degree)) * (size_range[1]-size_range[0]) / (np.max(degree) - np.min(degree))) + size_range[0]

        ## Define color per power
        increasePOW = [data[1][i][2] - data[1][0][2] for i, t in enumerate(data[0])]
        increasePOW_norm = [[(increasePOW[ii][i] + np.max(increasePOW))/(2*np.max(increasePOW))
                             for i, roi in enumerate(regionLabels)] for ii, t in enumerate(data[0])]
        nodes_color = [[px.colors.sample_colorscale("Jet", e)[0] for e in inc] for inc in increasePOW_norm]

        # Create text labels per ROI
        hovertext3d = [["<b>" + roi + "</b>"
                        "<br>Power (dB) " + str(round(data[1][ii][2][i], 5)) +
                        "<br>Frequency (Hz) " + str(round(data[1][ii][1][i], 5)) +
                        "<br><br>Power increase (dB) " + str(round(data[1][ii][2][i] - data[1][0][2][i], 5))
                        for i, roi in enumerate(regionLabels)] for ii, t in enumerate(data[0])]

        fig.add_trace(go.Scatter3d(x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="Nodes",
                                   marker=dict(size=size, color=nodes_color[0], opacity=1, line=dict(color="gray", width=2))))

        # Update frames
        fig.update(frames=[go.Frame(data=[
            go.Scatter3d(line=dict(color=edges_color[i])),
            go.Scatter3d(hovertext=hovertext3d[i], marker=dict(color=nodes_color[i]))],
            traces=[0, 1], name=str(t)) for i, t in enumerate(data[0])])

        # CONTROLS : Add sliders and buttons
        fig.update_layout(
            template="plotly_white", legend=dict(x=0.8, y=0.5),
            scene=dict(xaxis=dict(title="Sagital axis<br>(R-L)"),
                       yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                       zaxis=dict(title="Horizontal axis<br>(sup-inf)")),
            sliders=[dict(
                steps=[
                    dict(method='animate', args=[[str(t)], dict(mode="immediate", frame=dict(duration=250, redraw=True,
                                                                                             easing="cubic-in-out"),
                                                                transition=dict(duration=0))], label=str(t)) for i, t
                    in enumerate(data[0])],
                transition=dict(duration=0), x=0.15, xanchor="left", y=1.1,
                currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
                len=0.8, tickcolor="white")],
            updatemenus=[dict(type="buttons", showactive=False, y=1.05, x=0, xanchor="left",
                              buttons=[
                                  dict(label="Play", method="animate",
                                       args=[None,
                                             dict(frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                                  transition=dict(duration=0),
                                                  fromcurrent=True, mode='immediate')]),
                                  dict(label="Pause", method="animate",
                                       args=[[None],
                                             dict(frame=dict(duration=250, redraw=False, easing="cubic-in-out"),
                                                  transition=dict(duration=0),
                                                  mode="immediate")])])])

        pio.write_html(fig, file=folder + "/IncreaseFC_3Dbrain_" + title + ".html", auto_open=True, auto_play=False)







## OlD stuff

def simulate(subj, model, g, s, g_wc=None, p_th=0.12, sigma=0.022, t=10):
    # Prepare simulation parameters
    simLength = t * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 2000  # ms

    tic = time.time()

    # STRUCTURAL CONNECTIVITY      #########################################

    conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
    conn.weights = conn.scaled_weights(mode="tract")
    conn.speed = np.array([s])

    # Define regions implicated in Functional analysis: not considering subcortical ROIs
    cortical_rois = ['ctx-lh-bankssts', 'ctx-rh-bankssts', 'ctx-lh-caudalanteriorcingulate',
                     'ctx-rh-caudalanteriorcingulate',
                     'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-rh-cuneus',
                     'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
                     'ctx-lh-fusiform', 'ctx-rh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
                     'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal', 'ctx-lh-insula', 'ctx-rh-insula',
                     'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                     'ctx-rh-lateraloccipital',
                     'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-rh-lingual',
                     'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-middletemporal',
                     'ctx-rh-middletemporal',
                     'ctx-lh-paracentral', 'ctx-rh-paracentral', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
                     'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-rh-parsorbitalis',
                     'ctx-lh-parstriangularis', 'ctx-rh-parstriangularis', 'ctx-lh-pericalcarine',
                     'ctx-rh-pericalcarine',
                     'ctx-lh-postcentral', 'ctx-rh-postcentral', 'ctx-lh-posteriorcingulate',
                     'ctx-rh-posteriorcingulate',
                     'ctx-lh-precentral', 'ctx-rh-precentral', 'ctx-lh-precuneus', 'ctx-rh-precuneus',
                     'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
                     'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
                     'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal', 'ctx-lh-superiorparietal',
                     'ctx-rh-superiorparietal',
                     'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'ctx-lh-supramarginal',
                     'ctx-rh-supramarginal',
                     'ctx-lh-temporalpole', 'ctx-rh-temporalpole', 'ctx-lh-transversetemporal',
                     'ctx-rh-transversetemporal']

    #  Load FC labels, transform to SC format; check if match SC.
    FClabs = list(np.loadtxt(data_folder + "FC_matrices/" + subj + "_roi_labels_rms.txt", dtype=str))
    FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    FC_cortex_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois

    # load SC labels.
    SClabs = list(conn.region_labels)
    SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

    #   NEURAL MASS MODEL  &  COUPLING FUNCTION   #########################################################
    sigma_array = np.asarray([sigma if 'Thal' in roi else 0 for roi in conn.region_labels])
    p_array = np.asarray([p_th if 'Thal' in roi else 0.09 for roi in conn.region_labels])

    if model == "jrd":  # JANSEN-RIT-DAVID
        # Parameters edited from David and Friston (2003).
        m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                               tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                               He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                               tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                               w=np.array([0.8]), c=np.array([135.0]),
                               c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                               c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                               v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                               p=np.array([p_array]), sigma=np.array([sigma_array]))

        # Remember to hold tau*H constant.
        m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
        m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

        coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=np.array([0.8]), e0=np.array([0.005]),
                                                v0=np.array([6.0]), r=np.array([0.56]))

    elif model == "jrwc":  # JANSEN-RIT(cx) + WILSON-COWAN(th)

        jrMask_wc = [[False] if 'Thal' in roi else [True] for roi in conn.region_labels]

        m = JansenRit_WilsonCowan(
            # Jansen-Rit nodes parameters. From Stefanovski et al. (2019)
            He=np.array([3.25]), Hi=np.array([22]),
            tau_e=np.array([10]), tau_i=np.array([20]),
            c=np.array([135.0]), p=np.array([p_array]),
            c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
            c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
            v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
            # Wilson-Cowan nodes parameters. From Abeysuriya et al. (2018)
            P=np.array([0.31]), sigma=np.array([sigma_array]), Q=np.array([0]),
            c_ee=np.array([3.25]), c_ei=np.array([2.5]),
            c_ie=np.array([3.75]), c_ii=np.array([0]),
            tau_e_wc=np.array([10]), tau_i_wc=np.array([20]),
            a_e=np.array([4]), a_i=np.array([4]),
            b_e=np.array([1]), b_i=np.array([1]),
            c_e=np.array([1]), c_i=np.array([1]),
            k_e=np.array([1]), k_i=np.array([1]),
            r_e=np.array([0]), r_i=np.array([0]),
            theta_e=np.array([0]), theta_i=np.array([0]),
            alpha_e=np.array([1]), alpha_i=np.array([1]),
            # JR mask | WC mask
            jrMask_wc=np.asarray(jrMask_wc))

        m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])

        coup = coupling.SigmoidalJansenRit_Linear(
            a=np.array([g]), e0=np.array([0.005]), v0=np.array([6]), r=np.array([0.56]),
            # Jansen-Rit Sigmoidal coupling
            a_linear=np.asarray([g_wc]),  # Wilson-Cowan Linear coupling
            jrMask_wc=np.asarray(jrMask_wc))  # JR mask | WC mask

    else:  # JANSEN-RIT
        # Parameters from Stefanovski 2019.
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([p_array]), sigma=np.array([sigma_array]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))

    # OTHER PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    # print("Simulating %s (%is)  ||  PARAMS: g%i sigma%0.2f" % (model, simLength / 1000, g, 0.022))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    if model == "jrd":
        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
    else:
        raw_data = output[0][1][transient:, 0, :, 0].T
    raw_time = output[0][0][transient:]

    # Further analysis on Cortical signals
    raw_data = raw_data[SC_cortex_idx, :]
    regionLabels = conn.region_labels[SC_cortex_idx]

    fftp = FFTpeaks(raw_data, simLength - transient)

    # PLOTs :: Signals and spectra
    # timeseries_spectra(raw_data[:], simLength, transient, regionLabels, mode="inline", freqRange=[2, 40], opacity=1)

    bands = [["3-alpha"], [(8, 12)]]
    # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

    for b in range(len(bands[0])):
        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut, verbose=False)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals", verbose=False)

        # Obtain Analytical signal
        efPhase = list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.angle(analyticalSignal))
            efEnvelope.append(np.abs(analyticalSignal))

        # Check point
        # from toolbox import timeseriesPlot, plotConversions
        # regionLabels = conn.region_labels
        # timeseriesPlot(raw_data, raw_time, regionLabels)
        # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

        # CONNECTIVITY MEASURES
        ## PLV and plot
        plv = PLV(efPhase, verbose=False)

        # Load empirical data to make simple comparisons
        plv_emp = \
            np.loadtxt(data_folder + "FC_matrices/" + subj + "_" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
            FC_cortex_idx][
                FC_cortex_idx]

        # Comparisons
        t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
        t1[0, :] = plv[np.triu_indices(len(plv), 1)]
        t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
        plv_r = np.corrcoef(t1)[0, 1]

        # ## dynamical Functional Connectivity
        # # Sliding window parameters
        # window, step = 4, 2  # seconds
        #
        # ## dFC and plot
        # dFC = dynamic_fc(raw_data, samplingFreq, transient, window, step, "PLV")
        #
        # dFC_emp = np.loadtxt(data_folder + "FC_matrices/" + subj + "_" + bands[0][b] + "_dPLV4s_rms.txt")
        #
        # # Compare dFC vs dFC_emp
        # t2 = np.zeros(shape=(2, len(dFC) ** 2 // 2 - len(dFC) // 2))
        # t2[0, :] = dFC[np.triu_indices(len(dFC), 1)]
        # t2[1, :] = dFC_emp[np.triu_indices(len(dFC), 1)]
        # dFC_ksd = scipy.stats.kstest(dFC[np.triu_indices(len(dFC), 1)], dFC_emp[np.triu_indices(len(dFC), 1)])[0]

        ## PLOTs :: PLV + dPLV
        # print("REPORT_ \nrPLV = %0.2f " % (plv_r))

    # print("SIMULATION REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return raw_data, raw_time, fftp, plv, plv_emp, plv_r, regionLabels, simLength, transient, time.time() - tic


def animate_propagation_v2(output, conn, timeref=True):
    # Create text labels per ROI
    hovertext3d = [["<b>" + roi + "</b><br>"
                    + str(round(output[1][ii][0, i], 5)) + "(M) a-beta<br>"
                    + str(round(output[1][ii][1, i], 5)) + "(M) a-beta toxic <br>"
                    + str(round(output[1][ii][2, i], 5)) + "(M) pTau <br>"
                    + str(round(output[1][ii][3, i], 5)) + "(M) pTau toxic <br>"
                    for i, roi in enumerate(conn.region_labels)] for ii, t in enumerate(output[0])]

    sz_ab, sz_t = 22, 7  # Different sizes for AB and pT nodes

    ## ADD INITIAL TRACE for 3dBrain - t0
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {}]], column_widths=[0.6, 0.4],
                        subplot_titles=(['<b>Protein accumulation dynamics</b>', '']))

    # Add trace for AB
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="AB", showlegend=True, legendgroup="AB",
                               marker=dict(size=np.abs(output[1][0][0, :]) * sz_ab, color=output[1][0][0, :],
                                           opacity=0.5, cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="YlOrBr")), row=1, col=1)
    # Add trace for ABt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="AB toxic", showlegend=True,
                               legendgroup="AB toxic",
                               marker=dict(size=np.abs(output[1][0][1, :]) * sz_ab, color=output[1][0][1, :],
                                           opacity=0.5, cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="YlOrRd")), row=1, col=1)
    # Add trace for TAU
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="pTAU", showlegend=True,
                               legendgroup="pTAU",
                               marker=dict(size=np.abs(output[1][0][2, :]) * sz_t, color=output[1][0][2, :], opacity=1,
                                           cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="BuPu", symbol="diamond")),
                  row=1, col=1)
    # Add trace for TAUt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="pTAU toxic", showlegend=True,
                               legendgroup="pTAU toxic",
                               marker=dict(size=np.abs(output[1][0][3, :]) * sz_t, color=output[1][0][3, :], opacity=1,
                                           cmax=2, cmin=0,
                                           line=dict(color="grey", width=1), colorscale="Greys", symbol="diamond")),
                  row=1, col=1)

    ## ADD INITIAL TRACE for lines
    sim_pet_avg = np.average(np.asarray(output[1]), axis=2)
    cmap_s, cmap_d = px.colors.qualitative.Set2, px.colors.qualitative.Dark2

    # Add static lines
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 0], mode="lines", name="AB", legendgroup="AB",
                             line=dict(color=cmap_s[1], width=3), showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 1], mode="lines", name="AB toxic", legendgroup="AB toxic",
                             line=dict(color=cmap_s[3], width=3), showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 2], mode="lines", name="pTAU", legendgroup="pTAU",
                             line=dict(color=cmap_s[2], width=3), showlegend=True), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=output[0], y=sim_pet_avg[:, 3], mode="lines", name="pTAU toxic", legendgroup="pTAU toxic",
                   line=dict(color=cmap_s[7], width=3), showlegend=True), row=1, col=2)

    # Add dynamic reference - t0
    if timeref:
        fig.add_trace(go.Scatter(x=[output[0][0]], y=[sim_pet_avg[0, 0]], mode="markers", legendgroup="AB",
                                 marker=dict(color=cmap_d[1]), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[output[0][0]], y=[sim_pet_avg[0, 1]], mode="markers", legendgroup="AB toxic",
                                 marker=dict(color=cmap_d[3]), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[output[0][0]], y=[sim_pet_avg[0, 2]], mode="markers", legendgroup="pTAU",
                                 marker=dict(color=cmap_d[2]), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[output[0][0]], y=[sim_pet_avg[0, 3]], mode="markers", legendgroup="pTAU toxic",
                                 marker=dict(color=cmap_d[7]), showlegend=False), row=1, col=2)

    # fig.show("browser")

    ## ADD FRAMES - t[1:end]
    fig.update(frames=[go.Frame(data=[
        go.Scatter3d(hovertext=hovertext3d[i],
                     marker=dict(size=np.abs(output[1][i][0, :]) * sz_ab, color=output[1][i][0, :])),
        go.Scatter3d(hovertext=hovertext3d[i],
                     marker=dict(size=np.abs(output[1][i][1, :]) * sz_ab, color=output[1][i][1, :])),
        go.Scatter3d(hovertext=hovertext3d[i],
                     marker=dict(size=np.abs(output[1][i][2, :]) * sz_t, color=output[1][i][2, :])),
        go.Scatter3d(hovertext=hovertext3d[i],
                     marker=dict(size=np.abs(output[1][i][3, :]) * sz_t, color=output[1][i][3, :])),

        go.Scatter(x=[output[0][i]], y=[sim_pet_avg[i, 0]]),
        go.Scatter(x=[output[0][i]], y=[sim_pet_avg[i, 1]]),
        go.Scatter(x=[output[0][i]], y=[sim_pet_avg[i, 2]]),
        go.Scatter(x=[output[0][i]], y=[sim_pet_avg[i, 3]])

    ],
        traces=[0, 1, 2, 3, 8, 9, 10, 11], name=str(i)) for i, t in enumerate(output[0])])

    # CONTROLS : Add sliders and buttons
    fig.update_layout(
        template="plotly_white", legend=dict(x=1.05, y=1.1),
        scene=dict(xaxis=dict(title="Sagital axis<br>(L-R)"),
                   yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                   zaxis=dict(title="Horizontal axis<br>(inf-sup)")),
        yaxis=dict(range=[0, 1]),
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(i)], dict(mode="immediate", frame=dict(duration=100, redraw=True,
                                                                                            easing="cubic-in-out"),
                                                               transition=dict(duration=300))], label=str(t)) for i, t
                   in enumerate(output[0])],
            transition=dict(duration=100), x=0.15, xanchor="left", y=1.4,
            currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
            len=0.8, tickcolor="white")],
        updatemenus=[dict(type="buttons", showactive=False, y=1.35, x=0, xanchor="left",
                          buttons=[
                              dict(label="Play", method="animate",
                                   args=[None,
                                         dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                              transition=dict(duration=300),
                                              fromcurrent=True, mode='immediate')]),
                              dict(label="Pause", method="animate",
                                   args=[[None],
                                         dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                              transition=dict(duration=300),
                                              mode="immediate")])])])
    fig.show("browser")


def animate_propagation_v3(output, corrs, refgroups, reftype, conn, timeref=True):
    # Create text labels per ROI
    hovertext3d = [["<b>" + roi + "</b><br>"
                    + str(round(output[1][ii][0, i], 5)) + "(M) a-beta<br>"
                    + str(round(output[1][ii][1, i], 5)) + "(M) a-beta toxic <br>"
                    + str(round(output[1][ii][2, i], 5)) + "(M) pTau <br>"
                    + str(round(output[1][ii][3, i], 5)) + "(M) pTau toxic <br>"
                    for i, roi in enumerate(conn.region_labels)] for ii, t in enumerate(output[0])]

    sz_ab, sz_t = 25, 10  # Different sizes for AB and pT nodes

    ## ADD INITIAL TRACE for 3dBrain - t0
    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2, "type": "surface"}, {}], [{}, {}]],
                        column_widths=[0.6, 0.4], shared_xaxes=True,
                        subplot_titles=(['<b>Protein accumulation dynamics</b>', '', '', 'Correlations (emp-sim)']))

    # Add trace for AB + ABt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="AB", showlegend=True, legendgroup="AB",
                               marker=dict(size=(np.abs(output[1][0][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                           cmax=0.5, cmin=-0.25,
                                           color=np.abs(output[1][0][1, :]) / np.abs(output[1][0][0, :]), opacity=0.5,
                                           line=dict(color="grey", width=1), colorscale="YlOrBr")), row=1, col=1)

    # Add trace for TAU + TAUt
    fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="TAU", showlegend=True, legendgroup="TAU",
                               marker=dict(size=(np.abs(output[1][0][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                           cmax=0.5, cmin=-0.25,
                                           color=np.abs(output[1][0][3, :]) / np.abs(output[1][0][2, :]), opacity=1,
                                           line=dict(color="grey", width=1), colorscale="BuPu", symbol="diamond")),
                  row=1, col=1)

    ## ADD INITIAL TRACE for lines
    sim_pet_avg = np.average(np.asarray(output[1]), axis=2)

    if timeref:
        # Add dynamic reference - t0
        min_lp, max_lp = np.min(sim_pet_avg) - 0.15, np.max(sim_pet_avg) + 0.15
        fig.add_trace(
            go.Scatter(x=[output[0][0], output[0][0]], y=[min_lp, max_lp], mode="lines", legendgroup="timeref",
                       line=dict(color="black", width=1), showlegend=False), row=1, col=2)

        min_r, max_r = np.min(corrs) - 0.15, np.max(corrs) + 0.15
        fig.add_trace(go.Scatter(x=[output[0][0], output[0][0]], y=[min_r, max_r], mode="lines", legendgroup="timeref",
                                 line=dict(color="black", width=1), showlegend=False), row=2, col=2)

    # Add static lines - PET proteins concentrations
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 0], mode="lines", name="AB", legendgroup="AB",
                             line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7, showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 1], mode="lines", name="AB toxic", legendgroup="AB",
                             line=dict(color=px.colors.sequential.YlOrBr[5], width=3), opacity=0.7, showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 2], mode="lines", name="TAU", legendgroup="TAU",
                             line=dict(color=px.colors.sequential.BuPu[3], width=3), opacity=0.7, showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 3], mode="lines", name="TAU toxic", legendgroup="TAU",
                             line=dict(color=px.colors.sequential.BuPu[5], width=3), opacity=0.7, showlegend=True),
                  row=1, col=2)

    # Add static lines - data correlations
    cmap_p = px.colors.qualitative.Pastel2
    for ii, group in enumerate(refgroups):
        if "PET" in reftype:
            fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 0], mode="lines", name=group + " - AB",
                                     legendgroup="corr",
                                     line=dict(color=cmap_p[ii], width=3), showlegend=True), row=2, col=2)

            fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 1], mode="lines", name=group + " - rTAU",
                                     legendgroup="corr",
                                     line=dict(color=cmap_p[ii], width=2, dash="dash"),  # visible="legendonly",
                                     showlegend=True), row=2, col=2)

        else:
            fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :], mode="lines", name=group + " - r" + reftype,
                                     legendgroup="corr",
                                     line=dict(color=cmap_p[ii], width=3), showlegend=True), row=2, col=2)

    # fig.show("browser")

    ## ADD FRAMES - t[1:end]
    if timeref:

        fig.update(frames=[go.Frame(data=[
            go.Scatter3d(hovertext=hovertext3d[i],
                         marker=dict(size=(np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                     color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

            go.Scatter3d(hovertext=hovertext3d[i],
                         marker=dict(size=(np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                     color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),

            go.Scatter(x=[output[0][i], output[0][i]]),
            go.Scatter(x=[output[0][i], output[0][i]])
        ],
            traces=[0, 1, 2, 3], name=str(i)) for i, t in enumerate(output[0])])
    else:
        fig.update(frames=[go.Frame(data=[
            go.Scatter3d(hovertext=hovertext3d[i],
                         marker=dict(size=np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :]) * sz_ab,
                                     color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

            go.Scatter3d(hovertext=hovertext3d[i],
                         marker=dict(size=np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :]) * sz_t,
                                     color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),
        ],
            traces=[0, 1], name=str(i)) for i, t in enumerate(output[0])])

    # CONTROLS : Add sliders and buttons
    fig.update_layout(
        template="plotly_white", legend=dict(x=1.05, y=0.55, tracegroupgap=40, groupclick="toggleitem"),
        scene=dict(xaxis=dict(title="Sagital axis<br>(L-R)"),
                   yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                   zaxis=dict(title="Horizontal axis<br>(inf-sup)")),
        xaxis1=dict(title="Time (Years)"), xaxis3=dict(title="Time (Years)"),
        yaxis1=dict(title="Concentration (M)"), yaxis3=dict(title="Pearson's corr"),
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(i)], dict(mode="immediate", frame=dict(duration=100, redraw=True,
                                                                                            easing="cubic-in-out"),
                                                               transition=dict(duration=300))], label=str(t)) for i, t
                   in enumerate(output[0])],
            transition=dict(duration=100), x=0.15, xanchor="left", y=1.4,
            currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
            len=0.8, tickcolor="white")],
        updatemenus=[dict(type="buttons", showactive=False, y=1.35, x=0, xanchor="left",
                          buttons=[
                              dict(label="Play", method="animate",
                                   args=[None,
                                         dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                              transition=dict(duration=300),
                                              fromcurrent=True, mode='immediate')]),
                              dict(label="Pause", method="animate",
                                   args=[[None],
                                         dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                              transition=dict(duration=300),
                                              mode="immediate")])])])

    pio.write_html(fig, file="figures/ProteinPropagation_&corr" + reftype + ".html", auto_open=True)


def correlations(output, refgroups, reftype="PETtoxic", band="3-alpha", plot=False, title="new"):
    # Here we wanna get a simplify array with (n_refcond, time) containing correlation values
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

    # PET correlations
    if ("PET" in reftype) and ("rel" not in reftype):

        #    ADNI PET DATA       ##########
        ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)

        # Check label order
        conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/HC-fam_aparc_aseg-mni_09c.zip")
        PETlabs = list(ADNI_AVG.columns[12:])
        PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]

        # loop over refgroups: ["CN", "SMC", "EMCI", "LMCI", "AD"]
        corr = []
        df_corr = pd.DataFrame()
        for j, group in enumerate(refgroups):

            AB_emp = np.squeeze(
                np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == group)].iloc[:, 12:]))
            AB_emp = AB_emp[PET_idx]

            TAU_emp = np.squeeze(
                np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == group)].iloc[:, 12:]))
            TAU_emp = TAU_emp[PET_idx]

            cond, frames1, frames2 = [], [], []
            for i in range(len(output[1])):
                if "toxic" in reftype:
                    ABt = output[1][i][1, :]
                    TAUt = output[1][i][3, :]
                    cond.append([np.corrcoef(AB_emp, ABt)[0, 1], np.corrcoef(TAU_emp, TAUt)[0, 1]])

                    if plot:
                        # Create dataframe to plot dynamically
                        df_corr = df_corr.append(
                            pd.DataFrame(
                                [[output[0][i]] * len(ABt), [group] * len(ABt), ["ABt"] * len(ABt), ABt, AB_emp,
                                 conn.region_labels]).transpose())
                        df_corr = df_corr.append(
                            pd.DataFrame([[output[0][i]] * len(TAUt), [group] * len(TAUt), ["TAUt"] * len(TAUt), TAUt,
                                          TAU_emp, conn.region_labels]).transpose())

                else:
                    sumAB_ABt = output[1][i][0, :] + output[1][i][1, :]
                    sumTAU_TAUt = output[1][i][2, :] + output[1][i][3, :]
                    cond.append([np.corrcoef(AB_emp, sumAB_ABt)[0, 1], np.corrcoef(TAU_emp, sumTAU_TAUt)[0, 1]])

                    if plot:
                        # Create dataframe to plot dynamically
                        df_corr = df_corr.append(
                            pd.DataFrame([[output[0][i]] * len(sumAB_ABt), [group] * len(sumAB_ABt),
                                          ["sumAB_ABt"] * len(sumAB_ABt), sumAB_ABt, AB_emp,
                                          conn.region_labels]).transpose())
                        df_corr = df_corr.append(
                            pd.DataFrame([[output[0][i]] * len(sumTAU_TAUt), [group] * len(sumTAU_TAUt),
                                          ["sumTAU_TAUt"] * len(sumTAU_TAUt), sumTAU_TAUt, TAU_emp,
                                          conn.region_labels]).transpose())

            corr.append(cond)

        if plot is not False:
            add_space = 0.05
            print("CORRELATIONS  _Plotting animation - wait patiently")
            df_corr.columns = ["time", "group", "pet", "sim", "emp", "roi"]
            if "m" in plot:
                fig = px.scatter(df_corr, x="emp", y="sim", facet_row="pet", facet_col="group", animation_frame="time",
                                 hover_name="roi")
                fig.update_layout(title=title + " _" + reftype, template="plotly_white",
                                  yaxis1=dict(range=[min(df_corr["sim"].values) - add_space,
                                                     max(df_corr["sim"].values) + add_space]))
                pio.write_html(fig, "figures/sScatterCorr" + reftype + "_" + title + ".html", auto_open=True)

            if "c" in plot:
                df_sub = df_corr.loc[df_corr["group"] == "AD"]
                fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time",
                                 hover_name="roi", color="roi")
                fig.update_layout(title=title + " _" + reftype, template="plotly_white",
                                  yaxis1=dict(
                                      range=[min(df_corr["sim"].values) - add_space,
                                             max(df_corr["sim"].values) + add_space]))
            if "s" in plot:
                df_sub = df_corr.loc[df_corr["group"] == "AD"]
                fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time", hover_name="roi")
                fig.update_layout(title=title + " _" + reftype, template="plotly_white",
                                  yaxis1=dict(range=[min(df_corr["sim"].values) - add_space,
                                                     max(df_corr["sim"].values) + add_space]))

                pio.write_html(fig, "figures/cScatterCorr" + reftype + "_" + title + ".html", auto_open=True)
        return np.array(corr)

    # PET relative correlations
    elif ("PET" in reftype) and ("rel" in reftype):

        #    ADNI PET DATA       ##########
        ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupREL_2PrevStage.csv", index_col=0)

        # Check label order
        conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/HC-fam_aparc_aseg-mni_09c.zip")
        PETlabs = list(ADNI_AVG.columns[7:])
        PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]

        # loop over refgroups: ["CN", "SMC", "EMCI", "LMCI", "AD"]
        corr = []
        df_corr = pd.DataFrame()
        for j, group in enumerate(refgroups):

            AB_emp = np.squeeze(
                np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Transition"] == group)].iloc[:, 7:]))
            AB_emp = AB_emp[PET_idx]

            TAU_emp = np.squeeze(
                np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Transition"] == group)].iloc[:, 7:]))
            TAU_emp = TAU_emp[PET_idx]

            cond = []

            dABt = np.asarray([np.asarray(output[1])[i + 1, 1, :] - np.asarray(output[1])[i, 1, :] for i in
                               range(len(output[1]) - 1)])
            dTAUt = np.asarray([np.asarray(output[1])[i + 1, 3, :] - np.asarray(output[1])[i, 3, :] for i in
                                range(len(output[1]) - 1)])

            for i in range(len(dABt)):
                if "toxic" in reftype:
                    ## TODO dont take state vars but derivatives

                    cond.append([np.corrcoef(AB_emp, dABt[i, :])[0, 1], np.corrcoef(TAU_emp, dTAUt[i, :])[0, 1]])

                    if plot:
                        # Create dataframe to plot
                        df_corr = df_corr.append(
                            pd.DataFrame([[output[0][i]] * len(dABt[i, :]), [group] * len(dABt[i, :]),
                                          ["ABt"] * len(dABt[i, :]), dABt[i, :], AB_emp,
                                          conn.region_labels]).transpose())
                        df_corr = df_corr.append(
                            pd.DataFrame([[output[0][i]] * len(dTAUt[i, :]), [group] * len(dTAUt[i, :]),
                                          ["TAUt"] * len(dTAUt[i, :]), dTAUt[i, :],
                                          TAU_emp, conn.region_labels]).transpose())

                else:
                    sumAB_ABt = output[1][i][0, :] + output[1][i][1, :]
                    sumTAU_TAUt = output[1][i][2, :] + output[1][i][3, :]
                    cond.append([np.corrcoef(AB_emp, sumAB_ABt)[0, 1], np.corrcoef(TAU_emp, sumTAU_TAUt)[0, 1]])

                    if plot:
                        # Create dataframe to plot dynamically
                        df_corr = df_corr.append(
                            pd.DataFrame([[output[0][i]] * len(sumAB_ABt), [group] * len(sumAB_ABt),
                                          ["sumAB_ABt"] * len(sumAB_ABt), sumAB_ABt, AB_emp,
                                          conn.region_labels]).transpose())
                        df_corr = df_corr.append(
                            pd.DataFrame([[output[0][i]] * len(sumTAU_TAUt), [group] * len(sumTAU_TAUt),
                                          ["sumTAU_TAUt"] * len(sumTAU_TAUt), sumTAU_TAUt, TAU_emp,
                                          conn.region_labels]).transpose())

            corr.append(cond)

        if plot is not False:
            print("CORRELATIONS  _Plotting animation - wait patiently")
            df_corr.columns = ["time", "transition", "pet", "sim", "emp", "roi"]
            if "m" in plot:
                fig = px.scatter(df_corr, x="emp", y="sim", facet_row="pet", facet_col="transition",
                                 animation_frame="time",
                                 hover_name="roi")
                fig.update_layout(title=title + " _" + reftype, template="plotly_white")
                pio.write_html(fig, "figures/sScatterCorr" + reftype + "_" + title + ".html", auto_open=True)

            if "c" in plot:
                df_sub = df_corr.loc[df_corr["transition"] == "LMCI-AD"]
                fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time",
                                 hover_name="roi", color="roi")
                fig.update_layout(title=title + " _" + reftype, template="plotly_white")
            if "s" in plot:
                df_sub = df_corr.loc[df_corr["transition"] == "LMCI-AD"]
                fig = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time", hover_name="roi")
                fig.update_layout(title=title + " _" + reftype, template="plotly_white")

                pio.write_html(fig, "figures/cScatterCorr" + reftype + "_" + title + ".html", auto_open=True)
        return np.array(corr)

    # PLV correlations
    elif "PLV" in reftype:

        # loop over refgroups
        corr = []
        for group in refgroups:
            plv_emp = np.loadtxt(data_folder + "FC_matrices/" + group + "_" + band + "_plv_rms.txt", delimiter=',')

            t1 = np.zeros(shape=(2, len(plv_emp) ** 2 // 2 - len(plv_emp) // 2))
            t1[0, :] = plv_emp[np.triu_indices(len(plv_emp), 1)]

            cond = []
            for i in range(len(output[2])):
                # Comparisons
                t1[1, :] = output[2][i][np.triu_indices(len(plv_emp), 1)]
                cond.append(np.corrcoef(t1)[0, 1])

            corr.append(cond)

        return np.array(corr)

    # SC correlations
    elif "SC" in reftype:

        # loop over refgroups
        corr = []
        for group in refgroups:
            conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + group + "_aparc_aseg-mni_09c.zip")
            conn.weights = conn.scaled_weights(mode="tract")

            t1 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
            t1[0, :] = conn.weights[np.triu_indices(len(conn.region_labels), 1)]

            # correlate and save
            cond = []
            for i in range(len(output[3])):
                t1[1, :] = output[3][i][np.triu_indices(len(conn.region_labels), 1)]
                cond.append(np.corrcoef(t1)[0, 1])

            corr.append(cond)

            return np.array(corr)

