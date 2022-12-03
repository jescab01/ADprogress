
import time
import numpy as np
import scipy.signal
import scipy.stats

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from mpi4py import MPI
import datetime


def adpg_parallel(params_):
    result = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

        import sys
        sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
        from toolbox.fft import multitapper
        from toolbox.signals import epochingTool
        from toolbox.fc import PLV
        from toolbox.dynamics import dynamic_fc, kuramoto_order

    ## Folder structure - CLUSTER
    else:
        wd = "/home/t192/t192950/mpi/"
        data_folder = wd + "ADprogress_data/"

        import sys
        sys.path.append(wd)
        from toolbox.fft import multitapper
        from toolbox.signals import epochingTool
        from toolbox.fc import PLV
        from toolbox.dynamics import dynamic_fc, kuramoto_order


    # Prepare simulation parameters
    simLength = 10 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 2000  # ms

    for ii, set in enumerate(params_):

        tic = time.time()
        print("Rank %i out of %i  ::  %i/%i " % (rank, size, ii + 1, len(params_)))

        print(set)
        emp_subj, model, g, s, r = set

        # STRUCTURAL CONNECTIVITY      #########################################
        # Use "pass" for subcortical (thalamus) while "end" for cortex
        # based on [https://groups.google.com/g/dsi-studio/c/-naReaw7T9E/m/7a-Y1hxdCAAJ]

        conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + emp_subj + "_aparc_aseg-mni_09c.zip")
        conn.weights = conn.scaled_weights(mode="tract")
        conn.speed = np.array([s])

        # Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
        cortical_rois = ['ctx-lh-bankssts', 'ctx-rh-bankssts', 'ctx-lh-caudalanteriorcingulate',
                         'ctx-rh-caudalanteriorcingulate',
                         'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-rh-cuneus',
                         'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
                         'ctx-lh-fusiform', 'ctx-rh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
                         'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal', 'ctx-lh-insula', 'ctx-rh-insula',
                         'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                         'ctx-rh-lateraloccipital',
                         'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal', 'ctx-lh-lingual',
                         'ctx-rh-lingual',
                         'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-middletemporal',
                         'ctx-rh-middletemporal',
                         'ctx-lh-paracentral', 'ctx-rh-paracentral', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
                         'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsorbitalis',
                         'ctx-rh-parsorbitalis',
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
        FClabs = list(np.loadtxt(data_folder + "FC_matrices/" + emp_subj + "_roi_labels_rms.txt", dtype=str))
        FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
        FC_cortex_idx = [FClabs.index(roi) for roi in
                         cortical_rois]  # find indexes in FClabs that matches cortical_rois

        # load SC labels.
        SClabs = list(conn.region_labels)
        SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]


        # NEURAL MASS MODEL    #########################################################

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
                                   p=np.array([0.22]), sigma=np.array([0.022]))

            # Remember to hold tau*H constant.
            m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
            m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

        else:  # JANSEN-RIT
            # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
            m = JansenRit1995(He=np.array([3.5]), Hi=np.array([22]),
                              tau_e=np.array([10]), tau_i=np.array([16]),
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array([0.1085]), sigma=np.array([0]),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

        # COUPLING FUNCTION   #########################################
        if model == "jrd":
            coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
        else:
            coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))


        # OTHER PARAMETERS   ###
        # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
        # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
        integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

        mon = (monitors.Raw(),)

        print("Simulating %s (%is)  ||  PARAMS: g%0.2f s%0.2f" % (model, simLength / 1000, g, s))

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
        regionLabels = conn.region_labels

        # Extract signals of interest
        raw_data = raw_data[SC_cortex_idx, :]

        # Save min/max(signal) for bifurcation
        max_cx = np.average(np.array([max(signal) for i, signal in enumerate(raw_data)]))
        min_cx = np.average(np.array([min(signal) for i, signal in enumerate(raw_data)]))


        # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
        _, _, IAF, module, band_module = multitapper(raw_data, samplingFreq, regionLabels, peaks=True)

        # bands = [["3-alpha"], [(8, 12)]]
        bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]

        for b in range(len(bands[0])):
            (lowcut, highcut) = bands[1][b]

            # Band-pass filtering
            filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

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
            ## PLV
            plv = PLV(efPhase)

            # ## PLE - Phase Lag Entropy
            # ## PLE parameters - Phase Lag Entropy
            # tau_ = 25  # ms
            # m_ = 3  # pattern size
            # ple, patts = PLE(efPhase, tau_, m_, samplingFreq, subsampling=20)

            # Load empirical data to make simple comparisons
            plv_emp = \
                np.loadtxt(data_folder + "FC_matrices/" + emp_subj + "_" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
                FC_cortex_idx][
                    FC_cortex_idx]

            # Comparisons
            t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
            t1[0, :] = plv[np.triu_indices(len(plv), 1)]
            t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
            plv_r = np.corrcoef(t1)[0, 1]

            ## dynamical Functional Connectivity
            # Sliding window parameters
            # window, step = 4, 2  # seconds
            #
            # ## dFC
            # dFC = dynamic_fc(raw_data, samplingFreq, transient, window, step, "PLV")
            #
            # dFC_emp = np.loadtxt(ctb_folderOLD + "FC_" + emp_subj + "/" + bands[0][b] + "_dPLV4s.txt")
            #
            # # Compare dFC vs dFC_emp
            # t2 = np.zeros(shape=(2, len(dFC) ** 2 // 2 - len(dFC) // 2))
            # t2[0, :] = dFC[np.triu_indices(len(dFC), 1)]
            # t2[1, :] = dFC_emp[np.triu_indices(len(dFC), 1)]
            # dFC_ksd = scipy.stats.kstest(dFC[np.triu_indices(len(dFC), 1)], dFC_emp[np.triu_indices(len(dFC), 1)])[0]

            # ## Metastability: Kuramoto Order Parameter
            # ko_std, ko_mean = kuramoto_order(raw_data, samplingFreq)
            # ko_emp = np.loadtxt(ctb_folderOLD + "FC_" + emp_subj + "/" + bands[0][b] + "_sdKO.txt")

            ## Gather results
            result.append(
                (emp_subj, model,  g, s, r, min_cx, max_cx,
                 IAF[0], module[0], band_module[0], bands[0][b],
                 plv_r))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(result, dtype=object)
