
import time
import numpy as np
from mpi4py import MPI

from tvb.simulator.lab import *
from tvb.simulator.lab import connectivity
from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995


## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTpeaks

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"
    import sys
    sys.path.append(wd)
    from toolbox.fft import FFTpeaks



def simulations(params, conn, out="fft", mode="classic", rois="bnm"):
    """
    Returning peaks and powers for every simulated region;
    Whole spectra only for the last one (due to FFTpeaks function design).

    :param params:
    :return:
    """

    # This simulation will generate FC for a virtual "Subject".
    # Define identifier (i.e. could be 0,1,11,12,...)
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

    tic0 = time.time()

    samplingFreq = 1000  # Hz
    simLength = 5000  # ms - relatively long simulation to be able to check for power distribution
    transient = 1000  # seconds to exclude from timeseries due to initial transient

    if rois == "pair":
        if "classic" in mode:
            m = JansenRit1995(He=np.array([params[0]]), Hi=np.array([params[1]]),
                              tau_e=np.array([params[2]]), tau_i=np.array([params[3]]),
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array([0.22]), sigma=np.array([0]),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

            coup = coupling.SigmoidalJansenRit(a=np.array([0]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))

        elif "prebif" in mode:
            m = JansenRit1995(He=np.array([params[0], 3.25]), Hi=np.array([params[1], 22]),
                              tau_e=np.array([params[2], 10]), tau_i=np.array([params[3], 20]),
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array([0, 0.15]), sigma=np.array([0, 0.22]),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

            # Coupling function
            coup = coupling.SigmoidalJansenRit(a=np.array([10]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))

    elif rois == "bnm":
        if "classic" in mode:
            m = JansenRit1995(He=params[0], Hi=params[1],
                              tau_e=params[2], tau_i=params[3],
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array([0.09]), sigma=np.array([0]),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

            coup = coupling.SigmoidalJansenRit(a=np.array([4]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))

        elif "prebif" in mode:

            sigma_array = [0.22 if 'Thal' in roi else 0 for roi in conn.region_labels]
            p_array = [0.15 if 'Thal' in roi else 0.09 for roi in conn.region_labels]

            m = JansenRit1995(He=params[0], Hi=params[1],
                              tau_e=params[2], tau_i=params[3],
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array(p_array), sigma=np.array(sigma_array),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

            # Coupling function
            coup = coupling.SigmoidalJansenRit(a=np.array([2]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))


    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.EulerDeterministic(dt=1000 / samplingFreq)

    conn.weights = conn.scaled_weights(mode="tract")

    if rois=="pair":
        # Subset of 2 nodes is enough
        conn.weights = conn.weights[:2][:, :2]
        conn.tract_lengths = conn.tract_lengths[:2][:, :2]
        conn.region_labels = conn.region_labels[:2]

    mon = (monitors.Raw(),)

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator,
                              monitors=mon)
    sim.configure()

    output = sim.run(simulation_length=simLength)
    # print("Simulation time: %0.2f sec" % (time.time() - tic0,))
    # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
    raw_data = output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T

    # Check initial transient and cut data
    # timeseriesPlot(raw_data, raw_time, conn.region_labels, "figures", mode="html", title=title)

    if out == "fft":
        # Fourier Analysis plot
        peaks, modules, band_modules, fft, freqs = FFTpeaks(raw_data, simLength - transient, curves=True)
        return peaks, modules, band_modules, fft, freqs

    elif out == "signals":
        return raw_data

    elif out == "cluster":

        type = "flat" if np.average(raw_data.max(axis=1) - raw_data.min(axis=1)) < 1e-6 \
            else "endflat" if np.average(raw_data[:, -500:].max(axis=1) - raw_data[:, -500:].min(axis=1)) < 1e-6 \
            else "nsnc"

        meanS = np.average(raw_data)

        freq = 0 if "flat" in type else np.average(FFTpeaks(raw_data, 4000, curves=False)[0])

        pow = 0 if "flat" in type else np.average(FFTpeaks(raw_data, 4000, curves=False)[2])

        signals = None if "flat" in type else raw_data[:10, :]

        return type, meanS, freq, pow, signals

def calibTransfOne(params_):

    results = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    for ii, set in enumerate(params_):

        tic = time.time()
        print("Rank %i out of %i  ::  %i/%i SIMULATING for_" % (rank, size, ii + 1, len(params_)))

        print(set)

        He, Hi, taue, taui = set

        #  0. STRUCTURAL CONNECTIVITY   #########
        #  Define structure through which the proteins will spread;
        #  Not necessarily the same than the one used to simulate activity.
        subj = "HC-fam"
        conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")

        sim_params = np.array([[He for roi in conn.region_labels], [Hi for roi in conn.region_labels],
                               [taue for roi in conn.region_labels], [taui for roi in conn.region_labels]])

        ## Gather results
        type, meanS, freq, pow, signals = simulations(sim_params, conn, out="cluster", mode="classic", rois="bnm")

        results.append((He, Hi, taue, taui, type, meanS, freq, pow))  #, signals))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(results, dtype=object)



