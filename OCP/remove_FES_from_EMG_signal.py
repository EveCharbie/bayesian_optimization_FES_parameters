import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os
import scipy
from enum import Enum


class FilterType(Enum):
    IIRNOTCH = 0
    FFT = 1
    ACSR = 2


def linear_interpolation_of_missing_data(data):
    in_a_nan_sequence = False
    for i in range(data.shape[0]):
        if np.isnan(data[i]):
            if not in_a_nan_sequence:
                in_a_nan_sequence = True
                start_nan_sequence = i
            if i == data.shape[0] - 1:
                end_nan_sequence = i
                data[start_nan_sequence:end_nan_sequence+1] = np.linspace(data[start_nan_sequence-1], data[end_nan_sequence+1], end_nan_sequence-start_nan_sequence+1)
        else:
            if in_a_nan_sequence:
                end_nan_sequence = i
                data[start_nan_sequence:end_nan_sequence] = np.linspace(data[start_nan_sequence-1], data[end_nan_sequence], end_nan_sequence-start_nan_sequence)
                in_a_nan_sequence = False
    return data


def acsr_filter(v_rest, v_target, ws):
    """
    Artifact Component Specific Rejection (ACSR) filter for removing artifacts.

    Parameters:
        v_rest (numpy.ndarray): The raw signal to train filter parameters.
        v_target (numpy.ndarray): The raw signal to filter.
        ws (int): Window length.

    Returns:
        numpy.ndarray: The filtered signal.
    """
    mmags = acsr_init(v_rest, ws)
    overlap = ws // 2
    ev = window_division(v_target, ws, overlap)
    fv = []

    for tt in range(ev.shape[1]):
        comp = artifact_removal(ev[:, tt], 'ind_app', mmags)
        if overlap > 0:
            fv.append(comp[overlap:])
        else:
            fv.append(comp)

    fv = np.concatenate(fv, axis=0)
    fv = np.concatenate((np.zeros(overlap), fv))

    if len(fv) < len(v_target):
        fv = np.concatenate((fv, np.zeros(len(v_target) - len(fv))))

    return fv.reshape(v_target.shape)

def acsr_init(v, windows):
    """Initialize the ACSR parameters."""
    rv = window_division(v, windows, windows - 1)
    mmags, _ = artifact_removal(rv, 'init')
    return mmags

def window_division(data, windows, overlap):
    """Divide the data into overlapping windows."""
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_ch = data.shape[0]
    end_points = np.arange(windows, len(data[0]) + 1, windows - overlap)

    output = []
    for sig in data:
        sig_windows = [sig[start - windows:start] for start in end_points]
        output.append(np.array(sig_windows).T)

    if n_ch == 1:
        return output[0]
    return np.array(output)

def artifact_removal(input_data, state, params=None):
    """Perform artifact removal based on the state."""
    if state == 'init':
        target_fft = np.fft.fft(input_data, axis=0)
        mags = np.abs(target_fft)
        mmags = np.max(mags, axis=1) if mags.shape[1] > 1 else mags.flatten()
        return mmags, mags

    elif state == 'ind_app':
        mmags = params
        target_fft = np.fft.fft(input_data)
        comp_freq = np.zeros_like(target_fft, dtype=complex)

        for ii, freq in enumerate(target_fft):
            a, b = acsr_computation(freq, mmags[ii])
            comp_freq[ii] = complex(a, b)

        return np.fft.ifft(comp_freq, axis=0).real

def acsr_computation(freq, mag):
    """Compute the adjusted frequency component."""
    a0, b0 = np.real(freq), np.imag(freq)
    s_a = np.sign(a0)

    p0 = np.arctan2(b0, a0)
    m0 = np.sqrt(a0**2 + b0**2)
    ratio = np.tan(p0)

    m1 = max(m0 - mag, 0)

    a1 = s_a * np.sqrt((m1**2) / (1 + ratio**2))
    b1 = a1 * ratio

    return a1, b1


data_path = "/home/charbie/Documents/Programmation/bayesian_optimization_FES_parameters/OCP/data/EMG_Thomas/"
# resting_data_path = "/home/charbie/Documents/Programmation/bayesian_optimization_FES_parameters/OCP/data/EMG_Thomas/Resting/"
filtering_method = FilterType.ACSR


axis_label_set = False
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
color = colormaps["viridis"]
for i_file, file in enumerate(os.listdir(data_path)):
    if not file.endswith('.c3d'):
        continue

    freq_to_remove_to_remove = float(file.split('_')[4][1:])  # Hz
    c3d = ezc3d.c3d(data_path + file)
    emg_sampling_frequency = c3d["parameters"]["ANALOG"]["RATE"]["value"][0]  #Hz

    analogs_data = c3d["data"]["analogs"]
    analogs_names = c3d["parameters"]["ANALOG"]["LABELS"]["value"]
    emg_names = [m for m in analogs_names if "Channel" not in m]
    emg_indices = [i for i, m in enumerate(analogs_names) if "Channel" not in m]
    emg_data = analogs_data[0, emg_indices, :]
    non_nan_emg_data = np.zeros_like(emg_data)
    non_nan_emg_data[0, :] = linear_interpolation_of_missing_data(emg_data[0, :])
    non_nan_emg_data[1, :] = linear_interpolation_of_missing_data(emg_data[1, :])


    if filtering_method == FilterType.IIRNOTCH:
        emg_data_filtered = np.zeros_like(emg_data)
        # Design a notch filter
        b, a = scipy.signal.iirnotch(freq_to_remove_to_remove, 30, emg_sampling_frequency)
        # Apply the filter to the data
        emg_data_filtered[0, :] = scipy.signal.filtfilt(b, a, non_nan_emg_data[0, :])
        emg_data_filtered[1, :] = scipy.signal.filtfilt(b, a, non_nan_emg_data[1, :])

    elif filtering_method == FilterType.FFT:
        nb_emg_frames = emg_data.shape[1]
        freq_to_removes = np.fft.rfftfreq(nb_emg_frames, d=1/emg_sampling_frequency)  # freq_to_removeuency axis
        spectrum_0 = np.fft.rfft(emg_data[0, :])        # FFT of the signal
        spectrum_1 = np.fft.rfft(emg_data[1, :])        # FFT of the signal
        spectrum = np.vstack((spectrum_0, spectrum_1))
        axs[2, 0].plot(freq_to_removes, spectrum[0, :], '-', color=color(i_file/len(os.listdir(data_path))))
        axs[2, 1].plot(freq_to_removes, spectrum[1, :], '-', color=color(i_file/len(os.listdir(data_path))))

        # Create a bandstop filter in the freq_to_removeuency domain
        filtered_spectrum = np.zeros_like(spectrum)
        bandwidth = 10
        mask = (freq_to_removes < freq_to_remove_to_remove - bandwidth) | (freq_to_removes > freq_to_remove_to_remove + bandwidth)
        filtered_spectrum[0, :] = spectrum[0, :] * mask
        filtered_spectrum[1, :] = spectrum[1, :] * mask

        # Inverse FFT to transform back to the time domain
        emg_data_filtered = np.zeros_like(emg_data)
        emg_data_filtered[0, :] = np.fft.irfft(filtered_spectrum[0, :], n=nb_emg_frames)
        emg_data_filtered[1, :] = np.fft.irfft(filtered_spectrum[1, :], n=nb_emg_frames)

    elif filtering_method == FilterType.ACSR:
        window_size = 0.200 * emg_sampling_frequency  #200ms in the original paper
        filtered_signal_0 = acsr_filter(rest_data[0, :], emg_data[0, :], window_size)
        filtered_signal_1 = acsr_filter(rest_data[1, :], emg_data[1, :], window_size)
        emg_data_filtered = np.vstack((filtered_signal_0, filtered_signal_1))
    else:
        raise ValueError(f"Filtering method {filtering_method} not implemented yet")


    axs[0, 0].plot(emg_data[0, :], '-', color=color(i_file/len(os.listdir(data_path))), label=f"{freq_to_remove_to_remove} Hz")
    axs[0, 1].plot(emg_data[1, :], '-', color=color(i_file/len(os.listdir(data_path))))
    axs[1, 0].plot(emg_data_filtered[0, :], '-', color=color(i_file/len(os.listdir(data_path))))
    axs[1, 1].plot(emg_data_filtered[1, :], '-', color=color(i_file/len(os.listdir(data_path))))

    if not axis_label_set:
        axs[0, 0].set_title(emg_names[0])
        axs[0, 1].set_title(emg_names[1])
        axs[1, 0].set_title(f"EMG filtered ({emg_names[0]})")
        axs[1, 1].set_title(f"EMG filtered ({emg_names[1]})")
        axs[0, 0].set_xlim(50000, 70000)
        axs[0, 1].set_xlim(50000, 70000)
        axs[1, 0].set_xlim(50000, 70000)
        axs[1, 1].set_xlim(50000, 70000)
        if filtering_method == FilterType.FFT:
            axs[2, 0].set_title(f"Fourrier transform ({emg_names[0]})")
            axs[2, 1].set_title(f"Fourrier transform ({emg_names[1]})")
            axs[2, 0].set_xlim(0, 200)
            axs[2, 1].set_xlim(0, 200)
        axis_label_set = True
        

axs[0, 0].legend()
plt.tight_layout()
plt.savefig("EMG_signals_FFT.png")
plt.show()







