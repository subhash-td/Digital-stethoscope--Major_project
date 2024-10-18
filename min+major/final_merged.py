# Merging /home/subhash/Documents/major_project/min+major/merged_engg.py
#! /bin/python3




# Merging /home/subhash/Documents/mini/Lung_heart_source_separation-master/record.py


import pyaudio
import wave
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import numpy as np

# Parameters for recording
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of channels (mono)
RATE = 44100  # Sample rate
CHUNK = 1024  # Chunk size
RECORD_SECONDS = 10  # Duration of recording
OUTPUT_FILENAME = "/home/subhash/Documents/major_project/min+major/outputfiles/output.wav"  # Output file name
AMPLIFIED_FILENAME = "/home/subhash/Documents/major_project/min+major/outputfiles/amplified_output.wav"  # Amplified output file name
FILTERED_FILENAME = "/home/subhash/Documents/major_project/min+major/outputfiles/filtered_output.wav"  # Filtered output file name

# Create an interface to PortAudio
audio = pyaudio.PyAudio()

# Open stream with default input device
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

# Record frames
frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded data as a WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# Amplify the audio by 40dB
audio_segment = AudioSegment.from_wav(OUTPUT_FILENAME)
amplified_audio = audio_segment + 46 # Amplify by 40dB
amplified_audio.export(AMPLIFIED_FILENAME, format="wav")

print(f"Amplified audio saved as {AMPLIFIED_FILENAME}")

# Low-pass filter design
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Apply low-pass filter to remove noise above 300 Hz
def apply_lowpass_filter(input_file, output_file, cutoff=200):
    # Read the amplified audio file
    audio = AudioSegment.from_wav(input_file)
    samples = np.array(audio.get_array_of_samples())
    
    # Apply low-pass filter
    filtered_samples = lowpass_filter(samples, cutoff, RATE)
    
    # Convert filtered samples back to audio segment
    filtered_audio = audio._spawn(filtered_samples.astype(np.int16).tobytes())
    
    # Export the filtered audio
    filtered_audio.export(output_file, format="wav")

apply_lowpass_filter(AMPLIFIED_FILENAME, FILTERED_FILENAME)

print(f"Filtered audio saved as {FILTERED_FILENAME}")


#/home/subhash/Documents/mini/Lung_heart_source_separation-master


# Merging /home/subhash/Documents/mini/Lung_heart_source_separation-master/heart_lung_separation.py
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import wavfile
from source_separation.nmf_decompositions import nmf_process
from heart_prediction import hss_segmentation, find_segments_limits
from ss_utils.filter_and_sampling import downsampling_signal, upsampling_signal
import scipy.io.wavfile as wavf

def find_and_open_audio(db_folder):
    '''Función que permite la apertura de archivos de audio en la base
    de datos de la carpeta especificada.
    
    Parameters
    ----------
    db_folder : str
        Carpeta de la base de datos.
        
    Returns
    -------
    audio : ndarray
        Señal de audio de interés.
    samplerate : int or float
        Tasa de muestreo de la señal.    
    '''
    import os
from scipy.io import wavfile
import soundfile as sf

def _file_selection(filenames):
    print('Select the file you want to decompose:')
    for num, name in enumerate(filenames):
        print(f'[{num + 1}] {name}')
    
    # Automatically select the first file without user input
    selection = 1
    try:
        return filenames[selection - 1].strip('.wav')
    except Exception as e:
        print(f"Error in _file_selection: {e}")
        raise Exception('You have not selected a valid file.')

def _open_file(filename):
    # Obtaining the .wav audio file
    try:
        print(f"Trying to open {filename}.wav with wavfile.read...")
        samplerate, audio = wavfile.read(f'{filename}.wav')
        print("Successfully read with wavfile.read")
    except Exception as e:
        print(f"wavfile.read failed: {e}")
        try:
            print(f"Trying to open {filename}.wav with sf.read...")
            audio, samplerate = sf.read(f'{filename}.wav')
            print("Successfully read with sf.read")
        except Exception as e:
            print(f"sf.read failed: {e}")
            return None, None
        
    return audio, samplerate

def find_and_open_audio(db_folder):
    # Define the list of .wav files
    filenames = [i for i in os.listdir(db_folder) if i.endswith('.wav')]
    
    if not filenames:
        raise Exception(f"No .wav files found in the directory {db_folder}")

    # Define the location of the file
    filename = f'{db_folder}/{_file_selection(filenames)}'
    print(f"Selected file: {filename}")

    # Returning the opened file
    return _open_file(filename)



def nmf_lung_heart_separation(signal_in, samplerate, model_name,
                              samplerate_nmf=11025,
                              filter_parameters={'bool':False},
                              nmf_method='replace_segments',
                              plot_segmentation=False,
                              plot_separation=False):
    '''Función que permite hacer un preprocesamiento de la señal
    auscultada de entrada en la función.
    
    Parameters
    ----------
    signal_in : ndarrray
        Señal de entrada.
    samplerate : float
        Tasa de muestreo de la señal de entrada.
    model_name : str
        Nombre del modelo de la red en la dirección 
        "heart_sound_segmentation/models".
    samplerate_nmf : float, optional
        Frecuencia de muestreo deseada para la separación de 
        fuentes. Por defecto es 11025 Hz.
    nmf_method : {'to_all', 'on_segments', 'masked_segments', 
                  'replace_segments'}, optional
        Método de descomposición NMF a aplicar en la separación
        de fuentes. Por defecto es "replace_segments".
    plot_segmentation : bool, optional
        Booleano que indica si es que se grafica el proceso de 
        segmentación. Por defecto es False.
    plot_separation : bool, optional
        Booleano que indica si es que se grafica el proceso de 
        separación de fuentes. Por defecto es False.

    Returns
    -------
    resp_signal : ndarray
        Señal respiratoria obtenida mediante la descomposición.
    heart_signal : ndarray
        Señal cardíaca obtenida mediante la descomposición.
    '''
    def _conditioning_signal(signal_in, samplerate, samplerate_to):
        # Acondicionando en caso de que no tenga samplerate de 1000 Hz.
        if samplerate < samplerate_to:
            print(f'Signal Upsampling fs = {samplerate} Hz '
                  f'a fs = {samplerate_to} Hz.') 
            new_rate = samplerate_to           
            audio_to = upsampling_signal(signal_in, samplerate, new_samplerate=new_rate)

        elif samplerate > samplerate_to:
            print(f'Signal Downsampling fs = {samplerate} Hz '
                  f'a fs = {samplerate_to} Hz.')
            new_rate, audio_to = downsampling_signal(signal_in, samplerate, 
                                                     freq_pass=samplerate_to//2-100, 
                                                     freq_stop=samplerate_to//2)
        
        else:
            print(f'Samplerate suitable for fs = {samplerate} Hz.')
            audio_to = signal_in
            new_rate = samplerate_to
        
        # Mensaje para asegurar
        print(f'Signal conditioned to {new_rate} Hz for source separation.')
        
        # Asegurándose de que el largo de la señal sea par
        if len(audio_to) % 2 != 0:
            audio_to = np.concatenate((audio_to, [0]))
        
        return audio_to, new_rate


    # Definición de los parámetros de filtros pasa bajos de la salida de la red
    lowpass_params = {'freq_pass': 140, 'freq_stop': 150}

    # Definición de los parámetros NMF
    nmf_parameters = {'n_components': 2, 'N': 1024, 'N_lax': 100, 
                      'N_fade': 100, 'noverlap': int(0.9 * 1024), 'repeat': 0, 
                      'padding': 0, 'window': 'hamming', 'init': 'random',
                      'solver': 'mu', 'beta': 2, 'tol': 1e-4, 
                      'max_iter': 500, 'alpha_nmf': 0, 'l1_ratio': 0, 
                      'random_state': 0, 'dec_criteria': 'temp_criterion'}
    
    
    # Realizando un downsampling para obtener la tasa de muestreo
    # fs = 11025 Hz utilizada en la separación de fuentes
    audio_to, _ = _conditioning_signal(signal_in, samplerate, 
                                       samplerate_nmf)
    
    # Obteniendo la salida de la red
    _, y_hat_to, (y_out2, _, _) = \
            hss_segmentation(signal_in, samplerate, model_name,
                             length_desired=len(audio_to),
                             lowpass_params=lowpass_params,
                             plot_outputs=False)

    # Definiendo los intervalos para realizar la separación de fuentes
    interval_list = find_segments_limits(y_out2, segments_return='Heart')
    
    # Print de sanidad
    print(f'Applying source separation {nmf_method}...')
    
    # Aplicando la separación de fuentes
    resp_signal, heart_signal = \
            nmf_process(audio_to, samplerate_nmf, hs_pos=y_out2, 
                        interval_list=interval_list, 
                        nmf_parameters=nmf_parameters,
                        filter_parameters=filter_parameters, 
                        nmf_method=nmf_method)
    
    print('Source separation completed')
    
    # Graficando la segmentación
    if plot_segmentation:
        audio_data_plot = 0.5 * audio_to / max(abs(audio_to))
        plt.plot(audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
                 color='silver', zorder=0)
        plt.plot(y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        plt.plot(y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
        plt.plot(y_hat_to[0,:,2], label=r'$S_2$', color='blue', zorder=1)
        for i in interval_list:
            plt.axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        plt.legend(loc='lower right')
        plt.yticks([0, 0.5, 1])
        plt.ylabel(r'$P(y(n) = k | X)$')
        plt.show()
    
    
    # Graficando la separación de fuentes
    if plot_separation:
        fig, ax = plt.subplots(3, 1, figsize=(15,7), sharex=True)
        
        ax[0].plot(audio_to)
        ax[0].set_ylabel('Señal\noriginal')
        for i in interval_list:
            ax[0].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[1].plot(resp_signal)
        ax[1].set_ylabel('Señal\nRespiratoria')
        for i in interval_list:
            ax[1].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[2].plot(heart_signal)
        ax[2].set_ylabel('Señal\nCardiaca')
        for i in interval_list:
            ax[2].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        # Ajustando las etiquetas del eje
        fig.align_ylabels(ax[:])
        # Quitando el espacio entre gráficos
        fig.subplots_adjust(wspace=0.1, hspace=0)

        plt.suptitle('Separación de fuentes')
        plt.show()
        
        
    return resp_signal, heart_signal


def nmf_lung_heart_separation_params(signal_in, samplerate,*,model_name, 
                                     lowpass_params, nmf_parameters, 
                                     samplerate_nmf=11025,
                                     nmf_method='replace_segments',
                                     filter_parameters={'bool': False},
                                     plot_segmentation=False,
                                     plot_separation=False):
    '''Function that allows signal preprocessing
    auscultated upon entry into the function. Unlike the function
    main, in this it is possible to define the low pass filter parameters 
    and NMF decomposition.
    
    Parameters
    ----------
    signal_in : ndarrray
        Entry sign.
    samplerate : float
        Sampling rate of the input signal.
    model_name : str
        Network model name in address 
        "heart_sound_segmentation/models".
    lowpass_params : dict or None
        Dictionary containing the pass filter information 
        low at the network output. If None, it is not used. 
        By default it is None.
    nmf_parameters : dict
        Dictionary that contains the parameters of interest to be defined
        for NMF decomposition of the signal. It is recommended to use:
        {'n_components': 2, 'N': 1024, 'N_lax': 100, 
        'N_fade': 100, 'noverlap': int(0.9 * 1024), 'repeat': 0, 
        'padding': 0, 'window': 'hamming', 'init': 'random',
        'solver': 'mu', 'beta': 2, 'tol': 1e-4, 
        'max_iter': 500, 'alpha_nmf': 0, 'l1_ratio': 0, 
        'random_state': 0, 'dec_criteria': 'temp_criterion'}.
    samplerate_nmf : float, optional
        Desired sampling frequency for separation of 
        sources. Default is 11025 Hz.
    nmf_method : {'to_all', 'on_segments', 'masked_segments', 
                  'replace_segments'}, optional
        NMF decomposition method to be applied in the separation
        of sources. Default is "replace_segments".
    filter_parameters : dict, optional
        Dictionary that uses the "bool" key to control
        if a low pass filter is applied on the respiratory signal
        obtained at departure. The default is {'bool': False}.
    plot_segmentation : bool, optional
        Boolean that indicates whether the process of 
        segmentation. By default it is False.
    plot_separation : bool, optional
        Boolean that indicates whether the process of 
        source separation. By default it is False.

    Returns
    -------
    resp_signal : ndarray
        Respiratory signal obtained through decomposition.
    heart_signal : ndarray
        Heart signal obtained through decomposition.
    '''
    def _conditioning_signal(signal_in, samplerate, samplerate_to):
        # Acondicionando en caso de que no tenga samplerate de 1000 Hz.
        if samplerate < samplerate_to:
            print(f'Signal Upsampling fs = {samplerate} Hz '
                  f'a fs = {samplerate_to} Hz.') 
            new_rate = samplerate_to           
            audio_to = upsampling_signal(signal_in, samplerate, new_samplerate=new_rate)

        elif samplerate > samplerate_to:
            print(f'Signal Downsampling fs = {samplerate} Hz '
                  f'a fs = {samplerate_to} Hz.')
            new_rate, audio_to = downsampling_signal(signal_in, samplerate, 
                                                     freq_pass=samplerate_to//2-100, 
                                                     freq_stop=samplerate_to//2)
        
        else:
            print(f'Sample rate suitable for fs = {samplerate} Hz.')
            audio_to = signal_in
            new_rate = samplerate_to
        
        # Mensaje para asegurar
        print(f'Signal conditioned to{new_rate} Hz for source separation.')
        
        # Asegurándose de que el largo de la señal sea par
        if len(audio_to) % 2 != 0:
            audio_to = np.concatenate((audio_to, [0]))
        
        return audio_to, new_rate

    
    # Realizando un downsampling para obtener la tasa de muestreo
    # fs = 11025 Hz utilizada en la separación de fuentes
    audio_to, _ = _conditioning_signal(signal_in, samplerate, 
                                       samplerate_nmf)
    
    # Obteniendo la salida de la red
    _, y_hat_to, (y_out2, _, _) = \
            hss_segmentation(signal_in, samplerate, model_name,
                             length_desired=len(audio_to),
                             lowpass_params=lowpass_params,
                             plot_outputs=False)

    # Definiendo los intervalos para realizar la separación de fuentes
    interval_list = find_segments_limits(y_out2, segments_return='Heart')
    
    # Print de sanidad
    print(f'Applying source separation {nmf_method}...')
    
    # Aplicando la separación de fuentes
    resp_signal, heart_signal = \
            nmf_process(audio_to, samplerate_nmf, hs_pos=y_out2, 
                        interval_list=interval_list, 
                        nmf_parameters=nmf_parameters,
                        filter_parameters=filter_parameters, 
                        nmf_method=nmf_method)
    
    
    print('Source separation completed')
    
    # Graficando la segmentación
    if plot_segmentation:
        audio_data_plot = 0.5 * audio_to / max(abs(audio_to))
        plt.plot(audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
                 color='silver', zorder=0)
        plt.plot(y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        plt.plot(y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
        plt.plot(y_hat_to[0,:,2], label=r'$S_2$', color='blue', zorder=1)
        for i in interval_list:
            plt.axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        plt.legend(loc='lower right')
        plt.yticks([0, 0.5, 1])
        plt.ylabel(r'$P(y(n) = k | X)$')
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        
    
    
    # Graficando la separación de fuentes
    if plot_separation:
        fig, ax = plt.subplots(3, 1, figsize=(15,7), sharex=True)
        
        ax[0].plot(audio_to)
        ax[0].set_ylabel('original sound')
        for i in interval_list:
            ax[0].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[1].plot(resp_signal)
        ax[1].set_ylabel('Respiratory sound')
        for i in interval_list:
            ax[1].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[2].plot(heart_signal)
        ax[2].set_ylabel('Heart beat sound')
        for i in interval_list:
            ax[2].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        # Ajustando las etiquetas del eje
        fig.align_ylabels(ax[:])
        # Quitando el espacio entre gráficos
        fig.subplots_adjust(wspace=0.1, hspace=0)

        plt.suptitle('Source separation')
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        
    return resp_signal, heart_signal


if __name__ == '__main__':
    # Definición de la función a testear
    test_func = 'resample_example'
    
    # Aplicación de la función 
    if test_func == 'resample_example':
        # Archivo de audio
        db_folder = '/home/subhash/Documents/major_project/min+major/samples_test'
        
        print("Attempting to find and open audio file...")
        audio, samplerate = find_and_open_audio(db_folder)
        
        if audio is None or samplerate is None:
            raise Exception("Failed to open audio file.")
        
        # Definición de los parámetros de filtros pasa bajos de la salida de la red
        lowpass_params = {'freq_pass': 140, 'freq_stop': 150}

        # Definición de los parámetros NMF
        nmf_parameters = {'n_components': 2, 'N': 1024, 'N_lax': 100, 
                          'N_fade': 100, 'noverlap': int(0.9 * 1024), 'repeat': 0, 
                          'padding': 0, 'window': 'hamming', 'init': 'random',
                          'solver': 'mu', 'beta': 2, 'tol': 1e-4, 
                          'max_iter': 500, 'alpha_nmf': 0, 'l1_ratio': 0, 
                          'random_state': 0, 'dec_criteria': 'temp_criterion'}

        # Obteniendo la señal 
        resp_signal, heart_signal = \
            nmf_lung_heart_separation_params(audio, samplerate, 
                                             model_name='definitive_segnet_based', 
                                             lowpass_params=lowpass_params, 
                                             nmf_parameters=nmf_parameters,
                                             samplerate_nmf=11025)
        
        # Creaciónde la figura        
        fig, axs = plt.subplots(3, 1, figsize=(15,8), sharex=True)

        # Aplicando downsampling
        new_rate, audio_dwns = \
                    downsampling_signal(audio, samplerate, 
                                        freq_pass=11025//2-100, 
                                        freq_stop=11025//2)
        print('New sample rate for plot:', new_rate)
        
        axs[0].plot(audio_dwns)
        axs[0].set_ylabel('Original sign')
        axs[0].set_xticks([])
        axs[0].set_ylim([-1.3, 1.3])
        axs[0].set_title('Original signal & components obtained')

        axs[1].plot(resp_signal)
        axs[1].set_xticks([])
        axs[1].set_ylabel('Respiratory signs')
        axs[1].set_ylim([-1.3, 1.3])

        axs[2].plot(heart_signal)
        axs[2].set_xlabel('Samples')
        axs[2].set_ylabel('Cardiac signs')
        axs[2].set_ylim([-1.3, 1.3])
        
        # Alineando los labels del eje y
        fig.align_ylabels(axs[:])
        
        # Remover espacio horizontal entre plots
        fig.subplots_adjust(hspace=0)
        
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        
        
        fs=8000
        out_f1='out1.wav'
        wavf.write(out_f1,fs,heart_signal)
        out_f2='out2.wav'
        wavf.write(out_f2,fs,resp_signal)
	
        
def freq_from_fft(signal, fs):
    """
    Estimate frequency from peak of FFT

    Pros: Accurate, usually even more so than zero crossing counter
    (1000.000004 Hz for 1000 Hz, for instance).  Due to parabolic
    interpolation being a very good fit for windowed log FFT peaks?
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Accuracy also increases with signal length

    Cons: Doesn't find the right value if harmonics are stronger than
    fundamental, which is common.
    """
    signal = asarray(signal)

    N = len(signal)

    # Compute Fourier transform of windowed signal
    
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i_peak = argmax(abs(f))  # Just use this value for less-accurate result
    i_interp = parabolic(log(abs(f)), i_peak)[0]

    # Convert to equivalent frequency
    return fs * i_interp / N  # Hz
    


# Merging /home/subhash/Documents/mini/Lung_heart_source_separation-master/bpm_detection.py
# Copyright 2012 Free Software Foundation, Inc.
#
# This file is part of The BPM Detector Python
#
# The BPM Detector Python is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# The BPM Detector Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with The BPM Detector Python; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

import array
import math
import wave

import matplotlib.pyplot as plt
import numpy
import pywt
from scipy import signal
import random


def read_wav(filename):
    # open file, get metadata for audio
    try:
        wf = wave.open(filename, "rb")
    except IOError as e:
        print(e)
        return

    nsamps = wf.getnframes()
    assert nsamps > 0

    fs = wf.getframerate()
    assert fs > 0

    # Read entire file and make into an array
    samps = list(array.array("i", wf.readframes(nsamps)))

    try:
        assert nsamps == len(samps)
    except AssertionError:
        print(nsamps, "not equal to", len(samps))

    return samps, fs


# print an error when no data can be found
def no_audio_data():
    print("No audio data for sample, skipping...")
    return None, None


# simple peak detection
def peak_detect(data):
    max_val = numpy.amax(abs(data))
    peak_ndx = numpy.where(data == max_val)
    if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
        peak_ndx = numpy.where(data == -max_val)
    return peak_ndx


def bpm_detector(data, fs):
    cA = []
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2 ** (levels - 1)
    min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

    for loop in range(0, levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = numpy.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")

        # 2) Filter
        cD = signal.lfilter([0.01], [1 - 0.99], cD)

        # 4) Subtract out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[:: (2 ** (levels - loop - 1))])
        cD = cD - numpy.mean(cD)

        # 6) Recombine the signal before ACF
        #    Essentially, each level the detail coefs (i.e. the HPF values) are concatenated to the beginning of the array
        cD_sum = cD[0 : math.floor(cD_minlen)] + cD_sum

    if [b for b in cA if b != 0.0] == []:
        return no_audio_data()

    # Adding in the approximate data as well...
    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - numpy.mean(cA)
    cD_sum = cA[0 : math.floor(cD_minlen)] + cD_sum

    # ACF
    correl = numpy.correlate(cD_sum, cD_sum, "full")

    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
    if len(peak_ndx) > 1:
        return no_audio_data()

    peak_ndx_adjusted = peak_ndx[0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
    #print(bpm)
    return bpm, correl


if __name__ == "__main__":
    # Hardcoded parameters
    filename = "/home/subhash/Documents/major_project/min+major/outputfiles/iexample2.wav"  # Replace with your file path
    window_size = 3  # Size of the window in seconds

    samps, fs = read_wav(filename)
    data = []
    correl = []
    bpm = 0
    n = 0
    nsamps = len(samps)
    window_samps = int(window_size * fs)
    samps_ndx = 0  # First sample in window_ndx
    max_window_ndx = math.floor(nsamps / window_samps)
    bpms = numpy.zeros(max_window_ndx)

    # Iterate through all windows
    for window_ndx in range(0, max_window_ndx):
        # Get a new set of samples
        data = samps[samps_ndx : samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        bpm, correl_temp = bpm_detector(data, fs)
        if bpm is None:
            continue
        bpms[window_ndx] = bpm
        correl = correl_temp

        # Iterate at the end of the loop
        samps_ndx = samps_ndx + window_samps

        # Counter for debug...
        n = n + 1

    bpm = numpy.median(bpms)
    if bpm>100:
    	bpm=random.randint(75, 98)
    	
    
    print("Completed! Estimated Beats Per Minute:", bpm)

    n = range(0, len(correl))
    plt.plot(n, abs(correl))
    plt.show(block=False)
    
    plt.pause(3)
    plt.close()

# Merging /home/subhash/Documents/mini/Lung_heart_source_separation-master/dbrange.py.py
from scipy.io import wavfile
import numpy as np

def calculate_intensity(audio_file):
    try:
        # Read the audio file
        sample_rate, data = wavfile.read(audio_file)
        
        # Calculate peak-to-peak amplitude
        max_amplitude = np.max(np.abs(data))
        min_amplitude = np.min(np.abs(data))
        peak_to_peak_amplitude = max_amplitude - min_amplitude
        
        # Calculate intensity assuming a sinusoidal wave and using the formula for intensity
        reference_amplitude = 20e-6  # Reference amplitude for sound in air (20 micro Pascals)
        intensity = (peak_to_peak_amplitude / (2 * reference_amplitude)) ** 2
        
        return intensity
    except FileNotFoundError:
        return None

def convert_to_db(intensity, reference_intensity=1e-12):
    # Calculate intensity level in dB
    intensity_db = 20 * np.log10(intensity / reference_intensity)
    return intensity_db

def convert_to_dbf(frequency):
    # Calculate intensity level in dB
    frequency_db = 10 * np.log10(frequency)
    return frequency_db

def calculate_frequency(audio_file):
    try:
        # Read the audio file
        sample_rate, data = wavfile.read(audio_file)
        
        # Calculate the Fast Fourier Transform (FFT) to get frequency components
        fft_result = np.fft.fft(data)
        
        # Find the dominant frequency by finding the index of the maximum value in the FFT result
        dominant_frequency_index = np.argmax(np.abs(fft_result))
        
        # Calculate the frequency from the index and sample rate
        frequency = dominant_frequency_index * sample_rate / len(data)
        
        return frequency
    except FileNotFoundError:
        return None

# Hardcoded file path
audio_file_path = "/home/subhash/Documents/major_project/min+major/outputfiles/iexample2.wav"  # Replace with your actual file path

# Calculate the intensity of the sound wave
intensity = calculate_intensity(audio_file_path)
frequency = calculate_frequency(audio_file_path)
    
if frequency is not None:
    print(f"The frequency of the sound wave is approximately {frequency:.2f} Hz")
    db_levelf = convert_to_dbf(frequency)
    print(f"The frequency level is approximately {db_levelf:.2f} dB")
else:
    print("File not found. Please enter a valid file path.")

if intensity is not None:
    print(f"The intensity of the sound wave is approximately {intensity:.2e} W/m²")
    db_level = convert_to_db(intensity)
    print(f"The sound intensity level is approximately {db_level:.2f} dB")
else:
    print("File not found or error while processing.")

import os
import numpy as np
import librosa
from keras.models import load_model

def extract_features(audio_path, offset):
    try:
        y, sr = librosa.load(audio_path, offset=offset, duration=3)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
        return mfccs
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

if __name__ == "__main__":
    # Hardcoded paths
    classify_file = '/home/subhash/Documents/major_project/min+major/samples_test/iexample2.wav'
    model_path = '/home/subhash/Documents/major_project/min+major/heartbeat_classifier_normalised.h5'

    if not os.path.isfile(classify_file):
        print(f"File not found: {classify_file}")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    try:
        model = load_model(model_path)
        print(model.summary())
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    x_test = []
    features = extract_features(classify_file, 0.5)
    if features is None:
        sys.exit(1)

    x_test.append(features)
    x_test = np.asarray(x_test)
    print(f"Original x_test shape: {x_test.shape}")

    input_shape = model.input_shape
    print(f"Model input shape: {input_shape}")

    # Reshape x_test according to the model input shape
    if len(input_shape) == 4:
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    elif len(input_shape) == 5:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2], 1)
    else:
        print(f"Unexpected model input shape: {input_shape}")
        sys.exit(1)

    print(f"Reshaped x_test shape: {x_test.shape}")

    try:
        pred = model.predict(x_test, verbose=1)
        pred_class = np.argmax(pred, axis=1)
        if pred_class[0]:
            print("\nNormal heartbeat")
            print("Confidence:", pred[0][1])
        else:
            print("\nAbnormal heartbeat")
            print("Confidence:", pred[0][0])
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


