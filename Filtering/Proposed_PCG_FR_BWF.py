import scipy.io.wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from Code.Heart_disease_prediction_Using_PJM_DJRNN_Testing import global_input_pcg_signal

from scipy import signal
import math
import cv2
sampleRate, data = scipy.io.wavfile.read(global_input_pcg_signal)
times = np.arange(len(data))/sampleRate
Wn = 2*constants.pi*54
cutoff = 54  # want this to be 54 Hz
frequency = 0.5 * sampleRate
normal_cutoff = cutoff / frequency
order = 5
# apply a 3-pole lowpass filter at 0.1x frequency
b, a = scipy.signal.butter(3, 0.1, btype='highpass', analog=True)
filtered = scipy.signal.filtfilt(b, a, data)
# plot the original data next to the filtered data
#plt.figure(figsize=(10, 4))
#plt.plot(times, data)
#plt.title("PCG Signal with Noise")
#plt.margins(0, .05)
#plt.show()
#plt.close()
plt.close()
plt.plot(times, filtered)
plt.margins(0, .05)
plt.tight_layout()
plt.axis('off')

plt.savefig("../Run/Result/FR_BF_PCG_signal_noise_removal.png")
plt.close()
for cutoff in [.03, .05, .1]:
    b, a = scipy.signal.butter(3, cutoff)
    filtered = scipy.signal.filtfilt(b, a, data)
    label = f"{int(cutoff*100):d}%"
    # Frequency Ratio based Butterworth Filter (FR-BF)
    Frequency_Ratio = filtered * normal_cutoff
    #plt.plot(Frequency_Ratio, label=label)



