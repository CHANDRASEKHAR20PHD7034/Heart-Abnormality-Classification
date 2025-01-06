from numpy.random import randn
from numpy.fft import rfft
from scipy.signal import butter, lfilter
from matplotlib.pyplot import loglog
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from Code.Heart_disease_prediction_Using_PJM_DJRNN_Testing import global_input_ecg_signal
import math
print("Existing Butterworth filter was executing...")
f_sample = 40000

# pass band frequency
f_pass = 4000

# stop band frequency
f_stop = 8000

# pass band ripple
fs = 0.5

# pass band freq in radian
wp = f_pass / (f_sample / 2)

# stop band freq in radian
ws = f_stop / (f_sample / 2)

# Sampling Time
Td = 1

# pass band ripple
g_pass = 0.5

# stop band attenuation
g_stop = 40
omega_p = (2 / Td) * np.tan(wp / 2)
omega_s = (2 / Td) * np.tan(ws / 2)

# Design of Filter using signal.buttord function
N, Wn = signal.buttord(omega_p, omega_s, g_pass, g_stop, analog=True)


# Conversion in Z-domain

# b is the numerator of the filter & a is the denominator
b, a = signal.butter(N, Wn, 'low', True)
z, p = signal.bilinear(b, a, fs)
# w is the freq in z-domain & h is the magnitude in z-domain
w, h = signal.freqz(z, p, 512)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

fs = 250
lowcut = 1.0
highcut = 50.0
x_Vtcr = randn(10000)
x2_Vtcr = butter_bandpass_filter(x_Vtcr, lowcut, highcut, fs, order=4)
loglog(abs(rfft(x_Vtcr)))
loglog(abs(rfft(x2_Vtcr)))
# Printing the values of order & cut-off frequency!
print("Order of the Filter=", N)  # N is the order
# Wn is the cut-off freq of the filter
print("Cut-off frequency= {:.3f} rad/s ".format(Wn))
'''plt.semilogx(w, 20*np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green')
plt.show()'''
imp = signal.unit_impulse(40)
c, d = signal.butter(N, 0.5)
response = signal.lfilter(c, d, imp)

'''plt.stem(np.arange(0, 40), imp, use_line_collection=True)
plt.stem(np.arange(0, 40), response, use_line_collection=True)
plt.margins(0, 0.1)

plt.xlabel('Time [samples]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()'''
print("Existing Butterworth filter was executed successfully...")
