import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
from scipy.signal import bessel, step, impulse
from Code.Heart_disease_prediction_Using_PJM_DJRNN_Testing import global_input_ecg_signal
from scipy import signal
from scipy import constants
class Brownian():
    def __init__(self, x0=0):
        assert (type(x0) == float or type(x0) == int or x0 is None), "Expect a float or None for the initial value"
        self.x0 = float(x0)
    def gen_cut_off_frequency_selection(self, n_step=100):
        if n_step < 0:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        w = np.ones(n_step) * self.x0
        for i in range(1, n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1, -1])
            # Weiner process
            w[i] = w[i - 1] + (yi / np.sqrt(n_step))
        return w
    def gen_normal(self, n_step=100):
        if n_step < 0:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        w = np.ones(n_step) * self.x0
        for i in range(1, n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i - 1] + (yi / np.sqrt(n_step))
        return w
    def stock_price(
            self,
            s0=100,
            mu=0.2,
            sigma=0.68,
            deltaT=52,
            dt=0.1
    ):
        n_step = int(deltaT / dt)
        time_vector = np.linspace(0, deltaT, num=n_step)
        # Stock variation
        stock_var = (mu - (sigma ** 2 / 2)) * time_vector
        self.x0 = 0
        # Weiner process (calls the `gen_normal` method)
        weiner_process = sigma * self.gen_normal(n_step)
        # Add two time series, take exponent, and multiply by the initial stock price
        s = s0 * (np.exp(stock_var + weiner_process))
        return s
b = Brownian()
cutoff = b.gen_cut_off_frequency_selection(5)
print("Brownian motion function-based cut-off frequency value is :",cutoff)
for i in range(4):
    plt.plot(b.gen_cut_off_frequency_selection(1000))
plt.xlabel('Time(t)')
plt.ylabel('B(t)')
plt.title("A possible realization of Brownian motion")
plt.show()
rate, data = scipy.io.wavfile.read(global_input_ecg_signal)
# rate is the sample rate, data is the data
rate == 44100
Wn = 2*constants.pi*54
cutoff_val = cutoff[1]  # want this to be cutoff Hz
nyquist = 0.5 * rate
normal_cutoff = -1*(cutoff_val / nyquist)
print(normal_cutoff)
order = 5
b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=True)
filtered = signal.filtfilt(b, a, data)
b, a = bessel(order, Wn, btype='low', analog=True, output='ba', norm='phase')
# Note: the upper limit for t was chosen after some experimentation.
# If you don't give a T argument to impulse or step, it will choose a
t = np.linspace(0, 0.00125, 2500, endpoint=False)
timp, yimp = impulse((b, a), T=t)
tstep, ystep = step((b, a), T=t)
'''plt.subplot(2, 1, 1)
plt.plot(timp, yimp, label='impulse response')
plt.legend(loc='upper right', framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.title('Impulse and step response of the Bessel filter')

plt.subplot(2, 1, 2)
plt.plot(tstep, ystep, label='step response')
plt.legend(loc='lower right', framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.xlabel('t')
plt.show()'''