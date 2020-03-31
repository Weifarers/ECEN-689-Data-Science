import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def c_xy_calc(xt, yt, k):
    # This is a generic cross covariance estimator. Since we need this value in multiple
    # locations, we make a generic function.
    # Inputs: xt - First data set.
    #         yt - Second data set.
    #         k - Order of the lag.
    # Outputs: c_xy - The cross covariance.

    # Initializing c_xy.
    c_xy = 0

    # First we get the number of data points:
    num_data = len(xt)

    # There's two situations to consider: when k >= 0, and when k < 0
    if k >= 0:
        # Then, we get the appropriate partitions of the data.
        xt_k = xt.iloc[0:num_data - k, :]
        yt_k = yt.iloc[k:num_data, :]

        # All these values have the means subtracted, so we also do that.
        xt_k = xt_k - np.mean(xt)
        yt_k = yt_k - np.mean(yt)

        # We multiply these element wise, then sum them up and divide by the number of data points.
        c_xy_list = pd.DataFrame(xt_k.values * yt_k.values)
        c_xy = c_xy_list.values.sum()
        c_xy /= num_data

    # In the second situation, we have a different multiplication.
    elif k < 0:
        yt_k = yt.iloc[0:num_data + k, :]
        xt_k = xt.iloc[-k:num_data, :]

        # All these values have the means subtracted, so we also do that.
        yt_k = yt_k - np.mean(yt)
        xt_k = xt_k - np.mean(xt)

        # We multiply these element wise, then sum them up and divide by the number of data points.
        c_xy_list = pd.DataFrame(yt_k.values * xt_k.values)
        c_xy = c_xy_list.values.sum()
        c_xy /= num_data

    return c_xy


def r_xy_calc(xt, yt, k):
    # This is a generic cross-correlation function calculator. This lets me confirm
    # my results against the class examples.
    # Inputs: xt - First data set.
    #         yt - Second data set.
    #         k - Order of the lag.
    # Outputs: r_xy - The cross correlation.

    # First we need to get s_x and s_y, which are dependent on the cross covariance.
    s_x = np.sqrt(c_xy_calc(xt, xt, 0))
    s_y = np.sqrt(c_xy_calc(yt, yt, 0))

    # We also need the cross covariance at the specific value of k.
    c_xy = c_xy_calc(xt, yt, k)

    return c_xy/(s_x * s_y)


def test_prob1():
    xt = pd.DataFrame([11, 7, 8, 12, 14])
    yt = pd.DataFrame([7, 10, 6, 7, 10])
    print('Cross Correlation for k = %1i: %.3f' % (1, r_xy_calc(xt, yt, 1)))
    print('Cross Correlation for k = %1i: %.3f' % (-1, r_xy_calc(xt, yt, -1)))


def prob1():
    print('********** Problem #1 **********')
    # Creating our data sequences.
    vt = pd.DataFrame([0.9157, 0.7922, 0.9595, 0.6557, 0.0357, 0.8491, 0.9340])
    efd = pd.DataFrame([0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712, 0.7060])

    # Passing the data through each of our values k.
    for k in range(1, 4):
        print('Cross Correlation for k = %1i: %.3f' % (k, r_xy_calc(vt, efd, k)))


def transfer_function(s):
    # This defines the transfer function.
    return (s**2 + 2*s + 1)/(s**3 + 3.2*s**2 + 5*s + 2)


def bode_estimate(f):
    # This is a generic comparison function that we'll use to compare inputs and outputs.
    # Inputs: f - List of input frequencies of the sine wave.
    # Outputs: w_list - List of angular frequencies for plotting.
    #          mag_list - List of ratios of magnitudes (in dB).
    #          phase_list - List of phases (in deg).

    # Initializing our mag_list and phase_list for storing values.
    mag_list = np.zeros(f.shape)
    phase_list = np.zeros(f.shape)
    i = 0

    # First, we convert our input list of frequencies into angular frequencies.
    w_list = f * 2 * np.pi

    # Then, we evaluate the transfer function at each of these points.
    for s in w_list:
        transfer_val = transfer_function(s*1j)
        # We also find the magnitude and the angle associated with each transfer function value, which
        # we'll plot as an estimation.
        mag_list[i] = 20*np.log10(abs(transfer_val))
        phase_list[i] = np.angle(transfer_val)*180/np.pi
        i += 1

    return w_list, mag_list, phase_list


def prob2():
    print('********** Problem #2 **********')
    # We'll begin with an estimation of the transfer function bode plots. Since we only need to input a list
    # of sine waves with various frequencies, we'll generate a list between our two points of interest, 0.1 Hz
    # and 10 Hz.
    f_list = np.linspace(0.01, 100, num=10000)

    # Then we call our bode plot estimator function.
    w_list, mag_list, phase_list = bode_estimate(f_list)

    # Setting the options for the magnitude plot.
    plt.subplot(221)
    plt.semilogx(w_list, mag_list)
    plt.title('Estimated Magnitude Bode Plot')
    plt.grid(True)
    plt.xlabel('Frequency (rad/s)')
    plt.xlim(0.1, 10)
    plt.ylabel('Magnitude (dB)')
    plt.ylim(-20, -5)

    # Setting the options for the phase plot.
    plt.subplot(222)
    plt.semilogx(w_list, phase_list)
    plt.title('Estimated Phase Bode Plot')
    plt.grid(True)
    plt.xlabel('Frequency (rad/s)')
    plt.xlim(0.1, 10)
    plt.ylabel('Phase (deg)')

    # Establishing the numerators and denominator of our transfer function.
    num = [1, 2, 1]
    den = [1, 3.2, 5, 2]
    # Constructs the transfer function in Python.
    transfer = signal.TransferFunction(num, den)

    # Then calls Python's version of a bode plot maker.
    w, mag, phase = transfer.bode()

    # Setting the options for the magnitude plot.
    plt.subplot(223)
    plt.semilogx(w, mag)
    plt.title('Python Magnitude Bode Plot')
    plt.grid(True)
    plt.xlabel('Frequency (rad/s)')
    plt.xlim(0.1, 10)
    plt.ylabel('Magnitude (dB)')
    plt.ylim(-20, -5)

    # Setting the options for the phase plot.
    plt.subplot(224)
    plt.semilogx(w, phase)
    plt.title('Python Phase Bode Plot')
    plt.grid(True)
    plt.xlabel('Frequency (rad/s)')
    plt.xlim(0.1, 10)
    plt.ylabel('Phase (deg)')


if __name__ == '__main__':
    prob1()
    prob2()
    plt.show()
