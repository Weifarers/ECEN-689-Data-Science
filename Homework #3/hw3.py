import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from statsmodels.tsa import arima_model, stattools


def gamma_calc(k, zt):
    # This is a generic autocovariance function, that takes in data and the specified parameters,
    # and produces the associated covariance. Using this function makes it easy to double check
    # against simple data sets.
    # Inputs: k - the order of the lag.
    #         zt - the original data.
    # Outputs: gamma_k - Autocovariance of the data at lag k.
    # Gets the number of data points.
    num_data = len(zt)
    # Defines the Z_tk that we use in the calculation of the autocovariance. Note that since it's t+k,
    # we start at the index k, and go to the end of the data set.
    z_tk = zt.iloc[k:num_data, :]
    # Then, we multiply the data together. However, since we're starting at some time lag k, any points
    # beyond num_data are just 0, so we don't need to consider them in the multiplication.
    gamma_k_list = pd.DataFrame(zt.iloc[0:num_data - k, :].values * z_tk.values)
    gamma_k = gamma_k_list.values.sum()
    # We also divide by the number of data points.
    gamma_k /= num_data
    # Return the autocovariance.
    return gamma_k


def prob1():
    print('********** Problem #1 **********')
    # First, we import the data from the CSV. Note that there isn't a header in the file.
    time_data = pd.read_csv('TimeSeriesData.csv', header=None)

    # Since an autocorrelation function requires a detrended version of the data, we also implement that.
    full_z = time_data
    average_time = np.mean(full_z)
    full_z -= average_time

    # In order to determine the autocorrelation, we need the autocovariance first.
    # This defines the order of the lag.
    k = 2
    print('****** k = ', k, '******')
    # This passes the information through our generic function to calculate gamma.
    gamma_k = gamma_calc(k, full_z)
    print('Gamma from Equation: ', gamma_k)
    print('Rho from Equation: ', gamma_k/gamma_calc(0, full_z))

    # In order to compare results, we plot the ACF of the data, and then overlay our
    # calculated value on top.
    rho_py = stattools.acf(time_data, nlags=15, fft=False)
    print('Rho from Python: ', rho_py[2])


def test_prob1():
    print('********** Testing Autocovariance Calculator Using Data from Class Example **********')
    # As an aside, we'll use the sample data from class to confirm our function is working properly.
    test_data = pd.DataFrame([3.6923, 4.5664, 5.3426, 8.5784, 7.7694])
    # First we detrend the data.
    zt = test_data
    average_zt = np.mean(zt)
    zt -= average_zt

    # Then, we output the autocovariance and autocorrelation for each k value.
    for k in [0, 1, 2, 3]:
        print('****** k = ', k, '******')
        gamma_k = gamma_calc(k, zt)
        print('gamma = ', gamma_k)
        print('rho = ', gamma_k/gamma_calc(0, zt))


def arma_construct(p, q, x_train, y_train, x_valid, y_valid, num_data):
    # This is a generic ARMA model constructor, which is used to estimate certain ARMA models on a set of data.
    # Inputs: p - The order of the AR model.
    #         q - The order of the MA model.
    #         x_train - Partition of noise set for training.
    #         y_train - Training data evaluated using original model.
    #         x_valid - Partition of noise set for validation.
    #         y_valid - Validation data evaluated using original model.
    #         num_data - Amount of data being generated.
    # Outputs: No outputs, put does do a lot of print statements.

    # First, we create the model.
    arma = arima_model.ARMA(y_train, order=(p, q))
    arma_fit = arma.fit(disp=0)

    # Then, we convert the parameters to lists, so they match the format of the original construction.
    ma_co = arma_fit.maparams.tolist()
    ar_co = list(-arma_fit.arparams)
    # We also need to add in a 1 at the start, since those are neglected when presenting the results from
    # the ARMA function we use.
    ma_co.insert(0, 1)
    ar_co.insert(0, 1)

    # Getting the coefficients of the AR and MA models, separately.
    print('**** ARMA({0},{1}) ****'.format(p, q))
    print('AR Coefficients: ', ["%.3f" % item for item in ar_co])
    print('MA Coefficients: ', ["%.3f" % item for item in ma_co])

    # Next, we calculate the error for the model.
    # Starting with the training error.
    arma_train = signal.lfilter(ma_co, ar_co, x_train)
    ave_t_error = (1 / (num_data / 2)) * np.sum(np.abs((y_train - arma_train) / arma_train))
    print('Training Error: %.3f' % ave_t_error)

    # Then we calculate the validation error.
    arma_valid = signal.lfilter(ma_co, ar_co, x_valid)
    ave_v_error = (1 / (num_data / 2)) * np.sum(np.abs((y_valid - arma_valid) / arma_valid))
    print('Validation Error: %.3f' % ave_v_error)


def prob2a():
    print('********** Problem #2, Seed: 2020 **********')
    # First we define the model coefficients.
    ar_coe = [1, 0.5, 0.3]
    ma_coe = [1, 0.2, 0.5, 0.4, 0.1]

    # Seeding my random number generator for consistent testing.
    np.random.seed(2020)
    # Then we generate some random noise with mean 0 and standard deviation 1.
    num_data = 20000
    a_t = np.random.normal(loc=0, scale=1, size=num_data)
    # Then we split the data into training and validation sets.
    a_t_train = a_t[0:int(num_data/2)]
    a_t_valid = a_t[int(num_data/2):]

    # Then we construct the linear filter using the coefficients and the training data.
    y_arma = signal.lfilter(ma_coe, ar_coe, a_t_train)
    # We also construct it for the validation data, so we can reference later.
    y_valid = signal.lfilter(ma_coe, ar_coe, a_t_valid)
    # We plot this data, just to visualize the noise.
    plt.plot(y_arma)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('ARMA(2,4) Model, Training Data')

    # First, we plot the ACF and PACF.
    tsaplots.plot_acf(y_arma, lags=15, title='Autocorrelation for ARMA(2,4)')
    tsaplots.plot_pacf(y_arma, lags=15, title='Partial Autocorrelation for ARMA(2,4)')

    # Then, we're asked to fit a variety of ARMA models to the data, and see their accuracy.
    arma_construct(1, 1, a_t_train, y_arma, a_t_valid, y_valid, num_data)

    # Testing an ARMA(2,4) model.
    arma_construct(2, 4, a_t_train, y_arma, a_t_valid, y_valid, num_data)

    # Testing an ARMA(3,5) model.
    arma_construct(3, 5, a_t_train, y_arma, a_t_valid, y_valid, num_data)


def prob2b():
    print('********** Problem #2, Seed: 689 **********')
    # First we define the model coefficients.
    ar_coe = [1, 0.5, 0.3]
    ma_coe = [1, 0.2, 0.5, 0.4, 0.1]

    # Seeding my random number generator for consistent testing.
    np.random.seed(689)
    # Then we generate some random noise with mean 0 and standard deviation 1.
    num_data = 20000
    a_t = np.random.normal(loc=0, scale=1, size=num_data)
    # Then we split the data into training and validation sets.
    a_t_train = a_t[0:int(num_data/2)]
    a_t_valid = a_t[int(num_data/2):]

    # Then we construct the linear filter using the coefficients and the training data.
    y_arma = signal.lfilter(ma_coe, ar_coe, a_t_train)
    # We also construct it for the validation data, so we can reference later.
    y_valid = signal.lfilter(ma_coe, ar_coe, a_t_valid)

    # Then, we're asked to fit a variety of ARMA models to the data, and see their accuracy.
    arma_construct(1, 1, a_t_train, y_arma, a_t_valid, y_valid, num_data)

    # Testing an ARMA(2,4) model.
    arma_construct(2, 4, a_t_train, y_arma, a_t_valid, y_valid, num_data)

    # Testing an ARMA(3,5) model.
    arma_construct(3, 5, a_t_train, y_arma, a_t_valid, y_valid, num_data)


def prob2c():
    print('********** Problem #2, Seed: 489 **********')
    # First we define the model coefficients.
    ar_coe = [1, 0.5, 0.3]
    ma_coe = [1, 0.2, 0.5, 0.4, 0.1]

    # Seeding my random number generator for consistent testing.
    np.random.seed(489)
    # Then we generate some random noise with mean 0 and standard deviation 1.
    num_data = 20000
    a_t = np.random.normal(loc=0, scale=1, size=num_data)
    # Then we split the data into training and validation sets.
    a_t_train = a_t[0:int(num_data/2)]
    a_t_valid = a_t[int(num_data/2):]

    # Then we construct the linear filter using the coefficients and the training data.
    y_arma = signal.lfilter(ma_coe, ar_coe, a_t_train)
    # We also construct it for the validation data, so we can reference later.
    y_valid = signal.lfilter(ma_coe, ar_coe, a_t_valid)

    # Then, we're asked to fit a variety of ARMA models to the data, and see their accuracy.
    arma_construct(1, 1, a_t_train, y_arma, a_t_valid, y_valid, num_data)

    # Testing an ARMA(2,4) model.
    arma_construct(2, 4, a_t_train, y_arma, a_t_valid, y_valid, num_data)

    # Testing an ARMA(3,5) model.
    arma_construct(3, 5, a_t_train, y_arma, a_t_valid, y_valid, num_data)


def main():
    prob1()
    prob2a()
    prob2b()
    prob2c()
    plt.show()


if __name__ == '__main__':
    main()
