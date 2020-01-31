import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits import mplot3d

from sklearn.decomposition import PCA


def prob_1():
    print('********** Problem #1 **********')
    # Initializing the matrix.
    a_matrix = np.array([[1, 0], [0, 2], [0, 1]])
    # Calculating A times its transpose.
    a_trans = np.transpose(a_matrix)
    aa_t = np.dot(a_matrix, a_trans)
    # The eigenvectors of this matrix make up U, so as a sanity check for
    # the actual work, we'll see what the eigenvalues and eigenvectors are.
    eigen_val, eigen_vec = np.linalg.eig(aa_t)
    print('Eigenvalues: ', eigen_val, '\n', 'Eigenvectors: ', '\n', eigen_vec)

    # Then we create the V matrix.
    v_matrix = np.dot(a_trans, a_matrix)
    # Gets the eigenvalues and eigenvectors.
    v_eigen_val, v_eigen_vec = np.linalg.eig(v_matrix)
    print('Eigenvalues: ', v_eigen_val, '\n', 'Eigenvectors: ', '\n', v_eigen_vec)

    # This is a sanity check for my final results.
    svd_results = np.linalg.svd(a_matrix)
    print('U = ', '\n', svd_results[0], '\n',
          'S = ', '\n', np.diag(svd_results[1]), '\n',
          'V = ',  '\n', np.transpose(svd_results[2]))

    # Also hard coding in my handwritten results to check.
    u_written = np.array([[0, 1, 0],
                          [2/np.sqrt(5), 0, 1/np.sqrt(5)],
                          [1/np.sqrt(5), 0, -2/np.sqrt(5)]])
    v_written = np.array([[0, 1], [1, 0]])
    s_written = np.array([[np.sqrt(5), 0], [0, 1], [0, 0]])
    result_written = np.dot(np.dot(u_written, s_written), np.transpose(v_written))
    print('Sanity Check: ', '\n', result_written)


# Generic PCA Function.
def pca(x):
    # Inputs: x - Input matrix we want to apply PCA to.
    # Outputs: p - The P matrix used to rotate the original data set.
    #          eigen_vals - This measures which direction is the most important.
    # First we calculate the covariance matrix.
    cov_x = np.dot(x, x.T) / (x.shape[1] - 1)
    # Then we get the eigenvalues and eigenvectors of the covariance matrix.
    eigen_vals, eigen_vecs = np.linalg.eig(cov_x)
    # Just to make sure, we use argsort to ensure that the order of the
    # eigenvalues and eigenvectors goes form greatest to least.
    sort_idx = eigen_vals.argsort()[::-1]
    eigen_vals = eigen_vals[sort_idx]
    eigen_vecs = eigen_vecs[:, sort_idx]

    return eigen_vals, eigen_vecs.T


def prob_2():
    print('********** Problem #2 **********')
    # First, we'll generate some random data to test on. This is similar data as
    # problem 3 in the future, but with a higher covariance.
    # Seeding my random number generators for consistent testing.
    np.random.seed(2020)
    mean = [1, 1]
    cov = [[1, 0.99], [0.99, 1]]
    x, y = np.random.multivariate_normal(mean, cov, 5000).T
    plt.figure()
    plt.scatter(x, y)
    # Setting all the options.
    plt.title('Randomly Generated Data with Mean [1,1] and Cov [[1, 0.99], [0.99, 1]]')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    # Since we generated two sets of data, we need to combine them.
    orig_x = np.vstack((x, y))
    # First we need to detrend the data along the rows. We do this by taking the
    # average across the rows, and then subtract that from the current values.
    # Making a copy of the original data to manipulate.
    full_x = orig_x
    # Then we take the average of the rows.
    average_x = np.mean(full_x, axis=1)
    # After, we reshape the matrix to be a column one, and then subtract it.
    average_x = average_x.reshape((average_x.shape[0], 1))
    full_x -= average_x

    # Then, we pass this new data through our pca function.
    eigen_vals, p = pca(full_x)
    # We can look at the ratio of the variances, which we will use to compare
    # to SciKit's implementation of PCA.
    variance_percent = eigen_vals/np.sum(eigen_vals) * 100

    # Next we call the SciKit's PCA implementation.
    sci_pca = PCA()
    sci_pca.fit(full_x.T)
    print('Variance Ratios from Manual PCA: ', variance_percent)
    print('Variance Ratios from SciKit PCA: ', sci_pca.explained_variance_ratio_ * 100)


def prob_3():
    print('********** Problem #3 **********')
    # First, we'll generate some random data to test on.
    mean = [1, 1]
    cov = [[1, 0.9], [0.9, 1]]
    np.random.seed(2020)
    x, y = np.random.multivariate_normal(mean, cov, 5000).T
    # Plotting the original data.
    plt.figure()
    plt.scatter(x, y)
    # Setting all the options.
    plt.title('Randomly Generated Data with Mean [1,1] and Cov [[1, 0.9], [0.9, 1]]')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    # Next we call the PCA function we wrote, which is functionally identical to SciKit's implementation.
    # Since we generated two sets of data, we need to combine them.
    orig_x = np.vstack((x, y))
    # First we need to detrend the data along the rows. We do this by taking the
    # average across the rows, and then subtract that from the current values.
    # Making a copy of the original data to manipulate.
    full_x = orig_x
    # Then we take the average of the rows.
    average_x = np.mean(full_x, axis=1)
    # After, we reshape the matrix to be a column one, and then subtract it.
    average_x = average_x.reshape((average_x.shape[0], 1))
    full_x -= average_x

    # Then, we pass this new data through our pca function.
    eigen_vals, p = pca(full_x)
    # Printing the principal coefficients.
    print('Principal Component Coefficients: ', '\n', p)
    # Then we transform the data, and plot it onto another figure.
    pca_y = np.dot(p, orig_x)
    plt.figure()
    plt.scatter(pca_y[0], pca_y[1])
    # Setting all the options for the second plot.
    plt.title('Randomly Generated Data in Principal Component Space')
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    # We can look at the ratio of the variances
    variance_percent = eigen_vals / np.sum(eigen_vals) * 100
    print('Variance Ratios from Manual PCA: ', variance_percent)


def prob_4():
    print('********** Problem #4 **********')
    # First, we import the data into a Pandas DataFrame.
    curr_path = os.getcwd()
    power_data = pd.read_csv(curr_path + '\Results - VS Test Case 02 Measurements.csv')

    # We'll also need to filter out all the bad data; if I recall in the demo,
    # the bad data was actually removed. But, we check the data and note that there's
    # no bad data, so that's nice of them.
    bad_data_check = power_data.isnull().sum().sum()
    if bad_data_check != 0:
        print('Bad data!')
    else:
        print('No bad data!')

    # We'll select two of the measurements; real power and reactive power.
    # Since all of the real and reactive powers start with the string 'BR P'
    # 'and 'BR Q', we'll just take out all indices involving that.
    time_series = power_data['Time']
    # Iteratively checks all the keys, and stores the data in separate dataframes.
    # This is a pretty crude way of forming the data I want, but that's okay.
    real_power = pd.DataFrame()
    reactive_power = pd.DataFrame()
    for key in power_data.keys():
        if 'BR P' in key:
            real_power[key] = power_data[key]
        elif 'BR Q' in key:
            reactive_power[key] = power_data[key]
        else:
            continue

    # We're also asked to partition the data between before the voltage collapse and
    # afterwards.
    pre_real_orig = real_power.iloc[0:1999]
    post_real_orig = real_power.iloc[2100:3099]
    pre_react_orig = reactive_power.iloc[0:1999]
    post_react_orig = reactive_power.iloc[2100:3099]

    # First, it'll be interesting to plot the data to see what it looks like.
    data_fig, data_axs = plt.subplots(2, 2)
    # Plotting pre-collapse real power.
    data_axs[0, 0].plot(time_series.iloc[0:1999], pre_real_orig)
    data_axs[0, 0].set_title('Pre Collapse P')
    data_axs[0, 0].set(xlabel='Time (s)', ylabel='Real Power (MW)')
    # Plotting post-collapse real power.
    data_axs[0, 1].plot(time_series.iloc[2100:3099], post_real_orig)
    data_axs[0, 1].set_title('Post Collapse P')
    data_axs[0, 1].set(xlabel='Time (s)', ylabel='Real Power (MW)')

    # Plotting pre-collapse reactive power.
    data_axs[1, 0].plot(time_series.iloc[0:1999], pre_react_orig)
    data_axs[1, 0].set_title('Pre Collapse Q')
    data_axs[1, 0].set(xlabel='Time (s)', ylabel='Reactive Power (Mvar)')
    # Plotting post-collapse reactive power.
    data_axs[1, 1].plot(time_series.iloc[2100:3099], post_react_orig)
    data_axs[1, 1].set_title('Post Collapse Q')
    data_axs[1, 1].set(xlabel='Time (s)', ylabel='Reactive Power (Mvar)')

    # Next, we apply PCA to the data, and we'll do all the analysis
    # step by step. But first, we need to do data pre-processing.
    # First we calculate the mean along the rows of the transposed matrix.
    pre_real_mean = np.mean(pre_real_orig.T, axis=1)
    # Then we subtract it from the data.
    pre_real_mean = pre_real_mean.values.reshape((pre_real_mean.shape[0], 1))
    pre_real = pre_real_orig.T - pre_real_mean
    # Then, we pass it through our PCA function.
    pre_real_eigen, pre_real_p = pca(pre_real)

    # First we calculate the mean along the rows of the transposed matrix.
    post_real_mean = np.mean(post_real_orig.T, axis=1)
    # Then we subtract it from the data.
    post_real_mean = post_real_mean.values.reshape((post_real_mean.shape[0], 1))
    post_real = post_real_orig.T - post_real_mean
    # Then, we pass it through our PCA function.
    post_real_eigen, post_real_p = pca(post_real)

    # First we calculate the mean along the rows of the transposed matrix.
    pre_react_mean = np.mean(pre_react_orig.T, axis=1)
    # Then we subtract it from the data.
    pre_react_mean = pre_react_mean.values.reshape((pre_react_mean.shape[0], 1))
    pre_react = pre_react_orig.T - pre_react_mean
    # Then, we pass it through our PCA function.
    pre_react_eigen, pre_react_p = pca(pre_react)

    # First we calculate the mean along the rows of the transposed matrix.
    post_react_mean = np.mean(post_react_orig.T, axis=1)
    # Then we subtract it from the data.
    post_react_mean = post_react_mean.values.reshape((post_react_mean.shape[0], 1))
    post_react = post_react_orig.T - post_react_mean
    # Then, we pass it through our PCA function.
    post_react_eigen, post_react_p = pca(post_react)

    # First we plot the actual variances first.
    v_fig, v_axs = plt.subplots(2, 2)
    v_axs[0, 0].bar(range(1, pre_real_eigen.shape[0] + 1), np.real(pre_real_eigen))
    v_axs[0, 0].set_title('Pre Collapse P Variance')
    v_axs[0, 0].set(xlabel='Principal Component', ylabel='Variance')

    v_axs[0, 1].bar(range(1, post_real_eigen.shape[0] + 1), np.real(post_real_eigen))
    v_axs[0, 1].set_title('Post Collapse P Variance %')
    v_axs[0, 1].set(xlabel='Principal Component', ylabel='Variance')

    v_axs[1, 0].bar(range(1, pre_react_eigen.shape[0] + 1), np.real(pre_react_eigen))
    v_axs[1, 0].set_title('Pre Collapse Q Variance')
    v_axs[1, 0].set(xlabel='Principal Component', ylabel='Variance')

    v_axs[1, 1].bar(range(1, post_react_eigen.shape[0] + 1), np.real(post_react_eigen))
    v_axs[1, 1].set_title('Post Collapse Q Variance')
    v_axs[1, 1].set(xlabel='Principal Component', ylabel='Variance')

    # Next, it's interesting to plot the variance percentages.
    pre_real_vp = pre_real_eigen/np.sum(pre_real_eigen) * 100
    post_real_vp = post_real_eigen / np.sum(post_real_eigen) * 100
    pre_react_vp = pre_react_eigen / np.sum(pre_react_eigen) * 100
    post_react_vp = post_react_eigen / np.sum(post_react_eigen) * 100

    vp_fig, vp_axs = plt.subplots(2, 2)
    vp_axs[0, 0].bar(range(1, pre_real_vp.shape[0]+1), np.real(pre_real_vp))
    vp_axs[0, 0].set_title('Pre Collapse P Variance %')
    vp_axs[0, 0].set(xlabel='Principal Component', ylabel='Variance %')

    vp_axs[0, 1].bar(range(1, post_real_vp.shape[0] + 1), np.real(post_real_vp))
    vp_axs[0, 1].set_title('Post Collapse P Variance %')
    vp_axs[0, 1].set(xlabel='Principal Component', ylabel='Variance %')

    vp_axs[1, 0].bar(range(1, pre_react_vp.shape[0] + 1), np.real(pre_react_vp))
    vp_axs[1, 0].set_title('Pre Collapse Q Variance %')
    vp_axs[1, 0].set(xlabel='Principal Component', ylabel='Variance %')

    vp_axs[1, 1].bar(range(1, post_react_vp.shape[0] + 1), np.real(post_react_vp))
    vp_axs[1, 1].set_title('Post Collapse Q Variance %')
    vp_axs[1, 1].set(xlabel='Principal Component', ylabel='Variance %')

    # Then we consider the cumulative sum, and calculate it.
    pre_real_cs = np.cumsum(pre_real_vp)
    post_real_cs = np.cumsum(post_real_vp)
    pre_react_cs = np.cumsum(pre_react_vp)
    post_react_cs = np.cumsum(post_react_vp)

    # Then we visualize the cumulative sum.
    vp_fig, vp_axs = plt.subplots(2, 2)
    vp_axs[0, 0].plot(range(1, pre_real_cs.shape[0] + 1), np.real(pre_real_cs))
    vp_axs[0, 0].set_title('Pre Collapse P Cumulative Sum of Variance %')
    vp_axs[0, 0].set(xlabel='Principal Component', ylabel='Cumulative Sum of Variance %')

    vp_axs[0, 1].plot(range(1, post_real_cs.shape[0] + 1), np.real(post_real_cs))
    vp_axs[0, 1].set_title('Post Collapse P Cumulative Sum of Variance %')
    vp_axs[0, 1].set(xlabel='Principal Component', ylabel='Cumulative Sum of Variance %')

    vp_axs[1, 0].plot(range(1, pre_react_cs.shape[0] + 1), np.real(pre_react_cs))
    vp_axs[1, 0].set_title('Pre Collapse Q Cumulative Sum of Variance %')
    vp_axs[1, 0].set(xlabel='Principal Component', ylabel='Cumulative Sum of Variance %')

    vp_axs[1, 1].plot(range(1, post_react_cs.shape[0] + 1), np.real(post_react_cs))
    vp_axs[1, 1].set_title('Post Collapse Q Cumulative Sum of Variance %')
    vp_axs[1, 1].set(xlabel='Principal Component', ylabel='Cumulative Sum of Variance %')

    # Then we project all the data into our new reference frame. Since we note that the
    # data is primarily dominated by its first two principal components, we only need to
    # plot relative to those two dimensions.
    pre_real_y = np.dot(pre_real_p, pre_real_orig.T)
    post_real_y = np.dot(post_real_p, pre_real_orig.T)
    pre_react_y = np.dot(pre_react_p, pre_react_orig.T)
    post_react_y = np.dot(post_react_p, post_react_orig.T)

    y_fig, y_axs = plt.subplots(2, 2)
    y_axs[0, 0].scatter(pre_real_y[0], pre_real_y[1])
    y_axs[0, 0].set_title('Pre Collapse P in the Principal Component Space')
    y_axs[0, 0].set(xlabel='PC1', ylabel='PC2')
    y_axs[0, 1].scatter(post_real_y[0], post_real_y[1])
    y_axs[0, 1].set_title('Post Collapse P in the Principal Component Space')
    y_axs[0, 1].set(xlabel='PC1', ylabel='PC2')
    y_axs[1, 0].scatter(pre_react_y[0], pre_react_y[1])
    y_axs[1, 0].set_title('Pre Collapse Q in the Principal Component Space')
    y_axs[1, 0].set(xlabel='PC1', ylabel='PC2')
    y_axs[1, 1].scatter(post_react_y[0], post_react_y[1])
    y_axs[1, 1].set_title('Post Collapse Q in the Principal Component Space')
    y_axs[1, 1].set(xlabel='PC1', ylabel='PC2')

    # The reactive power seems to need more than 2 dimensions to visualize,
    # so we'll try a 3D scatter plot on it.
    y_3d_fig = plt.figure()
    y_3d_pre = y_3d_fig.add_subplot(221, projection='3d')
    y_3d_pre.scatter(pre_react_y[0], pre_react_y[1], pre_react_y[2])
    y_3d_pre.set_title('Pre Collapse Q in the 3D Principal Component Space')
    y_3d_pre.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
    y_3d_post = y_3d_fig.add_subplot(222, projection='3d')
    y_3d_post.scatter(np.real(post_react_y[0]), np.real(post_react_y[1]), np.real(post_react_y[2]))
    y_3d_post.set_title('Post Collapse Q in the 3D Principal Component Space')
    y_3d_post.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
    y_3d_pre = y_3d_fig.add_subplot(223, projection='3d')
    y_3d_pre.scatter(pre_react_y[0], pre_react_y[1], pre_react_y[2])
    y_3d_pre.set_title('Pre Collapse Q in the 3D Principal Component Space')
    y_3d_pre.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
    y_3d_post = y_3d_fig.add_subplot(224, projection='3d')
    y_3d_post.scatter(np.real(post_react_y[0]), np.real(post_react_y[1]), np.real(post_react_y[2]))
    y_3d_post.set_title('Post Collapse Q in the 3D Principal Component Space')
    y_3d_post.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')


def main():
    prob_1()
    prob_2()
    prob_3()
    prob_4()
    # Show all the plots.
    plt.show()


if __name__ == "__main__":
    main()
