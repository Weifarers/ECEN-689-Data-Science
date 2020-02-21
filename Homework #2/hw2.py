import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np


def prob1():
    # First, we import the data from the .mat format.
    p1_dict = scipy.io.loadmat('problem1.mat')
    # Since it's stored in dictionary format but we only care about the data,
    # we grab just the data portion of the dictionary.
    p1_data = p1_dict['data']
    # As a sanity check, we split the data into the two columns, and plot it to confirm.
    p1_x = p1_data[:, 0]
    p1_y = p1_data[:, 1]
    plt.scatter(p1_x, p1_y)
    plt.title('Plot of Original Data')
    plt.xlabel('x')
    plt.ylabel('y')
    # First, we generate a kmeans clustering algorithm with 2 clusters, and change the number of
    # centroid seeds and max iterations, just to see what the parameters do.
    kmeans_1 = KMeans(n_clusters=2, n_init=1, max_iter=1)
    kmeans_1_res = kmeans_1.fit_predict(p1_data)

    kmeans_2 = KMeans(n_clusters=2, n_init=100, max_iter=1)
    kmeans_2_res = kmeans_2.fit_predict(p1_data)

    kmeans_3 = KMeans(n_clusters=2, n_init=1, max_iter=500)
    kmeans_3_res = kmeans_3.fit_predict(p1_data)

    kmeans_4 = KMeans(n_clusters=2, n_init=100, max_iter=500)
    kmeans_4_res = kmeans_4.fit_predict(p1_data)

    # We'll compare some different instances of kmeans, just to see how the results will differ.
    kmeans_fig, kmeans_axs = plt.subplots(2, 2)
    kmeans_axs[0, 0].scatter(p1_x, p1_y, c=kmeans_1_res)
    kmeans_axs[0, 0].set_title('n_init = 1, max_iter = 1')
    kmeans_axs[0, 0].set(xlabel='x', ylabel='y')

    kmeans_axs[0, 1].scatter(p1_x, p1_y, c=kmeans_2_res)
    kmeans_axs[0, 1].set_title('n_init = 100, max_iter = 1')
    kmeans_axs[0, 1].set(xlabel='x', ylabel='y')

    kmeans_axs[1, 0].scatter(p1_x, p1_y, c=kmeans_3_res)
    kmeans_axs[1, 0].set_title('n_init = 1, max_iter = 500')
    kmeans_axs[1, 0].set(xlabel='x', ylabel='y')

    kmeans_axs[1, 1].scatter(p1_x, p1_y, c=kmeans_4_res)
    kmeans_axs[1, 1].set_title('n_init = 100, max_iter = 500')
    kmeans_axs[1, 1].set(xlabel='x', ylabel='y')


def prob2():
    # First, we import the data from the .mat format.
    p2_dict = scipy.io.loadmat('problem2.mat')
    # Since it's stored in dictionary format, we need the data and the labels.
    p2_data = p2_dict['data']
    p2_label = p2_dict['label']
    # As a sanity check, we split the data into the two columns, and plot it to confirm.
    p2_x = p2_data[:, 0]
    p2_y = p2_data[:, 1]
    plt.figure()
    plt.scatter(p2_x, p2_y, c=p2_label.T[0])
    plt.title('Plot of Original Data')
    plt.xlabel('x')
    plt.ylabel('y')

    # Now, we fit a bunch of different SVM kernels, to see how it is.
    svc_poly = svm.SVC(kernel='poly', degree=3)
    svc_poly.fit(p2_data, p2_label.T[0])

    svc_linear = svm.SVC(kernel='linear')
    svc_linear.fit(p2_data, p2_label.T[0])

    svc_rbf = svm.SVC(kernel='rbf')
    svc_rbf.fit(p2_data, p2_label.T[0])

    svc_sigmoid = svm.SVC(kernel='sigmoid')
    svc_sigmoid.fit(p2_data, p2_label.T[0])

    # We'll plot the boundaries using a color contour method. I didn't come up
    # with this method myself, but it definitely works out really well.
    x_min, x_max = p2_x.min() - 1, p2_x.max() + 1
    y_min, y_max = p2_y.min() - 1, p2_y.max() + 1
    # In order to implement this idea, you need to set up a meshgrid to
    # contour over, that includes all your points.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    svc_fig, svc_axs = plt.subplots(2, 2)

    # Then, you make a prediction on that data set with the associated labels.
    svc_poly_pred = svc_poly.predict(np.c_[xx.ravel(), yy.ravel()])
    # Reshaping the data to match the structure necessary for plotting.
    svc_poly_pred = svc_poly_pred.reshape(xx.shape)

    svc_linear_pred = svc_linear.predict(np.c_[xx.ravel(), yy.ravel()])
    svc_linear_pred = svc_linear_pred.reshape(xx.shape)

    svc_rbf_pred = svc_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
    svc_rbf_pred = svc_rbf_pred.reshape(xx.shape)

    svc_sigmoid_pred = svc_sigmoid.predict(np.c_[xx.ravel(), yy.ravel()])
    svc_sigmoid_pred = svc_sigmoid_pred.reshape(xx.shape)

    # Put the result into a color plot.
    svc_axs[0, 0].contourf(xx, yy, svc_poly_pred, cmap=plt.cm.coolwarm, alpha=0.8)
    # Plot also the training points
    svc_axs[0, 0].scatter(p2_x, p2_y, c=p2_label.T[0], cmap=plt.cm.coolwarm)
    svc_axs[0, 0].set_title('Polynomial Kernel with Degree 3')

    # Put the result into a color plot.
    svc_axs[0, 1].contourf(xx, yy, svc_linear_pred, cmap=plt.cm.coolwarm, alpha=0.8)
    # Plot also the training points
    svc_axs[0, 1].scatter(p2_x, p2_y, c=p2_label.T[0], cmap=plt.cm.coolwarm)
    svc_axs[0, 1].set_title('Linear Kernel')

    # Put the result into a color plot.
    svc_axs[1, 0].contourf(xx, yy, svc_rbf_pred, cmap=plt.cm.coolwarm, alpha=0.8)
    # Plot also the training points
    svc_axs[1, 0].scatter(p2_x, p2_y, c=p2_label.T[0], cmap=plt.cm.coolwarm)
    svc_axs[1, 0].set_title('RBF Kernel')

    # Put the result into a color plot.
    svc_axs[1, 1].contourf(xx, yy, svc_sigmoid_pred, cmap=plt.cm.coolwarm, alpha=0.8)
    # Plot also the training points
    svc_axs[1, 1].scatter(p2_x, p2_y, c=p2_label.T[0], cmap=plt.cm.coolwarm)
    svc_axs[1, 1].set_title('Sigmoid Kernel')


def main():
    prob1()
    prob2()
    plt.show()


if __name__ == '__main__':
    main()
