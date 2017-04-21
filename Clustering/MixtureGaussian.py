import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats

def GaussianParaInit(K, d):
    mu = np.random.normal(loc=0, scale=2, size=K*d).reshape((K,d))
    tmp = np.random.normal(loc=0, scale=2, size=K*d*d).reshape((K,d,d))
    Sigma = np.zeros((K,d,d))   # Covariance matrix needs to be semi-pos-definite
    for k in range(K):
        Sigma[k,:,:] = np.matmul(tmp[k,:,:], tmp[k,:,:].transpose())
    return([mu, Sigma])

def ClassProbInit(K):
    pi = np.random.dirichlet([1]*K)
    return(pi)

def Convergence(theta_old, theta_new):
    sum = 0
    for i in range(len(theta_new)):
        sum += np.sum((theta_new[i]-theta_old[i]) * (theta_new[i]-theta_old[i]))
    if np.sqrt(sum) < 1e-12:
        return(True)
    else:
        return(False)


def MixtureGaussianCluster(K, data):
    n, d = np.shape(data)
    data_exp = np.expand_dims(data, axis=1)

    # parameter initialization
    mu_old, Sigma_old = GaussianParaInit(K, d)
    pi_old = ClassProbInit(K)

    Ndens = np.zeros((n, K))
    iter = 0
    while(True):
        iter += 1

        # E step
        for k in range(K):
            Ndens[:,k] = scipy.stats.multivariate_normal.pdf(data, mu_old[k,:], Sigma_old[k,:,:]) * pi_old[k]
        p = Ndens / np.expand_dims(np.sum(Ndens, axis=-1), axis=-1)

        # M step
        pi_new = np.sum(p, axis=0) / n
        mu_new = np.sum(np.expand_dims(p,axis=-1) * data_exp, axis=0) / np.expand_dims(np.sum(p, axis=0), axis=-1)
        Sigma_new = np.zeros((K,d,d))
        for k in range(K):
            Sigma_new[k,:,:] = np.matmul( (data-np.expand_dims(mu_new[k,:], axis=0)).transpose(), np.expand_dims(p[:,k], axis=-1)*(data-np.expand_dims(mu_new[k,:], axis=0)) ) / np.sum(p[:,k])

        # judge convergence
        if Convergence([pi_old, mu_old, Sigma_old], [pi_new, mu_new, Sigma_new]):
            break

        # update
        pi_old = pi_new
        mu_old = mu_new
        Sigma_old = Sigma_new

    for k in range(K):
        Ndens[:, k] = scipy.stats.multivariate_normal.pdf(data, mu_new[k, :], Sigma_new[k, :, :]) * pi_new[k]
    p = Ndens / np.expand_dims(np.sum(Ndens, axis=-1), axis=-1)
    assign = np.argmax(p, axis=-1)

    return([iter, assign])









if __name__ == '__main__':
    data = np.loadtxt("./toydata.txt")
    random.seed(100)

    iter, assign = MixtureGaussianCluster(3, data)
    print(iter)

    plt.scatter(data[assign==0, 0], data[assign==0, 1], color="red")
    plt.scatter(data[assign==1, 0], data[assign==1, 1], color="blue")
    plt.scatter(data[assign==2, 0], data[assign==2, 1], color="green")
    plt.title("Mixture Gaussian: Cluster Assignment of Points")
    plt.show()





