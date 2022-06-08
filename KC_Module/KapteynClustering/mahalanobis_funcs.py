from scipy.stats._multivariate import _PSD
import numpy as np
# Normal


def fit_gaussian(X):
    mean = np.mean(X, axis=0)
    covar = np.cov(X, rowvar=0, bias=True)  # 1/n, not 1/n-1
    return mean, covar


def find_mahalanobis(covar, X, psd=None, allow_singular=False):
    # https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_multivariate.py
    # Use Modified Scipy to find Mahalanobis distance Fast
    if psd is None:
        psd = _PSD(covar, allow_singular=allow_singular)
    # dev = x - mean
    # maha = np.sum(np.square(np.dot(x-mean, psd.U)), axis=-1)
    maha = np.sqrt(np.sum(np.square(np.dot(X, psd.U)), axis=-1))
    
    return maha


def find_mahalanobis_members(N_std, mean, covar, X,psd=None):
    return (find_mahalanobis(covar, X-mean, psd=psd) <= N_std)


def find_mahalanobis_N_members(N_std, mean, covar, X, psd=None):
    N_members = find_mahalanobis_members(N_std, mean, covar, X,psd=psd).sum(axis=-1)
    return N_members


def maha_dis_to_clusters(X, Clusters, cluster_mean, cluster_covar):
    N = np.shape(X)[0]
    N_clusters = len(Clusters)
    d = np.zeros((N, N_clusters))
    for i, c in enumerate(Clusters):
        mean, covar = cluster_mean[c], cluster_covar[c]
        d[:, i] = find_mahalanobis(covar, X - mean[None, :])
    return d


def maha_dis_to_fit_clusters(stars, fit_file):
    from KapteynClustering import data_funcs as dataf
    from KapteynClustering import cluster_funcs as clusterf
    fit_data = dataf.read_data(fit_file)
    features, Clusters, cluster_mean, cluster_cov = [fit_data[p] for p in ["features", "Clusters", "mean", "covariance"]]
    X= clusterf.find_X(features, stars)
    dis= maha_dis_to_clusters(X, Clusters, cluster_mean, cluster_cov)
    return dis, Clusters

def add_maha_members_to_clusters(stars, fit_file, max_dis=2, plot=False):
    '''
    Returns cluster labels of the stars.
    Stars below the specified distance cut are labled.
    Stars above are given the fluff label -1
    '''
    dis, Clusters = maha_dis_to_fit_clusters(stars,fit_file)
    N = np.shape(dis)[0]
    i_min = np.argmin(dis,axis=1)
    closest_dis = dis[np.arange(N),i_min]
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(closest_dis, bins=100)
        plt.xlabel("closestt maha dis")
        plt.show()
    dis_filt = closest_dis<max_dis
    labels = np.full((N),-1, dtype=int)
    labels[dis_filt] = Clusters[i_min[dis_filt]]
    return labels