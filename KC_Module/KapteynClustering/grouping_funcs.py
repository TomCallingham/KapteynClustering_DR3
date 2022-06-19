import numpy as np
from KapteynClustering import mahalanobis_funcs as mahaf
from KapteynClustering import cluster_funcs as clusterf
from KapteynClustering import linkage_funcs as linkf
import hdbscan

def get_cluster_distance_matrix(Clusters, cluster_mean, cluster_cov):
    '''Uses Summed covariance'''
    N_clusters = len(Clusters)
    dm = np.zeros((int(N_clusters*(N_clusters-1)/2)))
    dis = np.zeros((N_clusters,N_clusters))
    k=0
    for i in range(N_clusters):
        c1 = Clusters[i]
        U_mean,U_Cov  = cluster_mean[c1], cluster_cov[c1]
        for j in range(i + 1, N_clusters):
            c2 = Clusters[j]
            V_mean , V_Cov  = cluster_mean[c2], cluster_cov[c2]
            dm[k] = mahaf.find_mahalanobis(U_Cov + V_Cov, U_mean - V_mean)
            dis[i,j] = dm[k]
            k += 1
    dis = dis + dis.T
    return dis, dm

def merge_clusters_labels(Clusters, Grouped_Clusters, labels):
    groups = -np.ones_like(labels)
    for c,g in zip(Clusters, Grouped_Clusters):
        c_filt = labels==c
        groups[c_filt] = g
    groups, Groups, Pops = clusterf.order_labels(groups)

    o_Grouped_Clusters = np.empty_like(Grouped_Clusters)
    for i,c in enumerate(Clusters):
        o_Grouped_Clusters[i] = groups[labels==c][0]

    return groups, Groups, Pops, o_Grouped_Clusters

def hdbscan_hierarcy_results(Z, cluster_selection_epsilon=0):
    result = hdbscan.hdbscan_._tree_to_labels(None,Z,min_cluster_size=2, allow_single_cluster=False,
                                             cluster_selection_method="eom"
                                              , cluster_selection_epsilon= cluster_selection_epsilon)
    Grouped_Clusters, probs, stabs, condensed_tree_array = result[:-1]

    N_single = (Grouped_Clusters==-1).sum()
    N_max = np.max(Grouped_Clusters)
    Grouped_Clusters[Grouped_Clusters==-1] = N_max + 1+np.arange(N_single)
    condensed_tree = hdbscan.plots.CondensedTree(condensed_tree_array=condensed_tree_array)
    result_dic = {"GroupedClusters":Grouped_Clusters, "probs":probs, "stabs":stabs,
                  "condensed_tree": condensed_tree}
    return result_dic

def get_cluster_feh(stars, feh_key = "feh_lamost", labels=None):
    print(f"using feh: {feh_key}")
    ''' sorted, non nan fe_h'''
    if labels is None:
        labels = stars["label"].values
    feh = np.array(stars[feh_key].values)
    good_feh= ~np.isnan(feh)

    Clusters = np.unique(labels)
    c_feh = {}
    for c in Clusters:
        c_filt = (labels==c)
        c_feh[c] = np.sort(feh[c_filt*good_feh])
    return c_feh

from scipy.stats import ks_2samp
import copy
def calc_Cluster_KS(Clusters,  c_feh, N_min=10, min_val=0.5):
    print(f"Using N_min of {N_min}")

    N_Clusters = len(Clusters)
    N2_Clusters = int(N_Clusters*(N_Clusters-1)/2)
    KSs = np.zeros((N2_Clusters))
    pvals = np.zeros((N2_Clusters))
    cpair = np.empty((N2_Clusters,2), dtype=int)
    k=0
    for i in range(N_Clusters):
        c1 = Clusters[i]
        for j in range(i + 1, N_Clusters):
            c2 = Clusters[j]
            cpair[k,:] = [c1,c2]
            if (len(c_feh[c1])>N_min) and (len(c_feh[c2])>N_min):
                KSs[k], pvals[k] = ks_2samp(c_feh[c1], c_feh[c2], mode='exact')
            else:
                pvals[k] = min_val
            k+=1
    return KSs, pvals, cpair

def extend_sample(stars,labels,features = ["En","Lz","Lperp"], max_dis=2.13, plot=True):
    '''
    Returns cluster labels of the stars.
    Stars below the specified distance cut are labled.
    Stars above are given the fluff label -1
    '''
    # labels = stars["label"].values
    Clusters = np.unique(labels)
    Clusters = Clusters[Clusters!=-1]

    cluster_mean, cluster_cov =  linkf.fit_gaussian_clusters(stars, features, Clusters, labels)

    X= linkf.find_X(features, stars)
    # dis= maha_dis_to_clusters(X, Clusters, cluster_mean, cluster_cov)
    dis= mahaf.maha_dis_to_clusters(X, Clusters, cluster_mean, cluster_cov)
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
    ext_labels = copy.deepcopy(labels)
    ext_labels[dis_filt*(labels==-1)] = Clusters[i_min[dis_filt*(labels==-1)]]
    return ext_labels
