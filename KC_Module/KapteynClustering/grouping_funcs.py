import numpy as np
from KapteynClustering import mahalanobis_funcs as mahaf
from KapteynClustering import linkage_funcs as linkf

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
    return groups