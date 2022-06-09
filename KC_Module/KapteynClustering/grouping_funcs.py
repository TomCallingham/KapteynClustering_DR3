from KapteynClustering import linkage_funcs as linkf
import numpy as np
import numpy as np
from KapteynClustering import mahalanobis_funcs as mahaf

def get_cluster_distance_matrix(Clusters, cluster_mean, cluster_cov):
    '''Uses Summed covariance'''
    N_clusters = len(Clusters)
    n_dim = len(cluster_mean[Clusters[0]])
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
