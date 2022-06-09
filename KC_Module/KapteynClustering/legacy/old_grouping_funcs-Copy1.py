from KapteynClustering import linkage_funcs as linkf
import numpy as np

def get_cluster_distance_matrix(df, features,  Clusters=None, distance_metric="mahalanobis"):
    '''
    Assumes that df has been passed as df[df.maxsig>3]
    The single linkage algorithm links clusters with the smallest distance between any data point from
    cluster one and two respectively. In order to remove the problem with overcrowding with data points
    in the dendrogram, we customize a distance matrix which reflects the relationship between the clusters
    according to your chosen distance metric.
    '''
    labels = df["labels"].values
    X = linkf.find_X(features, df, scaled=False)
    if Clusters is None:
        print("Finding Clusters")
        Clusters = np.unique(labels)


    print(f"Using {distance_metric} metric")
    if distance_metric == "euclidean":
        dm = get_cluster_euclidean_distance_matrix(X, labels, Clusters)
    elif distance_metric == "mahalanobis":
        dm = get_cluster_mahalanobis_distance_matrix(X, labels, Clusters)
    else:
        raise ValueError("distance metric is not recognised")

    print("Finished")
    return dm, Clusters


def get_cluster_euclidean_distance_matrix(X, labels, Clusters):
    N = len(Clusters)
    dm = np.zeros((N * (N - 1)) // 2)

    X_cluster = {}
    for g in Clusters:
        X_cluster[g] = X[(labels == g), :]

    k = 0
    for i in range(0, N - 1):
        g1 = Clusters[i]
        U = X_cluster[g1]
        for j in range(i + 1, N):
            g2 = Clusters[j]
            V = X_cluster[g2]
            dist = np.linalg.norm(U[:, None, :] - V[None, :, :], axis=2)
            mindist = np.min(dist)
            dm[k] = mindist
            k += 1
    return dm


def get_cluster_mahalanobis_distance_matrix(X, labels, Clusters):
    from KapteynClustering import mahalanobis_funcs as mahaf
    N = len(Clusters)
    dm = np.zeros((N * (N - 1)) // 2)

    mean_cluster = {}
    cov_cluster = {}
    # print("Needs bias correct!")
    for g in Clusters:
        g_filt = (labels == g)
        # mean_cluster[g], cov_cluster[g] = mahaf.fit_gaussian_bias(X[g_filt, :])
        mean_cluster[g] , cov_cluster[g] = mahaf.fit_gaussian(X[g_filt,:])

    k = 0
    for i in range(0, N - 1):
        g1 = Clusters[i]
        U_mean = mean_cluster[g1]
        U_Cov = cov_cluster[g1]
        for j in range(i + 1, N):
            g2 = Clusters[j]
            V_mean = mean_cluster[g2]
            V_Cov = cov_cluster[g2]
            dm[k] = mahaf.find_mahalanobis(U_Cov + V_Cov, U_mean - V_mean)
            k += 1
    return dm

