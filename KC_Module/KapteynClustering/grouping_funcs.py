from KapteynClustering import cluster_funcs as clusterf
import numpy as np

def get_cluster_distance_matrix(df, features,  Groups=None, distance_metric="mahalanobis"):
    '''
    Assumes that df has been passed as df[df.maxsig>3]
    The single linkage algorithm links groups with the smallest distance between any data point from
    group one and two respectively. In order to remove the problem with overcrowding with data points
    in the dendrogram, we customize a distance matrix which reflects the relationship between the clusters
    according to your chosen distance metric.
    '''
    labels = df["labels"].values
    X = clusterf.find_X(features, df, scaled=False)
    if Groups is None:
        print("Finding Groups")
        Groups = np.unique(labels)


    print(f"Using {distance_metric} metric")
    if distance_metric == "euclidean":
        dm = get_cluster_euclidean_distance_matrix(X, labels, Groups)
    elif distance_metric == "mahalanobis":
        dm = get_cluster_mahalanobis_distance_matrix(X, labels, Groups)
    else:
        raise ValueError("distance metric is not recognised")

    print("Finished")
    return dm, Groups


def get_cluster_euclidean_distance_matrix(X, labels, Groups):
    N = len(Groups)
    dm = np.zeros((N * (N - 1)) // 2)

    X_group = {}
    for g in Groups:
        X_group[g] = X[(labels == g), :]

    k = 0
    for i in range(0, N - 1):
        g1 = Groups[i]
        U = X_group[g1]
        for j in range(i + 1, N):
            g2 = Groups[j]
            V = X_group[g2]
            dist = np.linalg.norm(U[:, None, :] - V[None, :, :], axis=2)
            mindist = np.min(dist)
            dm[k] = mindist
            k += 1
    return dm


def get_cluster_mahalanobis_distance_matrix(X, labels, Groups):
    from KapteynClustering import mahalanobis_funcs as mahaf
    N = len(Groups)
    dm = np.zeros((N * (N - 1)) // 2)

    mean_group = {}
    cov_group = {}
    # print("Needs bias correct!")
    for g in Groups:
        g_filt = (labels == g)
        # mean_group[g], cov_group[g] = mahaf.fit_gaussian_bias(X[g_filt, :])
        mean_group[g] , cov_group[g] = mahaf.fit_gaussian(X[g_filt,:])

    k = 0
    for i in range(0, N - 1):
        g1 = Groups[i]
        U_mean = mean_group[g1]
        U_Cov = cov_group[g1]
        for j in range(i + 1, N):
            g2 = Groups[j]
            V_mean = mean_group[g2]
            V_Cov = cov_group[g2]
            dm[k] = mahaf.find_mahalanobis(U_Cov + V_Cov, U_mean - V_mean)
            k += 1
    return dm

