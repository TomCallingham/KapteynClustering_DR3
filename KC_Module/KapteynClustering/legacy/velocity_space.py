# '''Utility functions relating to computing subgroups in velocity space

# This module contains functions for computing subgroups in velocity space,
# as well as analysing their dispersion. Plotting utilities for the subgroups
# can be found in plotting_utils.

# Sofie L
# 09/06/2021
# '''

# import hdbscan
# import vaex, vaex.ml
# import numpy as np

""
# def apply_HDBSCAN_vspace(df, labelcol='labels', min_cluster_size=6, min_samples=1):
#     ''''
#     Applies HBSCAN to the the vR, vphi and vz-components of each cluster in df.
#     Inserts an addition column describing subgroup membership.
#     The parameters are set according to
#     min_cluster_size = max(3, int(0.05*stars.count())
#                       (a subgroup needs to contain at least 5% of the stars of a cluster,
#                        but at the minimum three stars)
#     min_samples = 1 (resulting in being lenient on possible noise)
#     allow_single_cluster = True if and only if the standard deviation in at least two polar velocity
#                            velocity components is small (<25km/s). This is a heuristic check to see if
#                            a cluster seems to be a single blob or string.

#     Parameters:
#     df(vaex.DataFrame): Input dataframe containing cluster labels and velocity components
#     labelcol(str): Column indicating the labels of the clusters. Can be e.g. labels (labels of
#                    original clusters according to Lovdal et al. (2021), or the labels_substructure
#                    (labels after grouping smaller clusters together accoring to Ruiz-Lara et al (2021)),
#                    or derived_labels(labels according to >0.05 membership probability).
#     min_cluster_size(int): Smallest accepted size of a subgroup in velocity space
#     min_samples(int): Smallest number of neighbors for a data point to be considered a
#                       core point. Higher values will give more data points classified as noise.

#     Returns:
#     df(vaex.DataFrame): Input dataframe with an additional column indicating velocity space
#                         subgroup membership as a positive integer. Zero indicates
#                         that the star is not in a subgroup.

#     '''
#     v_space_labels = np.zeros(df.count()) * np.nan

#     #Loop over clusters and apply HDBSCAN separately to each one of them.
#     for i in range(1, int(max(np.unique(df[labelcol].values)+1))):

#         stars = df[df[labelcol]==i]
#         N_members = stars.count()

#         X = stars['vR', 'vphi', 'vz'].values

#         single_cluster_bool = False
#         stds_1D = np.array([np.std(stars.vR.values), np.std(stars.vphi.values), np.std(stars.vz.values)])
#         if (sum(stds_1D < 25) >= 2):
#             single_cluster_bool = True

#         #proposed approach
#         clusterer = hdbscan.HDBSCAN(allow_single_cluster = single_cluster_bool,
#                                     min_cluster_size = max(3, int(0.05*stars.count())),
#                                     min_samples = 1)

#         clusterer.fit(X)

#         v_space_labels[np.where(df[labelcol].values==i)] = clusterer.labels_

#     #HDBSCAN gives -1 as not in cluster, but we put 0 for conformity with IOM labels
#     v_space_labels = v_space_labels + 1
#     v_space_labels = np.nan_to_num(v_space_labels)

#     if('substructure' in labelcol):
#         df['v_space_labels_substructure'] = v_space_labels.astype(int)
#     else:
#         df['v_space_labels'] = v_space_labels.astype(int)

#     return df

""
# def get_velocity_dispersions(df):
#     '''
#     Computes the dispersion in velocity components of each cluster,
#     by applying PCA to the xyz-velocity components.
#     Currently does this for all stars of a cluster, but keep in mind
#     that for a larger volume the dispersions may be containing
#     velocity gradients, making the dispersions larger than we would expect for a stream.

#     Parameters:
#     df(vaex.DataFrame): Input dataframe containing cluster labels and velocity components

#     Returns:
#     df_dispersions(vaex.DataFrame): Dataframe containing the velocity dispersion along each
#                                     principal component. The column labels indicates cluster.

#     '''
#     pca = vaex.ml.PCA(features=['vx', 'vy', 'vz'], n_components=3)
#     num_labels = len(np.unique(df.labels.values))

#     dispersion_xyz = np.zeros((num_labels-1, 3))

#     for i in np.unique(df.labels.values):
#         if(i==0):
#             continue #0 is not a cluster

#         cluster = df[(df.labels == i)]
#         pca.fit(cluster)
#         cluster = pca.transform(cluster)
#         dispersion_xyz[i-1, 0] = np.std(cluster.PCA_0.values)
#         dispersion_xyz[i-1, 1] = np.std(cluster.PCA_1.values)
#         dispersion_xyz[i-1, 2] = np.std(cluster.PCA_2.values)

#     df_dispersions = vaex.from_arrays(
#                     labels = np.arange(1, num_labels), std_PCA_0 = dispersion_xyz[:, 0],
#                     std_PCA_1 = dispersion_xyz[:, 1], std_PCA_2 = dispersion_xyz[:, 2])

#     return df_dispersions

""
# def get_velocity_dispersions_subgroups(df, minpoints=10):
#     '''
#     Computes velocity dispersions for each subgroup in velocity space by applying PCA on the
#     xyz-velocity components and measuring the standard deviation along each principal component.

#     Parameters:
#     df(vaex.DataFrame): Dataframe containing cluster labels and sublabels in velocity space
#     minpoints(int): The smallest size of a subgroup for which we want to consider the velocity dispersion

#     Returns:
#     df_v_subgroup_dispersions(vaex.DataFrame): The dispersion for each subgroup
#                                                in velocity space along each principal component
#                                                (std_PCA_n_subgroup), together with cluster label (labels),
#                                                subgroup label (v_space_labels),
#                                                size of the subgroup (subgroup_count) and norm of the
#                                                standard deviation across all PCAs (std_PCA_tot).
#     '''

#     pca = vaex.ml.PCA(features=['vx', 'vy', 'vz'], n_components=3)
#     num_labels = len(np.unique(df.labels.values))

#     dispersion_xyz = []
#     cluster_idx = []
#     counts = []
#     subgroup_idx = []
#     total_velocity_dispersion = []

#     for i in np.unique(df.labels.values):
#         if(i==0):
#             continue #0 is not a cluster

#         cluster = df[(df.labels == i)]
#         vspace_sublabels = np.unique(cluster.v_space_labels.values)

#         for j in range(1, max(vspace_sublabels)+1):
#             cluster_slice = cluster[cluster.v_space_labels == j]

#             if(cluster_slice.count() >= minpoints):
#                 pca.fit(cluster_slice)
#                 cluster_slice = pca.transform(cluster_slice)
#                 dispersion_xyz.append([np.std(cluster_slice.PCA_0.values),
#                                        np.std(cluster_slice.PCA_1.values),
#                                        np.std(cluster_slice.PCA_2.values)])

#                 cluster_idx.append(i)
#                 counts.append(cluster_slice.count())
#                 subgroup_idx.append(j)

#     dispersion_xyz = np.array(dispersion_xyz)
#     df_v_subgroup_dispersions = vaex.from_arrays(
#                                     labels = np.array(cluster_idx).astype(int), v_space_labels = np.array(subgroup_idx),
#                                     subgroup_count = np.array(counts), std_PCA_0_subgroup = dispersion_xyz[:, 0],
#                                     std_PCA_1_subgroup = dispersion_xyz[:, 1], std_PCA_2_subgroup = dispersion_xyz[:, 2],
#                                     std_PCA_tot = np.sqrt(dispersion_xyz[:, 0]**2+dispersion_xyz[:, 1]**2+dispersion_xyz[:, 2]**2))

#     return df_v_subgroup_dispersions

""
