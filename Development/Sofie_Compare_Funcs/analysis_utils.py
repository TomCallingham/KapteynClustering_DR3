'''
Contains helper functions for analysing the results of the IOM-clustering algorithm.

SL 26.7.2021
'''

import vaex
import numpy as np
from texttable import Texttable
from tqdm.notebook import tqdm
import latextable
import membership_probability

""
def get_derived_labels(df, probability_table, minsig=3):
    '''
    Make a column with derived cluster labels according to highest membership probability.
    So here we assigne a star to the highest probability cluster, given that the 
    membership probability is at least 0.05.
    
    Parameters:
    df(vaex.DataFrame): Our star catalogue
    probability_table(vaex.DataFrame): Table containing membership probability for any star and any cluster
    minsig(float): The minimum statistical significance we want to consider
    
    Returns:
    df(vaex.DataFrame): The star catalogue with two added columns: the derived labels and the corresponding
                        membership probability for this star.
    '''
    
    #initialize array to contain derived labels
    derived_labels = np.zeros(df.count())
    derived_membership_probability = np.zeros(df.count())
    
    df2 = df[(df.labels>0) & (df.maxsig>minsig)]
    labels = np.unique(df2.labels.values).astype(int)
    colnames = [f'cluster{i}' for i in labels]
    probabilities = probability_table[colnames].values 
    
    #for every row in the probability matrix, determine the index of the maximum probability
    indices = np.argmax(probabilities, axis = 1)
    
    for i in range(df.count()):
        
        #verify that it exceeds our minimum membership probability value
        if(probabilities[i][indices[i]] > 0.05): 
            derived_labels[i] = labels[indices[i]]
            derived_membership_probability[i] = probabilities[i][indices[i]]
    
    df['derived_labels'] = derived_labels.astype(int)
    df['derived_membership_probability'] = derived_membership_probability
    
    return df

""
def get_derived_labels_substructures(df, probability_table):
    '''
    Make a column with derived substructure label according to highest membership probability.
    So here we assigne a star to the highest probability substructure, given that the 
    membership probability is at least 0.05.
    
    Parameters:
    df(vaex.DataFrame): Our star catalogue
    probability_table(vaex.DataFrame): Table containing membership probability for any star and any substructure
    
    Returns:
    df(vaex.DataFrame): The star catalogue with two added columns: the derived labels and the corresponding
                        membership probability for this star.
    '''
    
    #initialize array to contain derived labels
    derived_labels = np.zeros(df.count())
    derived_membership_probability = np.zeros(df.count())
    
    labels = np.unique(df[df.labels_substructure > 0 ].labels_substructure.values).astype(int)
    colnames = [f'cluster{i}' for i in labels if i>0]
    probabilities = probability_table[colnames].values
    print(labels)
    print(colnames)
    #for every row in the probability matrix, determine the index of the maximum probability
    indices = np.argmax(probabilities, axis = 1)
    print(np.unique(indices))
    for i in range(df.count()):
        
        #verify that it exceeds our minimum membership probability value
        if(probabilities[i][indices[i]] > 0.05): 
            derived_labels[i] = labels[indices[i]]
            derived_membership_probability[i] = probabilities[i][indices[i]]
    
    df['labels_substr_0.05'] = derived_labels.astype(int)
    df['P_substr_0.05'] = derived_membership_probability
    
    return df

""
def get_cluster_distance_matrix(df, features = ['scaled_En', 'scaled_Lperp', 'scaled_Lz', 'circ'], 
                                distance_metric = 'mahalanobis'):
    '''
    Returns a custom distance matrix which is used to apply single linkage clustering on the clusters 
    extracted in paper 1 (rather than individual data points). This distance matrix is flattened and in
    condensed format, and can be passed to scipy's linkage function in order to obtain a linkage matrix 
    and a dendrogram.
    
    Parameters:
    df(vaex.DataFrame): A slice of the star catalogue containing only the relevant clusters,
                        e.g. df[df.maxsig>3] for at least 3sigma significance structures.
    distance_metric('str'): Which distance metric to use, can be 'euclidean' (the distance between cluster a
                            and cluster b is then the smallest distance between any data point from a and any 
                            from b respectively), or 'mahalanobis' (Mahalanobis distance between the two clusters
                            represented by Gaussian distributions)
    '''
    n_sigma_labels = np.unique(df.labels.values)
    
    m = len(np.unique(df.labels.values))
    dm = np.empty((m * (m - 1)) // 2, dtype=np.double)

    k = 0
    
    for i in tqdm(range(0, m-1)):
        for j in range(i + 1, m):
            
            cluster1 = df[df.labels == n_sigma_labels[i]]
            cluster2 = df[df.labels == n_sigma_labels[j]]

            U = cluster1[features].values
            V = cluster2[features].values
            
            if(distance_metric == 'euclidean'):
                mindist = 1000000
                for u in U:
                    for v in V:
                        dist = np.linalg.norm(u-v)
                        if(dist<mindist):
                            mindist = dist
                
            elif(distance_metric == 'mahalanobis'):
                mindist = np.sqrt(membership_probability.jit_md(np.mean(U, axis=0), np.mean(V, axis=0),
                                                        np.cov(U.T), np.cov(V.T)))

            dm[k] = mindist 
            k = k + 1
        
    return dm

""
def get_cluster_description_table(df, minsig=3):
    '''
    Returns Table 1 of Lovdal et al. (2021), 
    giving summary characteristics of the clusters extracted.
    Contains cluster ID, #core_members, #total_associated_members,
    cluster mean in non-scaled clustering space, and covariance matrix entries.
    [mu0, mu1, mu2, mu3] corresponds to ['En', 'Lperp', 'Lz', 'circ'] and similarly
    [cov00, cov01, cov02, cov03] corresponds to cov[EE, ELperp, ELz, Ecirc].
    Note that the covariance matrix is symmetric, so cov03=cov30 and we omit duplicate entries.
    
    Parameters:
    df(vaex.DataFrame): Star catalogue with cluster labels
    minsig(float): Minimum statistical significance level that we want to consider
    
    Returns:
    cluster_table(vaex.DataFrame): Dataframe containing cluster characteristics
    '''
    
    features = ['scaled_En*10e2', 'scaled_Lperp*10e2', 'scaled_Lz*10e2', 'circ*10e2']
    
    unique_labels = np.unique(df[df.maxsig>3].labels.values)
    n_clusters = len(unique_labels)
                              
    core_members = np.zeros(n_clusters)
    total_members = np.zeros(n_clusters)
    significance = np.zeros(n_clusters)
    
    #Mean of clusters across features E, Lperp, Lz, circ                       
    mu = np.zeros((n_clusters, 4))
    
    #Array of covariance matrices                      
    cov = np.zeros((n_clusters, 4, 4))
                              
    for idx, label in tqdm(enumerate(unique_labels)):
        
        cluster = df[df.labels == label]
        mean = np.mean(cluster[features].values, axis=0)
        covariance = np.cov(cluster[features].values.T)
        
        mu[idx, :] = mean
        cov[idx, :, :] = covariance
        significance[idx] = cluster.maxsig.values[0]
        core_members[idx] = cluster.count()
        total_members[idx] = df[df.derived_labels == label].count()
                              
    cluster_table = vaex.from_arrays(label = unique_labels, significance = significance,
                                     core_members = core_members, total_members = total_members,
                                     mu0 = mu[:, 0], mu1 = mu[:, 1], mu2 = mu[:, 2], mu3 = mu[:, 3],
                                     cov00 = cov[:, 0, 0], cov01 = cov[:, 0, 1], cov02 = cov[:, 0, 2], 
                                     cov03 = cov[:, 0, 3],
                                     cov11 = cov[:, 1, 1], cov12 = cov[:, 1, 2], cov13 = cov[:, 1, 3],
                                     cov22 = cov[:, 2, 2], cov23 = cov[:, 2, 3],
                                     cov33 = cov[:, 3, 3])
                              
    return cluster_table

""
def get_substructure_description_table(df, names, significance):
    '''
    Returns Table X of Ruiz-Lara et al. (2021), 
    giving summary characteristics of the substructures extracted.
    Contains cluster ID, #core_members, #total_associated_members,
    cluster mean in non-scaled clustering space, and covariance matrix entries.
    [mu0, mu1, mu2, mu3] corresponds to ['En', 'Lperp', 'Lz', 'circ'] and similarly
    [cov00, cov01, cov02, cov03] corresponds to cov[EE, ELperp, ELz, Ecirc].
    Note that the covariance matrix is symmetric, so cov03=cov30 and we omit duplicate entries.
    
    Parameters:
    df(vaex.DataFrame): Star catalogue with clustering features and substructure labels
    names: List of names that substructure labels correspond to
    
    Returns:
    substructure_table(vaex.DataFrame): Dataframe containing substructure characteristics
    '''
    
    features = ['scaled_En*10e2', 'scaled_Lperp*10e2', 'scaled_Lz*10e2', 'circ*10e2']
    
    unique_labels = np.unique(df.labels_substructure.values)
    print(unique_labels)
    n_substructures = len(unique_labels) - 1
                              
    members = np.zeros(n_substructures)
    clusters = np.zeros(n_substructures)
    #Mean of clusters across features E, Lperp, Lz, circ                       
    mu = np.zeros((n_substructures, 4))
    
    #Array of covariance matrices                      
    cov = np.zeros((n_substructures, 4, 4))
                              
    for label in tqdm(range(1, max(unique_labels)+1)):
        
        cluster = df[df.labels_substructure == label]
        mean = np.mean(cluster[features].values, axis=0)
        covariance = np.cov(cluster[features].values.T)
        
        mu[label-1, :] = mean
        cov[label-1, :, :] = covariance
        members[label-1] = cluster.count()
        clusters[label-1] = len(np.unique(cluster.derived_labels.values))
                              
    substructure_table = vaex.from_arrays(name = np.array(names), significance = significance,
                                          clusters = clusters, 
                                          members = members, 
                                          mu0 = mu[:, 0], mu1 = mu[:, 1], mu2 = mu[:, 2], mu3 = mu[:, 3],
                                          cov00 = cov[:, 0, 0], cov01 = cov[:, 0, 1], cov02 = cov[:, 0, 2], 
                                          cov03 = cov[:, 0, 3],
                                          cov11 = cov[:, 1, 1], cov12 = cov[:, 1, 2], cov13 = cov[:, 1, 3],
                                          cov22 = cov[:, 2, 2], cov23 = cov[:, 2, 3],
                                          cov33 = cov[:, 3, 3])
                              
    return substructure_table

""
def print_membership_statistics(df, probability_table, minsig=3):
    '''
    Prints general statistics regarding the cluster members according to 
    membership probability
    
    Parameters:
    df(vaex.DataFrame): Our star catalogue containing cluster labels
    probability_table(vaex.DataFrame): Dataframe indicating the membership probability for any star to belong
                                       to any cluster.
    minsig(float): Minimum statistical significance level we want to consider                                   
    '''
    df2 = df[(df.labels>0) & (df.maxsig>minsig)]
    labels = np.unique(df2.labels.values).astype(int)
    colnames = [f'cluster{i}' for i in labels]
    probabilities = probability_table[colnames].values
    
    stars_within_some_clusterboundary =  np.where(np.any(probabilities > 0.05, axis = 1))

    #Get the sum over the rows, so count for each star how many associations above 0.05 it has
    res = np.sum(probabilities > 0.05, axis = 1)
    res = res[res>1]

    print(f'Number of clusters with significance > 3 sigma: {len(np.unique(df2.labels.values))}')
    print(f'Number of stars in a {minsig} sigma cluster according to the single linkage process: {df2.count()}')

    print(f'Number of stars falling within some ellipsoidal cluster boundary (3.12 sigma axes): {len(stars_within_some_clusterboundary[0])}')

    print(f'Number of stars falling within some ellipsoidal cluster boundary (3.12 sigma axes) already having a core label:  {sum(df2.membership_probability.values > 0.05)}')

    print(f'Number of stars falling within some ellipsoidal cluster boundary (3.12 sigma axes) not having a core label: {len(stars_within_some_clusterboundary[0]) - sum(df2.membership_probability.values > 0.05)}')

    print(f'Number of stars either with a core label, or having a membership probability > 0.05: {df2.count() + len(stars_within_some_clusterboundary[0]) - sum(df2.membership_probability.values > 0.05)}')

    print(f'Number of stars with more than one membership probability > 0.05: {len(res)}')
    
    print(f'Number of stars that may belong to two clusters: {len(res[res==2])}, three clusters: {len(res[res==3])}, four clusters: {len(res[res==4])}')

""
def print_velocity_space_subgroups_statistics(dispersion_subgroups, df):
    '''
    Prints information about the number of subgroups in velocity space
    
    Parameters:
    dispersion_subgroups(vaex.DataFrame): Dataframe describing the velocity dispersion of the subgroups found
                                          in velocity space.
                                          
    Returns: General statistics about the number of subgroups per cluster, 
             according to the subgroup dispersion histogram, and all subgroups (at least five members).                                     
    '''
    
    n_subgroups = []
    unique_labels = np.unique(dispersion_subgroups.labels.values)

    for label in unique_labels:
        count = dispersion_subgroups[dispersion_subgroups.labels==label].count()
        n_subgroups.append(count)

    n_subgroups = np.array(n_subgroups)    

    print('Subgroups per cluster, according to dispersion_subgroups: ')
    print(f'Min: {np.min(n_subgroups)}')
    print(f'Max: {np.max(n_subgroups)}')
    print(f'Mean: {np.mean(n_subgroups):.3f}')
    print(f'Median: {np.median(n_subgroups)}')
    print(f'Total number of subgroups: {dispersion_subgroups.count()}\n')
    
    n_subgroups = []
    unique_labels = np.unique(df.labels.values)

    for label in np.unique(df.labels.values):
        if(label==0):
            continue
        count = max(np.unique(df[df.labels==label].v_space_labels.values))
        n_subgroups.append(count)

    n_subgroups = np.array(n_subgroups)
    
    print('Subgroups per cluster, according to df (also including subgroups with <10 members): ')
    print(f'Min: {np.min(n_subgroups)}')
    print(f'Max: {np.max(n_subgroups)}')
    print(f'Mean: {np.mean(n_subgroups):.3f}')
    print(f'Median: {np.median(n_subgroups)}')
    print(f'Total number of subgroups: {np.sum(n_subgroups)}')


""
def print_star_catalogue_latex_table(df, minsig=3):
    
    table = Texttable()

    selected_cols = ['source_id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'En']
    table_values = df[df.maxsig>minsig][selected_cols].head(10).values
    datatypes = ['i'] + 6*['f'] + ['e']*1 
    
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(datatypes)
    table.set_cols_align(["l"]*len(selected_cols))
    table.add_rows(np.vstack([selected_cols, table_values]))

    print(latextable.draw_latex(table, caption='First part'))
    
    table = Texttable()
    
    selected_cols = ['Lperp/10e2', 'Lz/10e2', 'circ', 
                     'maxsig', 'labels', 'derived_labels',
                     'membership_probability', 'derived_membership_probability']
    table_values = df[df.maxsig>minsig][selected_cols].head(10).values
    datatypes = ['f']*2 + 2*['f'] + 2*['i'] + 2*['f']
    
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(datatypes)
    table.set_cols_align(["l"]*len(selected_cols))
    table.add_rows(np.vstack([selected_cols, table_values]))

    print(latextable.draw_latex(table, caption='Second part'))

""
def print_cluster_latex_table(cluster_table):
    
    table = Texttable()
    table_values = cluster_table.head(10).values

    datatypes = ['i', 'f', 'i', 'i'] + ['f']*5
    
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(datatypes)
    table.set_cols_align(["l"]*9)

    table.add_rows(np.vstack([cluster_table.get_column_names()[0:9], table_values[:, 0:9]]))
    
    print(latextable.draw_latex(table, caption='Part 1'))
    
    table = Texttable()
    datatypes = ['f']*9
    
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(datatypes)
    table.set_cols_align(["l"]*9)
    
    table.add_rows(np.vstack([cluster_table.get_column_names()[9:], table_values[:, 9:]]))
    
    print(latextable.draw_latex(table, caption='Part 2'))

""
def print_substructure_latex_table(substructure_table):
    
    colnames = substructure_table.get_column_names()

    table = Texttable()
    datatypes = ['f', 'i', 'i'] + ['f']*5

    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(datatypes)
    table.set_cols_align(["l"]*8)

    #table.add_rows(substructure_table[colnames[1:9]].values)

    table.add_rows(np.vstack([substructure_table.get_column_names()[1:9], substructure_table[colnames[1:9]].values]))

    print(latextable.draw_latex(table, caption='Part 1 (add names manually)'))

    table = Texttable()
    datatypes = ['f']*9

    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(datatypes)
    table.set_cols_align(["l"]*9)

    table.add_rows(np.vstack([substructure_table.get_column_names()[9:], substructure_table[colnames[9:]].values]))

    print(latextable.draw_latex(table, caption='Part 2'))

""
def density_within_ellipse(df, df_artificial, clusterNr, features, labelcol='labels_substructure'):

    df_members = df[df[labelcol]==clusterNr]
    
    pca = vaex.ml.PCA(features=features, n_components=len(features))

    [[En_lower, En_upper], [Lperp_lower, Lperp_upper], [Lz_lower, Lz_upper]] = \
    df_members.minmax(['scaled_En', 'scaled_Lperp', 'scaled_Lz'])

    eps = 0.05 #large enough such that ellipse fits within

    #Take out minmax-region, so that we do not have to apply PCA to the full dataset.
    region = df[(df.scaled_En>En_lower-eps) & (df.scaled_En<En_upper+eps) & 
                (df.scaled_Lperp>Lperp_lower-eps) & (df.scaled_Lperp<Lperp_upper+eps) &
                (df.scaled_Lz>Lz_lower-eps) & (df.scaled_Lz<Lz_upper+eps)]

    pca.fit(df_members)
    region = pca.transform(region)
    
    n_std = 3.12
    
    #calculate length of radius in each dimension of the 4D-ellipse
    r0 = n_std*np.sqrt(pca.eigen_values_[0])
    r1 = n_std*np.sqrt(pca.eigen_values_[1])
    r2 = n_std*np.sqrt(pca.eigen_values_[2])
    r3 = n_std*np.sqrt(pca.eigen_values_[3])

    #the stars that actually reside within the ellipse
    within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                           (np.power(region.PCA_2, 2))/(np.power(r2, 2)) + (np.power(region.PCA_3, 2))/(np.power(r3, 2))) <=1]

    region_count = within_region.count()
    
    
    #Extract neighborhood of C, so that we do not have to PCA-map the full artificial dataset  
    region = df_artificial[(df_artificial.scaled_En>En_lower-eps) & (df_artificial.scaled_En<En_upper+eps) & 
                  (df_artificial.scaled_Lperp>Lperp_lower-eps) & (df_artificial.scaled_Lperp<Lperp_upper+eps) &
                  (df_artificial.scaled_Lz>Lz_lower-eps) & (df_artificial.scaled_Lz<Lz_upper+eps)]
    region = pca.transform(region)
    
    #Extract the artificial halo stars that reside within the ellipsoid
    within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                           (np.power(region.PCA_2, 2))/(np.power(r2, 2)) + (np.power(region.PCA_3, 2))/(np.power(r3, 2))) <=1]
    
    #Average count and standard deviation in this region.
    #Vaex takes limits as non-inclusive in count(), so we add a small value to the default minmax limits
    N_datasets = 100
    region_count_art = within_region.count()/N_datasets
    counts_per_halo = within_region.count(binby="index", limits=[-0.01, N_datasets-0.99], shape=N_datasets)
    region_count_art_std = np.std(counts_per_halo)
    

    return(region_count, region_count_art, region_count_art_std)

""
def get_significance_substructures(df, df_artificial, names, features):
    
    significances = np.zeros(len(names))

    print('Measuring signficance for joined structures: ')
    for structure in list(range(1, int(max(np.unique(df.labels_substructure.values)+1)))):

        region_count, region_count_art, region_count_art_std = density_within_ellipse(df, df_artificial, structure, features)
        significance = (region_count - region_count_art) / np.sqrt(region_count + region_count_art_std**2)
        
        significances[structure-1] = significance
        
        print(f'{names[structure-1]}: {significance:.3f}')
        
    return significances  
    
