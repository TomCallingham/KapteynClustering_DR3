import cluster_utils
import numpy as np
import vaex
def original_s_expected_density_parallel(idx, dic):
    art_region_count, art_region_count_std = original_s_only_expected_density_parallel(idx, dic)
    region_count = original_s_density_within_ellipse(idx, dic)
    return np.array([region_count, art_region_count, art_region_count_std])

def original_s_only_expected_density_parallel(idx,dic):
    features = ["En", "Lz", "Lperp"]
    '''
    The input parameter idx specifies the thread index of the parallel execution, and 
    thread i applies the function to the cluster formed at step i of the single linkage algorithm.
    
    The function applies PCA to a cluster C.
    We then define an ellipsoidal boundary around C, where the axes lengths of the ellipsoid 
    are chosen to be n standard deviations of spread along each axis (n=N_sigma_ellipse_axis).
    We then map stars from the artificial halo to the PCA-space defined by C and use the standard
    equation of an ellipse to check how many artificial halo stars fall within the ellipsoid defined by C.
    
    Parameters:
    idx(int): Thread index of the parallel execution
    
    Returns:
    [region_count, region_count_std](np.array): The number of stars in the artificial halo
                                                in the region of cluster i, and the standard
                                                deviation in counts across the artificial
                                                halo data sets.
    '''
    Z = dic["Z"]
    min_members = dic["min_members"]
    max_members = dic["max_members"]
    df = dic["df"]
    df_artificial = dic["df_artificial"]
    N_sigma_ellipse_axis = dic["N_sigma_ellipse_axis"]
    N_datasets = dic["N_datasets"]
    
    #get a list of members (indices in our halo set) of the cluster C we are currently investigating
    members = cluster_utils.get_members(idx, Z)
    
    #ignore these clusters, return placeholder (done for computational efficiency)
    if((len(members)>max_members) or (len(members)<min_members)):
        # print(len(members))
        # print(min_members,max_members)
        return(np.array([len(members), 0]))
    
    #Get a dataframe containing the members
    df_members = df.take(np.array(members))
    
    #Fit a PCA-object according to the members of cluster C
    pca = vaex.ml.PCA(features=features, n_components=len(features))
    pca.fit(df_members)
    
    #TODO: This is still hardcoded according to four features
    [[En_lower, En_upper], [Lperp_lower, Lperp_upper], [Lz_lower, Lz_upper]] = \
    df_members.minmax(['En', 'Lperp', 'Lz'])

    eps = 0.05 #large enough such that ellipse fits within

    #Extract neighborhood of C, so that we do not have to PCA-map the full artificial dataset  
    region = df_artificial[(df_artificial.En>En_lower-eps) & (df_artificial.En<En_upper+eps) & 
                  (df_artificial.Lperp>Lperp_lower-eps) & (df_artificial.Lperp<Lperp_upper+eps) &
                  (df_artificial.Lz>Lz_lower-eps) & (df_artificial.Lz<Lz_upper+eps)]
    
    #Map the stars in the artificial data set to the PCA-space defined by C
    region = pca.transform(region)
    
    #calculate the length of the axis in each dimension of the 4D-ellipsoid
    n_std = N_sigma_ellipse_axis
    
    #TODO: Here we could automate the expression and number of axes we need depending on which features we use.
    r0 = n_std*np.sqrt(pca.eigen_values_[0])
    r1 = n_std*np.sqrt(pca.eigen_values_[1])
    r2 = n_std*np.sqrt(pca.eigen_values_[2])
    #r3 = n_std*np.sqrt(pca.eigen_values_[3])

    #Extract the artificial halo stars that reside within the ellipsoid
    within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                           (np.power(region.PCA_2, 2))/(np.power(r2, 2)) #+ (np.power(region.PCA_3, 2))/(np.power(r3, 2))
                           ) <=1]
    
    #Average count and standard deviation in this region.
    #Vaex takes limits as non-inclusive in count(), so we add a small value to the default minmax limits
    region_count = within_region.count()/N_datasets
    counts_per_halo = within_region.count(binby="index", limits=[-0.01, N_datasets-0.99], shape=N_datasets)
    region_count_std = np.std(counts_per_halo)

    return(np.array([region_count, region_count_std]))

def original_s_density_within_ellipse(idx,dic):
    features = ["En", "Lz", "Lperp"]
    Z = dic["Z"]
    min_members = dic["min_members"]
    max_members = dic["max_members"]
    df = dic["df"]
    df_artificial = dic["df_artificial"]
    N_sigma_ellipse_axis = dic["N_sigma_ellipse_axis"]
    N_datasets = dic["N_datasets"]
    '''
    Applies PCA to a cluster to find the principal components a_i.
    Then defines cluster region as the ellipse e defined by a_i, and counts the number of
    stars in this region in the artificial data set.
    '''
    
    members = cluster_utils.get_members(idx, Z)
    
    #ignore these clusters, return placeholder (done for computational efficiency)
    if((len(members)>max_members) or (len(members)<min_members)):
        region_count = 0
        return(region_count)
    
    pca = vaex.ml.PCA(features=features, n_components=len(features))
    df_members = df.take(np.array(members))

    [[En_lower, En_upper], [Lperp_lower, Lperp_upper], [Lz_lower, Lz_upper]] = \
    df_members.minmax(['En', 'Lperp', 'Lz'])

    eps = 0.05 #large enough such that ellipse fits within

    #Take out minmax-region, so that we do not have to apply PCA to the full dataset.
    region = df[(df.En>En_lower-eps) & (df.En<En_upper+eps) & 
                (df.Lperp>Lperp_lower-eps) & (df.Lperp<Lperp_upper+eps) &
                (df.Lz>Lz_lower-eps) & (df.Lz<Lz_upper+eps)]

    pca.fit(df_members)
    region = pca.transform(region)
    
    n_std = N_sigma_ellipse_axis
    
    #calculate length of radius in each dimension of the 4D-ellipse
    r0 = n_std*np.sqrt(pca.eigen_values_[0])
    r1 = n_std*np.sqrt(pca.eigen_values_[1])
    r2 = n_std*np.sqrt(pca.eigen_values_[2])
    #r3 = n_std*np.sqrt(pca.eigen_values_[3])

    #the stars that actually reside within the ellipse
    within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                           (np.power(region.PCA_2, 2))/(np.power(r2, 2)) #+ (np.power(region.PCA_3, 2))/(np.power(r3, 2))
                           ) <=1]

    region_count = within_region.count()

    return(region_count)
