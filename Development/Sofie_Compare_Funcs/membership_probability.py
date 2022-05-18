'''
Utility functions relating to computing membership probability

This module contains utility functions relating to the extreme deconvolution (XDGMM) algorithm,
mahalanobis distances, and computing membership probability.

Sofie L
06/04/2021
'''

import numpy as np
import vaex
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal, chi2, norm
from numba import jit
#from xdgmm import XDGMM
from tqdm.notebook import tqdm

""
def mahalanobis_distance(m1, m2, cov1, cov2):
    '''
    Computes the squared mahalanobis distance between two distributions.
    
    Parameters:
    m1(np.array): mean vector of first distribution
    m2(np.array): mean vector of second distribution
    cov1(np.ndarray): full covariance matrix of first distribution
    cov2(np.ndarray): full covariance matrix of second distribution
    
    Returns:
    md(float): the squared mahalanobis distance between the two distributions
    '''
    a = (m2-m1)
    b = np.linalg.inv(cov1+cov2)
    return a.dot(b).dot(a.T)

""
jit_md = jit(mahalanobis_distance, nopython=True, fastmath=True, cache=True)

""
def get_means(df, features=['scaled_En', 'scaled_Lperp', 'scaled_Lz', 'circ']):
    '''
    Returns a numpy array containing the means of each source in the dataset
    
    Parameters:
    df(vaex.DataFrame): Dataframe containing our clustering features and their uncertainties
    features([str]): The clustering features of choice,
                     default: pertaining to the clustering of the extended sample
    
    Returns:
    means(np.ndarray): A 2D array containing the mean vectors of our clustering features
    '''

    means = df[features].values
    
    return means

""
def get_covariance_matrix_names(features=['scaled_En', 'scaled_Lperp', 'scaled_Lz', 'circ']):
    '''
    Returns a 4X4 matrix containing what the features corresponding to each covariance or 
    uncertainty (standard deviation) are called in the vaex dataframe. We use this in order 
    to extract arrays of the full covariance matrices fast, to avoid string comparison 
    and other slow operations.
    
    Parameters:
    features([str]): List of the features we want to consider
    
    Returns:
    cov_mat_names([[str]]): Matrix of the same shape as a covariance matrix, but containing the 
                            names of the corresponding quantities in our vaex data frame
    '''
    
    if('scaled_Lperp' in features):
        cov_mat_names = \
        [['scaled_En_uncertainty', 'scaled_Lperp_scaled_En_covariance', 
          'scaled_Lz_scaled_En_covariance', 'circ_scaled_En_covariance'],
         ['scaled_Lperp_scaled_En_covariance', 'scaled_Lperp_uncertainty',
          'scaled_Lz_scaled_Lperp_covariance', 'circ_scaled_Lperp_covariance'],
         ['scaled_Lz_scaled_En_covariance', 'scaled_Lz_scaled_Lperp_covariance',
          'scaled_Lz_uncertainty', 'circ_scaled_Lz_covariance'],
         ['circ_scaled_En_covariance', 'circ_scaled_Lperp_covariance',
          'circ_scaled_Lz_covariance', 'circ_uncertainty']]
    
    elif('scaled_Ltotal' in features):
        cov_m_names = \
        [['scaled_En_uncertainty', 'scaled_Ltotal_scaled_En_covariance',
          'scaled_Lz_scaled_En_covariance', 'circ_scaled_En_covariance'],
         ['scaled_Ltotal_scaled_En_covariance', 'scaled_Ltotal_uncertainty',
          'scaled_Lz_scaled_Ltotal_covariance', 'circ_scaled_Ltotal_covariance'],
         ['scaled_Lz_scaled_En_covariance', 'scaled_Lz_scaled_Ltotal_covariance',
          'scaled_Lz_uncertainty', 'circ_scaled_Lz_covariance'],
         ['circ_scaled_En_covariance', 'circ_scaled_Ltotal_covariance',
          'circ_scaled_Lz_covariance', 'circ_uncertainty']]
        
    else:
        print('Something went wrong. Are you passing the correct list of features?\n \
               Currently supported: scaled_En, scaled_Ltotal, scaled_Lz, circ (corresponds to thesis) \
               and scaled_En, scaled_Lperp, scaled_Lz, circ (corresponds to extended sample)')
        
    return cov_mat_names    

""
def get_covariances(cluster, features=['scaled_En', 'scaled_Lperp', 'scaled_Lz', 'circ']):
    '''
    Returns the covariance matrices of each source of a cluster.
    
    Parameters:
    cluster(vaex.DataFrame): Slice of a dataframe corresponding to a cluster
    features([str]): The clustering features, default corresponds to the extended sample
    
    Returns:
    cov_matrices(np.ndarray): Array containing full covariance matrices of all the stars in cluster
    '''
    dims = len(features)
    cov_matrices = np.zeros((cluster.count(), dims, dims))
    
    cov_mat_names = get_covariance_matrix_names(features)   
        
    for i in range(dims):
        for j in range(i, dims):
            if(i==j): #add a variance
                cov_matrices[:, i, j] = cluster[cov_mat_names[i][j]].values**2
            else: #add a covariance
                value = cluster[cov_mat_names[i][j]].values
                cov_matrices[:, i, j] = value
                cov_matrices[:, j, i] = value
                
    return cov_matrices


""
def squared_mah_dist_from_likelihood(X, Xerr, xdgmm, dims=4):
    '''
    Computes the mahalanobis distance d corresponding to the distance between a sample and the 
    cluster distribution when treating each star and cluster as a multivariate Gaussian PDF.
    Calculates d given the per-sample-likelihood outputted by the fitted XDGMM model.
    
    Parameters:
    X(np.ndarray): A 2D array containing the means of the PDFs (observed values of the stars)
    Xerr(np.ndarray): A 3D array containing the full covariance matrices of each data point in X
    xdgmm(XDGMM): The fitted extreme deconvolution model
    dims(int): The number of dimensions of the data points
    
    Returns:
    squared_mahalanobis_distances(np.array): The squared mahalanobis distance between each sample in X
                                             and the cluster distribution contained in the pre-fitted
                                             XDGMM model, according to equation 25 of the thesis
    '''
    logprobs, responsibilities = xdgmm.score_samples(X, Xerr)
    
    det_cov = np.linalg.det(xdgmm.V[0])
    pi_term = np.power((2*np.pi), dims/2)
    squared_mahalanobis_distances = -2*(logprobs + np.log(pi_term) + np.log(np.sqrt(det_cov)))
    
    return squared_mahalanobis_distances

""
def get_model():
    '''
    Instantiates an XDGMM model to be fitted on one component
    
    Returns:
    xdgmm(xdgmm.XDGMM): Model for extreme deconvolution
    '''
    xdgmm = XDGMM()
    xdgmm.n_components = 1
    xdgmm.n_iter = 1000000
    xdgmm.tol = 1e-8
    
    return xdgmm

""
def get_confidence_intervals(squared_mahalanobis_distances, df=4):
    '''
    Returns how much of the volume of the PDF is located outside the radius of
    where the data point is located, as a sort of confidence interval.
    E.g. if the data point is located at two standard deviations distance (for a 1D case),
    then this return value will be 0.05, aka 95% of the data points are closer to the mean
    and 5% are located further away.
    
    Parameters:
    squared_mahalanobis_distances(np.array): Array containing the squared mahalanobis distances representing
                                             the distance between a star and the cluster distribution
                                             
    df(int): Number of degrees of freedom, should equal the number of clustering features
    
    Returns:
    conf_intervals(np.array): Array indicating for each data point the ratio of the distribution that 
                              falls further away than the (squared) mahalanobis distance indicated in the input
    '''
    conf_intervals = 1 - chi2(df=df).cdf(squared_mahalanobis_distances)
    
    return conf_intervals

""
def confidence_ellipse(x, cov, i, j, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Draws an ellipse.
    
    Parameters:
    x(np.array): Mean vector of a multivariate gaussian
    cov(np.ndarray): Full covariance matrix of the multivariate gaussian
    i, j: indices of features to consider
    
    Returns:
    A patch added to the specified axis.
    """

    pearson = cov[i, j]/np.sqrt(cov[i, i] * cov[j, j])
    
    #in some rare cases (probably) rounding issues are causing the pearson coefficient
    #to minimally exceed 1, which should not be possible, so round this down.
    if(pearson>1):
        pearson = 0.999999
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl visualization data.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='dodgerblue', alpha=0.15, lw=1, edgecolor='black')

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[i, i]) * n_std
    mean_x = np.mean(x[i])

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[j, j]) * n_std
    mean_y = np.mean(x[j])

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)

""
def plot_cluster_subspaces(xdgmm, X, Xerr, confidences, features, savepath=None):
    '''
    TODO: Plot subspaces in same order as IOM-clusters.
    
    Plots a cluster for each combination of clustering features, displaying the cluster 
    covariance and individual uncertainty for each data point.
    
    Parameters:
    xdgmm(xdgmm.XDGMM): A fitted XDGMM model
    X(np.ndarray): The means of the contributing members
    Xerr(np.ndarray): An array containing the full covariance matrix of each cluster member
    confidences(np.array): Membership probability: The ratio of the distribution falling 
                           further away than the current data point
    features([str]): List of clustering features, used for axis labels
    savepath(str): Path to save location of plot, if None the plot is only displayed
    '''
    cmap = plt.get_cmap('Blues')
    fig, axs = plt.subplots(2,3, figsize = [21,12])
    
    mu = xdgmm.mu
    cov = xdgmm.V
    
    c = 0
    for i in range(4):
        for j in range(i+1, 4):
            
            #draw ellipse for each star (1 sigma along each 1D axis)
            for star in np.arange(len(X)):
                confidence_ellipse(X[star], Xerr[star], i, j, ax=axs[int(c/3), c%3], n_std=1.0,
                                    alpha=0.8, facecolor=cmap(confidences[i]), 
                                   edgecolor='black', zorder=2)
            
            #draw ellipse for the cluster (2 sigma along each 1D axis)
            clab = 'C'+str(2)
            confidence_ellipse(mu[0], cov[0], i, j, ax=axs[int(c/3), c%3], n_std=2.0,
                                alpha=1.0, facecolor=clab, fill=False, edgecolor='black',
                                zorder=3, lw=2)
            
            axs[int(c/3), c%3].scatter(X[:,i], X[:,j], s=40, alpha=1.0, zorder=1,
                                       cmap=cmap, c=confidences, edgecolors='black')
            
            axs[int(c/3), c%3].set_xlabel(features[i])
            axs[int(c/3), c%3].set_ylabel(features[j])
            c = c + 1
    
    if(savepath is not None):
        plt.savefig(savepath, dpi=300)  
    else:
        plt.show()

""
def get_membership_indication_PCA(df, features=['scaled_En', 'scaled_Lperp', 'scaled_Lz', 'circ']):
    '''
    Returns the membership probability for all stars that are a member of some cluster,
    using only PCA, ignoring measurement uncertainties.
    
    Parameters:
    df(vaex.DataFrame): The halo set with clustering features and labels
    features([str]): List of clustering features
    
    Returns:
    df(vaex.DataFrame): The input data frame containing an extra column
                        indicating membership probability of the stars 
                        assigned to some cluster, and the corresponding
                        squared Mahalanobis distance
    '''
    degrees_freedom = len(features)
    membership_probability = np.zeros(df.count())
    unique_labels = np.unique(df.labels.values)
    D_squared = np.zeros(df.count())
    
    for label in tqdm(range(1, int(max(unique_labels)+1))):
        
        cluster = df[df.labels == label]
      
        cluster_mean = cluster.mean(features)
        cluster_covariance_matrix = cluster.cov(features)
        
        VI = np.linalg.inv(cluster_covariance_matrix)
        
        delta = cluster[features].values - cluster_mean
        
        squared_mahalanobis_distances = np.zeros(cluster.count())
        
        for i in range(cluster.count()):
            m = np.dot(np.dot(delta[i], VI), delta[i].T)
            squared_mahalanobis_distances[i] = m
        
        confidences = get_confidence_intervals(squared_mahalanobis_distances, df=degrees_freedom)
        
        membership_probability[np.where(df.labels.values==label)] = confidences
        D_squared[np.where(df.labels.values==label)] = squared_mahalanobis_distances
        
    df[f'membership_probability'] = membership_probability
    df[f'D_squared'] = D_squared
    
    return df        

""
def get_membership_indication(df, features=['scaled_En', 'scaled_Lperp', 'scaled_Lz', 'circ'], plot=False):
    
    membership_probability = np.zeros(df.count())
    
    unique_labels = np.unique(df.labels.values)
    
    xdgmm = get_model()
    
    for label in tqdm(range(1, max(unique_labels)+1)):
        
        cluster = df[df.labels == label]
        cluster_count = cluster.count()
        
        X = get_means(cluster, features)
        Xerr = get_covariances(cluster, features)
        
        xdgmm.fit(X, Xerr)
        
        squared_mahalanobis_distances = squared_mah_dist_from_likelihood(X, Xerr, xdgmm, dims=4)
        
        confidences = get_confidence_intervals(squared_mahalanobis_distances, df=4)
        print(f'Cluster {label} - Possible outliers: {np.sum(confidences<0.05)} / {cluster_count}')
        
        membership_probability[np.where(df.labels.values==label)] = confidences
        
        if(plot==True):
            plot_cluster_subspaces(xdgmm, X, Xerr, confidences, features, savepath=None)
    
    df['membership_probability'] = membership_probability
    return df

""
def plot_chi_square_distribution(df=4, kind='pdf', savepath=None):
    '''
    Makes a pretty plot of the chi-square distribution with df degrees of freedom.
    Has been used for illustrative purposes in the thesis mostly.
    You can choose between either the probability density function or
    cumulative distribution function.
    '''
    fig, ax = plt.subplots(1, 1, figsize = (7, 4))

    if(kind=='pdf'):
        
        x = np.linspace(chi2.ppf(0.00, df), chi2.ppf(0.999, df), 100)
        ax.plot(x, chi2.pdf(x, df), 'r-', lw=2, alpha=0.9, c='forestgreen', label='$\chi^2$ pdf')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$f_n(x)$')
        ax.legend()
    
    elif(kind=='cdf'):

        x = np.linspace(chi2.ppf(0.00, df), chi2.ppf(0.999, df), 100)

        ax.plot(x, chi2.cdf(x, df), 'r-', lw=2.0, alpha=0.9, c='forestgreen', label='$\chi^2$ cdf')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$F_n(x)$')

        ax.vlines([4.722, 9.49, 16.01], ymin=[0, 0, 0], ymax=[0.68, 0.95, 0.997], colors='k', linestyles='dotted')

        section = np.linspace(chi2.ppf(0.00, df), chi2.ppf(0.68, df), 100)
        ax.fill_between(section, chi2.cdf(section, df), color='blue', alpha=0.5)

        section = np.linspace(chi2.ppf(0.68, df), chi2.ppf(0.95, df), 100)
        ax.fill_between(section, chi2.cdf(section, df), color='blue', alpha=0.35)

        section = np.linspace(chi2.ppf(0.95, df), chi2.ppf(0.997, df), 100)
        ax.fill_between(section, chi2.cdf(section, df), color='blue', alpha=0.2)

        section = np.linspace(chi2.ppf(0.997, df), chi2.ppf(0.999, df), 100)
        ax.fill_between(section, chi2.cdf(section, df), color='blue', alpha=0.1)

        ax.legend()

    if(savepath is not None):
        plt.savefig(savepath)
    else:
        plt.show()

""
def get_probability_table_PCA(df, features = ['scaled_En', 'scaled_Lperp', 'scaled_Lz', 'circ'], labelcol='labels'):
    '''
    Computes membership probability between any star and any cluster
    as the ratio of a Gaussian distribution which theoretically falls 
    further away than the data point, acccording to its mahalanobis distance 
    and the cumulative chi-square distribution.
    
    Parameters:
    df(vaex.DataFrame): The halo catalogue
    features([str]): The clustering features
    labelcol(str): Which labels to use, default = labels (labels for the original members of clusters according to Paper I)
              can also be labels_substructure (for labels of the merged clusters, according to Paper II)
    
    Returns:
    probability_table(vaex.DataFrame): Membership probability between each star and each cluster 
                                       (or substructure, depending on labelcol)
    '''
    
    unique_labels = np.unique(df[labelcol].values)
    probability_table = vaex.from_arrays()
    squared_mahalanobis_distances = np.zeros(df.count())
    
    for label in tqdm(range(1, int(max(unique_labels)+1))):
        
        cluster = df[df[labelcol] == label]
        cluster_mean = cluster.mean(features)
        cluster_covariance_matrix = cluster.cov(features)
        
        VI = np.linalg.inv(cluster_covariance_matrix)
        
        delta = df[features].values - cluster_mean
        
        for i in range(df.count()):
            m = np.dot(np.dot(delta[i], VI), delta[i].T)
            squared_mahalanobis_distances[i] = m
    
        membership_probability = get_confidence_intervals(squared_mahalanobis_distances, df=len(features))
        probability_table[f'cluster{label}'] = membership_probability

    return probability_table
