import numpy as np
# Normal
def fit_gaussian(X):
    mean = np.mean(X,axis=0)
    covar = np.cov(X,rowvar=0)
    return mean, covar

def find_mahalanobis(mean, covar, X):
    return np.sqrt(find_maha(X, mean=mean, cov=covar))

def find_mahalanobis_members(N_std, mean, covar, X):
    return (find_mahalanobis(mean, covar, X) < N_std)

def find_mahalanobis_N_members(N_std, mean, covar, X):
    N_members = find_mahalanobis_members(N_std, mean, covar, X).sum(axis=-1)
    return N_members

import scipy


# https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_multivariate.py
# Use Modified Scipy to find Mahalanobis distance Fast

def find_maha(x, mean=None, cov=1, allow_singular=False):
    """ maha distance
    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_mvn_doc_default_callparams)s
    Returns
    mahalanobis distance
    -------
    """
    psd = _PSD(cov, allow_singular=allow_singular)
    # dev = x - mean
    maha = np.sum(np.square(np.dot(x-mean, psd.U)), axis=-1)
    return maha

### UNCHANGED FROM SCIPY BELOW - import directly?
from scipy.stats._multivariate import _PSD

# class _PSD:
#     """
#     Compute coordinated functions of a symmetric positive semidefinite matrix.
#     This class addresses two issues.  Firstly it allows the pseudoinverse,
#     the logarithm of the pseudo-determinant, and the rank of the matrix
#     to be computed using one call to eigh instead of three.
#     Secondly it allows these functions to be computed in a way
#     that gives mutually compatible results.
#     All of the functions are computed with a common understanding as to
#     which of the eigenvalues are to be considered negligibly small.
#     The functions are designed to coordinate with scipy.linalg.pinvh()
#     but not necessarily with np.linalg.det() or with np.linalg.matrix_rank().
#     Parameters
#     ----------
#     M : array_like
#         Symmetric positive semidefinite matrix (2-D).
#     cond, rcond : float, optional
#         Cutoff for small eigenvalues.
#         Singular values smaller than rcond * largest_eigenvalue are
#         considered zero.
#         If None or -1, suitable machine precision is used.
#     lower : bool, optional
#         Whether the pertinent array data is taken from the lower
#         or upper triangle of M. (Default: lower)
#     check_finite : bool, optional
#         Whether to check that the input matrices contain only finite
#         numbers. Disabling may give a performance gain, but may result
#         in problems (crashes, non-termination) if the inputs do contain
#         infinities or NaNs.
#     allow_singular : bool, optional
#         Whether to allow a singular matrix.  (Default: True)
#     Notes
#     -----
#     The arguments are similar to those of scipy.linalg.pinvh().
#     """

#     def __init__(self, M, cond=None, rcond=None, lower=True,
#                  check_finite=True, allow_singular=True):
#         # Compute the symmetric eigendecomposition.
#         # Note that eigh takes care of array conversion, chkfinite,
#         # and assertion that the matrix is square.
#         s, u = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)

#         eps = _eigvalsh_to_eps(s, cond, rcond)
#         if np.min(s) < -eps:
#             raise ValueError('the input matrix must be positive semidefinite')
#         d = s[s > eps]
#         if len(d) < len(s) and not allow_singular:
#             raise np.linalg.LinAlgError('singular matrix')
#         s_pinv = _pinv_1d(s, eps)
#         U = np.multiply(u, np.sqrt(s_pinv))

#         # Initialize the eagerly precomputed attributes.
#         self.rank = len(d)
#         self.U = U
#         self.log_pdet = np.sum(np.log(d))

#         # Initialize an attribute to be lazily computed.
#         self._pinv = None

#     @property
#     def pinv(self):
#         if self._pinv is None:
#             self._pinv = np.dot(self.U, self.U.T)
#         return self._pinv
# def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
#     """Determine which eigenvalues are "small" given the spectrum.
#     This is for compatibility across various linear algebra functions
#     that should agree about whether or not a Hermitian matrix is numerically
#     singular and what is its numerical matrix rank.
#     This is designed to be compatible with scipy.linalg.pinvh.
#     Parameters
#     ----------
#     spectrum : 1d ndarray
#         Array of eigenvalues of a Hermitian matrix.
#     cond, rcond : float, optional
#         Cutoff for small eigenvalues.
#         Singular values smaller than rcond * largest_eigenvalue are
#         considered zero.
#         If None or -1, suitable machine precision is used.
#     Returns
#     -------
#     eps : float
#         Magnitude cutoff for numerical negligibility.
#     """
#     if rcond is not None:
#         cond = rcond
#     if cond in [None, -1]:
#         t = spectrum.dtype.char.lower()
#         factor = {'f': 1E3, 'd': 1E6}
#         cond = factor[t] * np.finfo(t).eps
#     eps = cond * np.max(abs(spectrum))
#     return eps
# def _pinv_1d(v, eps=1e-5):
#     """A helper function for computing the pseudoinverse.
#     Parameters
#     ----------
#     v : iterable of numbers
#         This may be thought of as a vector of eigenvalues or singular values.
#     eps : float
#         Values with magnitude no greater than eps are considered negligible.
#     Returns
#     -------
#     v_pinv : 1d float ndarray
#         A vector of pseudo-inverted numbers.
#     """
#     return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)


