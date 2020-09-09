
import sklearn.utils.extmath
import sklearn.utils.validation
import scipy.linalg
import scipy.sparse
import numpy
import warnings


def randomized_svd(M, n_components, *, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=0):
    """ This code is a copy of the randomized_svd function from scikit-learn library, modified to handle a complex matrix M.
    (sklearn.utils.extmath.randomized_svd)

    Computes a truncated randomized SVD

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.

    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.

        .. versionchanged:: 0.18

    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign : boolean, (True by default)
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 https://arxiv.org/abs/0909.4061

    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    if isinstance(M, (scipy.sparse.lil_matrix, scipy.sparse.dok_matrix)):
        warnings.warn("Calculating SVD of a {} is expensive. "
                      "csr_matrix is more efficient.".format(
                          type(M).__name__),
                      scipy.sparse.SparseEfficiencyWarning)

    random_state = sklearn.utils.validation.check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(
        M, size=n_random, n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    B = sklearn.utils.extmath.safe_sparse_dot(Q.T.conj(), M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = scipy.linalg.svd(B, full_matrices=False)

    del B
    U = numpy.dot(Q, Uhat)

    if flip_sign:
        if not transpose:
            U, V = sklearn.utils.extmath.svd_flip(U, V)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, V = sklearn.utils.extmath.svd_flip(U, V, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]

def randomized_range_finder(A, *, size, n_iter,
                            power_iteration_normalizer='auto',
                            random_state=None):
    """This code is a copy of the randomized_range_finder function in the scikit-learn library modified to handle complex input A.
    (sklearn.utils.extmath.randomized_range_finder)




    Computes an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A : 2D array
        The input data matrix

    size : integer
        Size of the return array

    n_iter : integer
        Number of power iterations used to stabilize the result

    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    Q : 2D array
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    random_state = sklearn.utils.validation.check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = random_state.normal(size=(A.shape[1], size))
    if A.dtype.kind == 'f':
        # Ensure f32 is preserved as f32
        Q = Q.astype(A.dtype, copy=False)

    # Deal with "auto" mode
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        else:
            power_iteration_normalizer = 'LU'

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        if power_iteration_normalizer == 'none':
            Q = sklearn.utils.extmath.safe_sparse_dot(A, Q)
            Q = sklearn.utils.extmath.safe_sparse_dot(A.T.conj(), Q)
        elif power_iteration_normalizer == 'LU':
            Q, _ = scipy.linalg.lu(sklearn.utils.extmath.safe_sparse_dot(A, Q), permute_l=True)
            Q, _ = scipy.linalg.lu(sklearn.utils.extmath.safe_sparse_dot(A.T.conj(), Q), permute_l=True)
        elif power_iteration_normalizer == 'QR':
            Q, _ = scipy.linalg.qr(sklearn.utils.extmath.safe_sparse_dot(A, Q), mode='economic')
            Q, _ = scipy.linalg.qr(sklearn.utils.extmath.safe_sparse_dot(A.T.conj(), Q), mode='economic')

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = scipy.linalg.qr(sklearn.utils.extmath.safe_sparse_dot(A, Q), mode='economic')
    return Q

