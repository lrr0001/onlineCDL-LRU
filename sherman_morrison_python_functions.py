import sporco.linalg
import sporco.admm.admm
import sporco.admm.cbpdn
import sporco.cnvrep
import sporco.prox
import numpy

def lowRankApprox(a, projIter=2,axisu=2,axisv=3,dimN=2):
    r""" This code constructs an approximation of a using a sum of a pair of low-rank terms. (Notably, their sum is not low-rank.)
    a = eps + u1*v1H + u2*v2H
    u1: (1,1,K,1)
    u2: (N1,N2,K,1)
    v1H: (N1,N2,1,L)
    v2H: (1,1,1,L)
    
    computation time increases linearly with projIter, but the approximation improves.
    dimN specifies how many N-dimensions
    axisu specifies the u-axis
    axisv specifies the v-axis
    example sizes above use default parameters.
    """

    # projections preform Rubinstien's SVD method for K-SVD. There's probably some fancy name for the algorithm, but I don't know it.
    axisN = range(0,dimN)
    u1 = numpy.mean(a,tuple(axisN) + (axisv,),keepdims=True)
    unorm = numpy.sqrt(numpy.sum(numpy.conj(u1)*u1))
    u1 = u1/unorm
    for ii in range(0,projIter):
        v1H = sporco.linalg.inner(numpy.conj(u1),a,axisu)
        u1 = numpy.mean(sporco.linalg.inner(a,numpy.conj(v1H),axisv),tuple(axisN),keepdims=True)
    unorm = numpy.sqrt(numpy.sum(numpy.conj(u1)*u1))
    u1 = u1/unorm
    v1H = sporco.linalg.inner(numpy.conj(u1),a,axisu)


    v2H = numpy.mean(a,tuple(axisN) + (axisu,),keepdims=True) - numpy.mean(u1,(axisu,),keepdims=True)*numpy.mean(v1H, tuple(axisN), keepdims=True)
    vnorm = numpy.sqrt(numpy.sum(numpy.conj(v2H)*v2H))
    v2H = v2H/vnorm
    for ii in range(0,projIter):    
        u2 = sporco.linalg.inner(a,numpy.conj(v2H),axisv)
        u2 = u2 - sporco.linalg.inner(numpy.conj(u1),u2,axisu)*u1
        v2H = numpy.mean(sporco.linalg.inner(numpy.conj(u2),a,axisu),tuple(axisN),keepdims=True)
    vnorm = numpy.sqrt(numpy.sum(numpy.conj(v2H)*v2H))
    v2H = v2H/vnorm
    u2 = sporco.linalg.inner(a,numpy.conj(v2H),axisv)
    u2 = u2 - sporco.linalg.inner(numpy.conj(u1),u2,axisu)*u1
    return ((u1,u2),(v1H,v2H),u1*v1H + u2*v2H)


def mdbi_dict_sm_r1update(ainv, dtu, utu, vt, dimN=2):
    r"""
    Compute the updated a multiple diagonal block inverse after
    a rank-one update for each block using the Sherman-Morrison
    equation. The computation yields an :math:`O(M^2)` time cost and
    :math:`O(M^2)` memory cost, where :math:`M` is the dimension of
    the axis over which inner products are taken.


    Parameters
    ----------
    ainv : array_like
      Current value of :math:`\mathbf{A}^{-1}`
    u : array_like
      Vertical factor for rank-one update :math:`\mathbf{u}`
    vt : array_like
      Horizontal factor for rank-one update :math:`\mathbf{v}^H`
    dimN : int, optional (default 2)
      Number of spatial dimensions arranged as leading axes in input array.
      Axis M is taken to be at dimN+2.

    Returns
    -------
    ainv : array_like
      Current value of :math:`\mathbf{A}^{-1}`
    """

    ainv = mdbi_sm_r1update(ainv, numpy.swapaxes(dtu,dimN + 2,dimN + 3), vt, dimN)
    ainv = mdbi_sm_r1update(ainv, numpy.swapaxes(numpy.conj(vt),dimN + 2,dimN + 3), numpy.conj(dtu), dimN)
    ainv = mdbi_sm_r1update(ainv, numpy.swapaxes(numpy.conj(vt),dimN + 3, dimN + 3), utu*vt, dimN)
    

    return ainv

def mdbi_sm_r1update(ainv, u, vt, dimN=2):
    r"""
    Compute the updated a multiple diagonal block inverse after
    a rank-one update for each block using the Sherman-Morrison
    equation. The computation yields an :math:`O(M^2)` time cost and
    :math:`O(M^2)` memory cost, where :math:`M` is the dimension of
    the axis over which inner products are taken.


    Parameters
    ----------
    ainv : array_like
      Current value of :math:`\mathbf{A}^{-1}`
    u : array_like
      Vertical factor for rank-one update :math:`\mathbf{u}`
    vt : array_like
      Horizontal factor for rank-one update :math:`\mathbf{v}^H`
    dimN : int, optional (default 2)
      Number of spatial dimensions arranged as leading axes in input array.
      Axis M is taken to be at dimN+2.

    Returns
    -------
    ainv : array_like
      Current value of :math:`\mathbf{A}^{-1}`
    """

    # Counterintuitively,
    # u shape (N1, N2, 1, 1, 1, M)
    # vt shape (N1, N2, 1, 1, M, 1)
    # The inputs are counter-intuitive, but I have verified that the function works as expected.

    aiu = sporco.linalg.inner(ainv, u, axis=dimN + 3)
    vtAiu = 1.0 + sporco.linalg.inner(vt, aiu, axis=dimN + 2)
    vtAi = sporco.linalg.inner(vt, ainv, axis=dimN + 2)
    aiuvtai = aiu * vtAi
    ainv = ainv - aiuvtai / vtAiu

    return ainv

def mdbi_sm_update(ainv, u, vt, axisK, adimN=2):
    r"""
    Compute the updated a multiple diagonal block inverse after
    a rank-one update for each block using the Sherman-Morrison
    equation. The computation yields an :math:`O(M^2K)` time cost and
    :math:`O(M^2)` memory cost, where :math:`M` is the dimension of
    the axis over which inner products are taken.


    Parameters
    ----------
    ainv : array_like
      Current value of :math:`\mathbf{A}^{-1}`
    u : array_like
      Vertical factor for rank-one update :math:`\mathbf{u}`
    vt : array_like
      Horizontal factor for rank-one update :math:`\mathbf{v}^H`
    dimN : int, optional (default 2)
      Number of spatial dimensions arranged as leading axes in input array.
      Axis M is taken to be at dimN+2.

    Returns
    -------
    ainv : array_like
      Current value of :math:`\mathbf{A}^{-1}`
    """
    slcnc = (slice(None),) * axisK
    K = u.shape[axisK]
    for k in range(0, K):
        slck = slcnc + (slice(k, k + 1),) + (slice(None), numpy.newaxis,)
        aiu = sporco.linalg.inner(ainv, u[slck], axis=dimN + 3)
        vtAiu = 1.0 + sporco.linalg.inner(vt[slck], aiu, axis=dimN + 2)
        vtAi = sporco.linalg.inner(vt[slck], ainv, axis=dimN + 2)
        aiuvtai = aiu * vtAi
        ainv = ainv - aiuvtAi / vtAiu

    return ainv

def computeNorms(v,dimN=2):
    axisN = tuple(range(0,dimN)) + (dimN,)
    C = v.shape[dimN]
    vn2 = numpy.sum(numpy.conj(v) * v,axisN,keepdims=True)/C
    return numpy.sqrt(vn2)

