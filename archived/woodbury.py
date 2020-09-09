import numpy
import sporco.linalg
import numpy.linalg


def woodburyIpUV(df,r,c,noc,nof):
    '" Computes the inverse of I + df^Hdf using the Woodbury inversion lemma "'

    # need identity matrix for small dimensions (r x c x C x C)
    idmat = numpy.zeros((r,c,noc,noc))
    for mm in range(noc):
        idmat[:,:,slice(mm,mm+1),slice(mm,mm+1)] = 1

    # need identity matrix for large dimensions (r x c x 1 x 1 x M x M)
    idmat2 = numpy.zeros((r,c,1,1,nof,nof))
    for mm in range(nof):
        idmat2[:,:,:,:,slice(mm,mm+1),slice(mm,mm+1)] = 1

    # compute (I + dfdf^H)^{-1}
    dt = numpy.conj(df.reshape((r,c,1,noc,nof)))
    d = df.reshape((r,c,noc,1,nof))
    ddt = sporco.linalg.inner(d,dt,axis=4)
    b = idmat + ddt.reshape((r,c,noc,noc))
    binv = numpy.linalg.inv(b)

    # expand for computation of a inverse
    binv = binv.reshape((r,c,noc,noc,1,1))

   # ainv = I - df^H binv df
    dt = dt.reshape((r,c,noc,1,nof,1))
    d = d.reshape((r,c,1,noc,1,nof))
    ainv = idmat2 - sporco.linalg.inner(dt,sporco.linalg.inner(binv,d,axis=3),axis=2)
    return ainv

def woodburyIpUV2(df,r,c,noc,nof):
    idmat = numpy.zeros((r,c,noc,noc))
    for mm in range(noc):
        idmat[:,:,slice(mm,mm+1),slice(mm,mm+1)] = 1
    idmat2 = numpy.zeros((r,c,nof,nof))
    for mm in range(nof):
        idmat2[:,:,slice(mm,mm+1),slice(mm,mm+1)] = 1
    dt = numpy.conj(numpy.swapaxes(df,2,3))
    b = idmat + numpy.matmul(df,dt)
    binv = numpy.linalg.inv(b)

    ainv = idmat2 - numpy.matmul(dt,numpy.matmul(binv,df))
    ainv = 1/2*(ainv + numpy.conj(numpy.swapaxes(ainv,2,3)))
    #return ainv.reshape((r,c,1,1,nof,nof))
    return(ainv)



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

