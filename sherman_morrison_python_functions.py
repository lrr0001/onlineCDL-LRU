import sporco.linalg
import sporco.admm.admm
import sporco.admm.cbpdn
import sporco.cnvrep
import sporco.prox
import numpy
import scipy.linalg

class factoredMatrix:
    def update(self,x):
        raise NotImplementedError
    def inv(self,b,D=None):
        raise NotImplementedError

class factoredMatrix_qr(factoredMatrix):
    def __init__(self,D=None,dtype=None):
        if D is None:
            pass
        else:
            m = D.shape[-2]
            n = D.shape[-1]
            if dtype is None:
                dtype= D.dtype
            self.dtype=dtype
            if m <= n:
                idMat = numpy.identity(m,dtype=self.dtype)
                self.Q = numpy.zeros(D.shape[0:-2] + (m,m),dtype=self.dtype)
                self.R = numpy.zeros(D.shape[0:-2] + (m,m),dtype=self.dtype)
                for inds in loop_magic(D.shape[0:-2]):
                    self.Q[inds],self.R[inds] = scipy.linalg.qr(idMat + numpy.matmul(D[inds],conj_tp(D)[inds]))
                self.flipped=True
            else:
                idMat = numpy.identity(n,dtype= self.dtype)
                self.Q = numpy.zeros(D.shape[0:-2] + (n,n),dtype=self.dtype)
                self.R = numpy.zeros(D.shape[0:-2] + (n,n),dtype=self.dtype)
                for inds in loop_magic(self.Q.shape[0:-2]):
                    self.Q[inds],self.R[inds] = scipy.linalg.qr(idMat + numpy.matmul(conj_tp(D)[inds],D[inds]))
                self.flipped=False

    def update(self,x):
        for inds in loop_magic(self.Q.shape[0:-2]):
            self.Q[inds],self.R[inds] = scipy.linalg.qr_update(self.Q[inds],self.R[inds],x[inds],conj_tp(x[inds]))


    def inv_vec(self,b,D=None):
        if self.flipped:
            assert D is not None
            y = numpy.matmul(conj_tp(self.Q),numpy.matmul(D,b.reshape(b.shape + (1,))))
            y = y.reshape(y.shape[0:-1])
            z = numpy.zeros(y.shape,dtype=self.dtype)
            for inds in loop_magic(self.Q.shape[0:-2]):
                z[inds] = scipy.linalg.solve_triangular(self.R[inds],y[inds])
            return b - (numpy.matmul(conj_tp(D),z.reshape(z.shape + (1,)))).reshape(b.shape)
        else:
            y = numpy.matmul(conj_tp(self.Q),b.reshape(b.shape + (1,)))
            y = y.reshape(y.shape[0:-1])
            z = numpy.zeros(y.shape,dtype=self.dtype)
            for inds in loop_magic(self.Q.shape[0:-2]):
                z[inds] = scipy.linalg.solve_triangular(self.R[inds],y[inds])
            return z
    def inv_mat(self,b,D=None):
        if self.flipped:
            assert D is not None
            y = numpy.matmul(conj_tp(self.Q),numpy.matmul(D,b))
            z = numpy.zeros(y.shape,dtype=self.dtype)
            for inds in loop_magic(self.Q.shape[0:-2]):
                z[inds] = scipy.linalg.solve_triangular(self.R[inds],y[inds])
            return b - numpy.matmul(conj_tp(D),z)
        else:
            y = numpy.matmul(conj_tp(self.Q),b)
            z = numpy.zeros(y.shape,dtype=self.dtype)
            for inds in loop_magic(self.Q.shape[0:-2]):
                z[inds] =scipy.linalg.solve_triangular(self.R[inds],y[inds])
            return z

def conj_tp(x):
    return numpy.conj(numpy.swapaxes(x,-2,-1)) 

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


class loop_magic: # missing zero
    def __init__(self,b):
        self.b = b
        self.L = len(b)
        self.ind = [0]*self.L
    def __iter__(self):
        return self
    def __next__(self):
        overflow = True
        for ii in range(self.L):
            self.ind[ii] += 1
            if self.ind[ii] == self.b[ii]:
                self.ind[ii] = 0
            else:
                overflow = False
                break
        if overflow:
            raise StopIteration
        return tuple(self.ind)

