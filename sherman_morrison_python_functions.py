import sporco.linalg
import sporco.admm.admm
import sporco.admm.cbpdn
import sporco.cnvrep
import sporco.prox
import numpy
import numpy.linalg
import scipy.linalg
import sklearn.utils.extmath

class factoredMatrix:
    def update(self,u,v,D):
        raise NotImplementedError
    def inv(self,b,D=None):
        raise NotImplementedError

class factoredMatrix_aIpBhB(factoredMatrix):
    def sym_update(self,x,sign):
        raise NotImplementedError

    def update(self,u,v,D):

        lamb1,lamb2,lamb3,x1,x2,x3 = self.get_update_vectors(u,v,D)
        # eigenvalues can be negative, so the order here matters. Third eigenvalue is guarenteed to be positive.
        self.sym_update(numpy.sqrt(lamb3)*x3,sign=1)
        self.sym_update(numpy.sqrt(lamb1)*x1,sign=numpy.sign(lamb1))
        self.sym_update(numpy.sqrt(lamb2)*x2,sign=numpy.sign(lamb2))

    def get_update_vectors(self,u,v,D): # requires that self.flipped is defined.
        """
        get_update_vectors(self,u,v,D)
          Converts update to symmetric rank-1 updates.
          Dnew = D + addDim(u)*np.swapaxes(np.conj(addDim(v)),-1,-2)
          ***Inputs***
            u shape:  (...,M)
            v shape:  (...,N)
            D shape:  (...,M,N)

          ***Outputs***
          (eigval1,eigval2,eigval3,eigvec1,eigvec2,eigvec3)
            eigval3 is positive semidefinite
            eigval1, eigval2, eigval3 shape: (...,1)
            eigvec1, eigvec2, eigvec3 shape: either (...,M) if flipped or (..., N) if not flipped
        """
        baseShape = D.shape[0:-2]
        flatShape = (1,)*len(baseShape)
        uhu = vec_sqrnorm(u)
        vhv = vec_sqrnorm(v)
        e1 = numpy.concatenate((numpy.ones(flatShape + (1,)),numpy.zeros(flatShape + (1,))),-1)
        e1 = addDim(e1)
        e2 = numpy.concatenate((numpy.zeros(flatShape + (1,)),numpy.ones(flatShape + (1,))),-1)
        e2 = addDim(e2)

        if self.flipped is True:
            # I think there is a numpy function that does this already
            u_broadcast = numpy.empty(baseShape + (u.shape[-1],) + (1,),dtype=u.dtype) # allows broadcasting for u
            u_broadcast[:] = addDim(u)
            dv = numpy.matmul(D,addDim(v))
            B = numpy.concatenate((dv,u_broadcast),axis=-1)
            A = conj_tp(numpy.concatenate((u_broadcast,dv),axis=-1))
            eigval3 = vhv # technically not an eigenvalue unless u is unit length.
            x3 = u
        else:
            v_broadcast = numpy.empty(baseShape + (v.shape[-1],) + (1,),dtype=v.dtype) # allows broadcasting for v
            v_broadcast[:] = addDim(v)
            dhu = numpy.matmul(conj_tp(D),addDim(u))
            B = numpy.concatenate((dhu,v_broadcast),axis=-1)
            A = conj_tp(numpy.concatenate((v_broadcast,dhu),axis=-1))
            eigval3 = uhu # technically not an eigenvalue unless v is unit length.
            x3 = v

        # Need eigenvalues of BA, identical to eigenvalues of AB:
        AB = numpy.matmul(A,B)
        a = numpy.matmul(conj_tp(e1),numpy.matmul(AB,e1))
        a = minusDim(a)
        b = numpy.matmul(conj_tp(e1),numpy.matmul(AB,e2))
        b = minusDim(b)
        c = numpy.matmul(conj_tp(e2),numpy.matmul(AB,e1))
        c = minusDim(c)
        d = numpy.matmul(conj_tp(e2),numpy.matmul(AB,e2))
        d = minusDim(d)
        eigval1, eigval2,seigvec1,seigvec2 = eig2by2(a,b,c,d)


        # If the eigenvalues are equal, any basis will serve for eigenvectors. Euclidean is good default choice.
        seigvec1[numpy.concatenate((numpy.abs(eigval1 - eigval2) < 1e-10,numpy.zeros(eigval1.shape,dtype=bool)),-1)] = 1
        seigvec2[numpy.concatenate((numpy.abs(eigval1 - eigval2) < 1e-10,numpy.zeros(eigval1.shape,dtype=bool)),-1)] = 0
        seigvec1[numpy.concatenate((numpy.zeros(eigval1.shape,dtype=bool),numpy.abs(eigval1 - eigval2) < 1e-10),-1)] = 0
        seigvec2[numpy.concatenate((numpy.zeros(eigval1.shape,dtype=bool),numpy.abs(eigval1 - eigval2) < 1e-10),-1)] = 1



        # Convert eigenvectors of AB to eigenvectors of BA:
        eigvec1 = numpy.matmul(B,addDim(seigvec1))
        eigvec1 = minusDim(eigvec1)
        mag1 = numpy.sqrt(vec_sqrnorm(eigvec1))
        mag1[numpy.abs(mag1)< 1e-10] = 1e10        
        eigvec2 = numpy.matmul(B,addDim(seigvec2))
        eigvec2 = minusDim(eigvec2)
        mag2 = numpy.sqrt(vec_sqrnorm(eigvec2))
        mag2[numpy.abs(mag2) < 1e-10] = 1e10
        x1 = eigvec1/mag1
        x2 = eigvec2/mag2
        return(eigval1,eigval2,eigval3,x1,x2,x3)

class factoredMatrix_qr(factoredMatrix_aIpBhB):
    def __init__(self,D=None,dtype=None,rho=1):
        if D is None:
            raise NotImplementedError('QR factored matrix code currently requires input dictionary.')
        else:
            m = D.shape[-2]
            n = D.shape[-1]
            if dtype is None:
                dtype= D.dtype
            self.dtype=dtype
            self.rho = rho
            if m <= n:
                idMat = numpy.identity(m,dtype=self.dtype)
                self.Q = numpy.zeros(D.shape[0:-2] + (m,m),dtype=self.dtype)
                self.R = numpy.zeros(D.shape[0:-2] + (m,m),dtype=self.dtype)
                for inds in loop_magic(D.shape[0:-2]):
                    self.Q[inds],self.R[inds] = scipy.linalg.qr(rho*idMat + numpy.matmul(D[inds],conj_tp(D)[inds]))
                self.flipped=True
            else:
                idMat = numpy.identity(n,dtype= self.dtype)
                self.Q = numpy.zeros(D.shape[0:-2] + (n,n),dtype=self.dtype)
                self.R = numpy.zeros(D.shape[0:-2] + (n,n),dtype=self.dtype)
                for inds in loop_magic(self.Q.shape[0:-2]):
                    self.Q[inds],self.R[inds] = scipy.linalg.qr(rho*idMat + numpy.matmul(conj_tp(D)[inds],D[inds]))
                self.flipped=False

    def sym_update(self,x,sign=1):
        for inds in loop_magic(self.Q.shape[0:-2]):
            self.Q[inds],self.R[inds] = scipy.linalg.qr_update(self.Q[inds],self.R[inds],sign*x[inds],conj_tp(x[inds]))


    def inv_vec(self,b,D=None):
        if self.flipped:
            assert D is not None
            y = numpy.matmul(conj_tp(self.Q),numpy.matmul(D,b.reshape(b.shape + (1,))))
            y = y.reshape(y.shape[0:-1])
            z = numpy.zeros(y.shape,dtype=self.dtype)
            for inds in loop_magic(self.Q.shape[0:-2]):
                z[inds] = scipy.linalg.solve_triangular(self.R[inds],y[inds])
            return (b - (numpy.matmul(conj_tp(D),z.reshape(z.shape + (1,)))).reshape(b.shape))/self.rho
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
            return (b - numpy.matmul(conj_tp(D),z))/self.rho
        else:
            y = numpy.matmul(conj_tp(self.Q),b)
            z = numpy.zeros(y.shape,dtype=self.dtype)
            for inds in loop_magic(self.Q.shape[0:-2]):
                z[inds] =scipy.linalg.solve_triangular(self.R[inds],y[inds])
            return z

class factoredMatrix_chol(factoredMatrix_aIpBhB):
    def __init__(self,D=None,dtype=None,rho=1):
        if D is None:
            raise NotImplementedError('Cholesky factorization currently requires input dictionary.')
        
        m = D.shape[-2]
        n = D.shape[-1]
        if dtype is None:
            dtype= D.dtype
        self.dtype=dtype
        self.rho = rho

        if m <= n: # (flipped) Woodbury formulation may be more efficient.
            idMat = numpy.identity(m,dtype=self.dtype)
            self.L = numpy.linalg.cholesky(rho*idMat + numpy.matmul(D,conj_tp(D)))
            self.flipped=True
        else:
            idMat = numpy.identity(n,dtype= self.dtype)
            self.L = numpy.linalg.cholesky(rho*idMat + numpy.matmul(conj_tp(D),D))
            self.flipped=False

    def sym_update(self,x,sign=1):
        self.L = cholesky_rank1_update(self.L,x,sign)

    def inv_vec(self, b, D=None):
        if self.flipped:
            assert D is not None
            y = numpy.matmul(D,addDim(b))
            y = minusDim(y)
            z = numpy.zeros(y.shape,dtype=self.dtype)


            w = solve_triangular(self.L,y,lower=True)
            z = solve_triangular(self.L,w,lower=True,trans=True)
            z = minusDim(z)
            #for inds in loop_magic(self.L.shape[0:-2]): # Loop slows things down, but I don't have vectorized triangular solver
            #    w = scipy.linalg.solve_triangular(self.L[inds],y[inds],lower=True,check_finite=False)
            #    z[inds] = scipy.linalg.solve_triangular(self.L[inds],w,lower=True,trans=2,overwrite_b=True,check_finite=False)
            return (b - (numpy.matmul(conj_tp(D),z.reshape(z.shape + (1,)))).reshape(b.shape))/self.rho


        else:
            y = solve_triangular(self.L,b,lower=True)
            z = solve_triangular(self.L,y,lower=True,trans=True)
            #z = numpy.zeros(b.shape,dtype=self.dtype)
            #for inds in loop_magic(self.L.shape[0:-2]): # Loop slows things down, but I don't have vectorized triangular solver
            #    y = scipy.linalg.solve_triangular(self.L[inds],b[inds],lower=True,check_finite=False)
            #    z[inds] = scipy.linalg.solve_triangular(self.L[inds],y,lower=True,trans=2,overwrite_b=True,check_finite=False)
            return minusDim(z)

    def inv_mat(self,b,D=None):
        if self.flipped:
            assert D is not None
            y = numpy.matmul(D,b)
            w = solve_triangular(self.L,y,lower=True)
            z = solve_triangular(self.L,w,lower=True,trans=True)
            #z = numpy.zeros(y.shape,dtype=self.dtype)
            #for inds in loop_magic(self.L.shape[0:-2]):
            #    w = scipy.linalg.solve_triangular(self.L[inds],y[inds],lower=True,check_finite=False)
            #    z[inds] = scipy.linalg.solve_triangular(self.L[inds],w,lower=True,trans=2,overwrite_b=True,check_finite=False)
            return (b - numpy.matmul(conj_tp(D),z))/self.rho
        else:
            y = solve_triangular(self.L,b,lower=True)
            z = solve_triangular(self.L,y,lower=True,trans=True)
            #z = numpy.zeros(b.shape,dtype=self.dtype)
            #for inds in loop_magic(self.L.shape[0:-2]):
            #    y = scipy.linalg.solve_triangular(self.L[inds],b[inds],lower=True,check_finite=False)
            #    z[inds] = scipy.linalg.solve_triangular(self.L[inds],y,lower=True,trans=2,overwrite_b=True,check_finite=False)
            return z

def solve_triangular(A,b,lower=True,trans=False):
    """
    This function solves the equation Ax = b for x (or A^H x = b if trans=True).

    A is a square, triangular matrix.
    b is either a vector or a matrix

    """
    Ashape = A.shape
    assert(Ashape[-1] == Ashape[-2])
    N = A.shape[-1]
    bshape = b.shape
    if len(bshape) != len(Ashape):
        b = addDim(b)
        bshape += (1,)
    assert(len(bshape) == len(Ashape))
    assert(bshape[-2] == N)
    assert(numpy.all( Ashape[0:-2] == bshape[0:-2] or Ashape[0:-2] == numpy.ones(len(bshape[0:-2])) ))
    
    A = numpy.swapaxes(A,0,-2)
    A = numpy.swapaxes(A,1,-1)
    b = numpy.swapaxes(b,1,-1)
    b = numpy.swapaxes(b,0,-2)
    b = addDim(b)
    b = numpy.swapaxes(b,1,-1)
    A = addDim(A)
    c = b.copy()

    # There is a problem here. The second dimension of b needs a corresponding singleton dimension in A.

    if lower^trans:
        for ii in range(0,N,1):
            for jj in range(0,ii,1):
                if trans:
                    c[ii] = c[ii] - c[jj]*numpy.conj(A[jj,ii])
                else:
                    c[ii] = c[ii] - c[jj]*A[ii,jj]
            c[ii] = c[ii]/A[ii,ii]
    else:
        for ii in range(N - 1,-1,-1):
            for jj in range(N - 1,ii, -1):
                if trans:
                    c[ii] = c[ii] - c[jj]*numpy.conj(A[jj,ii])
                else:
                    c[ii] = c[ii] - c[jj]*A[ii,jj]
            c[ii] = c[ii]/A[ii,ii]
    c = numpy.swapaxes(c,1,-1)
    c = minusDim(c)
    c = numpy.swapaxes(c,1,-1)
    c = numpy.swapaxes(c,0,-2)
    return c

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

def lowRankApprox_broken(a, projIter=5, axisu=2,axisv=3,dimN=2):
    # untested, but should work just fine
    numelN = a.shape[0:dimN]
    numelu = a.shape[axisu]
    numelv = a.shape[axisv]
    u = []
    v = []

    resid = external2mid_LRA(a,axisu,axisv,dimN)
    midshape = resid.shape
    #print(midshape)
    x = mid2internal_LRA(resid, numelu, numelN)
    u1,s1,vh1 = sklearn.utils.extmath.randomized_svd(x,n_components=1,n_iter=projIter,random_state=None)
    vh1 = vh1*s1
    tempu = internal2mid_u_LRA(u1, dimN, midshape)
    #print(vh1.shape)
    tempv = internal2mid_v_LRA(vh1, dimN, midshape)
    approx = tempu*tempv
    u.append(mid2external_LRA(tempu,axisu,axisv,dimN))
    v.append(mid2external_LRA(tempv,axisu,axisv,dimN))

    resid = resid - approx
    x = mid2internaltp_LRA(resid, numelv, numelN)
    vh2,s2,u2 = sklearn.utils.extmath.randomized_svd(x,n_components=1,n_iter=projIter,random_state=None)
    u2 = s2*u2
    tempu = internaltp2mid_u_LRA(u2, dimN, midshape)
    tempv = internaltp2mid_v_LRA(vh2, dimN, midshape)
    approx = approx + tempu*tempv
    u.append(mid2external_LRA(tempu,axisu,axisv,dimN))
    v.append(mid2external_LRA(tempv,axisu,axisv,dimN))
    
    return (u,v,mid2external_LRA(approx,axisu,axisv,dimN))

def external2mid_LRA(a, axisu=2, axisv=3, dimN=2):

    x = numpy.swapaxes(a,axisu, dimN)
    if axisv == dimN:
        return numpy.swapaxes(x,axisu,dimN + 1)
    else:
        return numpy.swapaxes(x,axisv,dimN + 1)

def mid2external_LRA(x, axisu=2, axisv=3, dimN=2):
    if axisv == dimN:
        a = numpy.swapaxes(x,axisu,dimN + 1)
    else:
        a = numpy.swapaxes(x,axisv,dimN + 1)
    return numpy.swapaxes(a,axisu,dimN)

def mid2internal_LRA(x, numelu, numelN):
    return x.reshape((numpy.prod(numelN)*numelu,-1))

def mid2internaltp_LRA(x, numelv, numelN):
    b = numpy.swapaxes(x,len(numelN),len(numelN) + 1)
    return b.reshape((numpy.prod(numelN)*numelv,-1))

def internal2mid_u_LRA(u,dimN,midshape):
    return u.reshape(midshape[0:dimN + 1] + (1,) + (1,)*len(midshape[dimN + 2:]))

def internal2mid_v_LRA(v,dimN,midshape):
    return v.reshape((1,)*dimN + (1,) + midshape[dimN + 1:])

def internaltp2mid_u_LRA(u, dimN, midshape):
    return u.reshape((1,)*dimN + (midshape[dimN],) + (1,) + midshape[dimN + 2:])

def internaltp2mid_v_LRA(v, dimN, midshape):
    return v.reshape(midshape[0:dimN] + (1,) + (midshape[dimN + 1],) + (1,)*len(midshape[dimN + 2:]))




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


class loop_magic:
    def __init__(self,b):
        self.b = b
        self.L = len(b)
        self.ind = [0]*self.L
        self.first = True
    def __iter__(self):
        return self
    def __next__(self):
        if self.first is True:
            self.first = False
            return tuple(self.ind)
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

def eig2by2(a,b,c,d):
    """ Computes the eigenvalues and eigenvectors of 2 x 2 matrices
      [  a   b  ]
      [  c   d  ]
      a, b, c, and d must be broadcastable. They are treated as an array of scalars.
      The last dimension must be singleton.
      output eigval1 and eigval2 are same size as a, b, c, and d.
              """ 

    # speed may be improved by checking whether broadcasting is necessary.
    b_broadcast = numpy.empty(numpy.maximum(numpy.maximum(a.shape,b.shape),numpy.maximum(c.shape,d.shape)),dtype = b.dtype)
    b_broadcast[:] = b
    apdo2 = (a + d)/2
    amdo2 = (a - d)/2
    radicandsqrt = numpy.sqrt(amdo2**2 + b*c)
    eigval1 = apdo2 + radicandsqrt
    eigval2 = apdo2 - radicandsqrt
    eigvec1 = numpy.conj(numpy.concatenate((-b_broadcast,-amdo2-radicandsqrt),axis=-1))
    eigvec2 = numpy.conj(numpy.concatenate((-b_broadcast,-amdo2+radicandsqrt),axis=-1))
    return (eigval1,eigval2,eigvec1,eigvec2)

def eigdecomp_test(eigval1,eigval2,eigvec1,eigvec2):
    V = numpy.concatenate((eigvec1.reshape(eigvec1.shape + (1,)),eigvec2.reshape(eigvec2.shape + (1,))),axis=-1)
    mag = numpy.sqrt(numpy.sum(numpy.conj(V)*V,axis=-2,keepdims=True))
    mag[numpy.abs(mag) < 1e-10] = 1e10
    V = V/mag
    Vinv = numpy.linalg.inv(V) # Note the small matrix is not hermitian, so this is necessary.
    eigval1zp = numpy.concatenate((eigval1,numpy.zeros(eigval1.shape)),axis=-1)
    eigval2zp = numpy.concatenate((numpy.zeros(eigval2.shape),eigval2),axis=-1)
    Diag = numpy.concatenate((eigval1zp.reshape(eigval1zp.shape + (1,)),eigval2zp.reshape(eigval2zp.shape + (1,))),-1)
    return numpy.matmul(V,numpy.matmul(Diag,Vinv))

def sym_eigdecomp_test(eigval1,eigval2,eigvec1,eigvec2):
    V = numpy.concatenate((eigvec1.reshape(eigvec1.shape + (1,)),eigvec2.reshape(eigvec2.shape + (1,))),axis=-1)
    mag = numpy.sum(numpy.conj(V)*V,axis=-2,keepdims=True)
    mag[numpy.abs(mag) < 1e-10] = 1e10
    V = V/mag
    Vinv = conj_tp(V)
    eigval1zp = numpy.concatenate((eigval1,numpy.zeros(eigval1.shape)),axis=-1)
    eigval2zp = numpy.concatenate((numpy.zeros(eigval2.shape),eigval2),axis=-1)
    Diag = numpy.concatenate((eigval1zp.reshape(eigval1zp.shape + (1,)),eigval2zp.reshape(eigval2zp.shape + (1,))),-1)
    return numpy.matmul(V,numpy.matmul(Diag,Vinv))


def cholesky_rank1_update_inplace(L,x):
    r"""
    This code implements the rank-1 Cholesky update from Wikipedia page, modified to handle complex numbers.
    """  
    L = numpy.swapaxes(L,0,-2)
    L = numpy.swapaxes(L,1,-1)
    x = numpy.swapaxes(x,0,-1)
    x = numpy.swapaxes(x,1,-2)
    x = numpy.swapaxes(x,-1,-2)
    for ii in range(L.shape[0]):
        r = numpy.sqrt(L[ii,ii]**2 + x[ii]*numpy.conj(x[ii]))
        c = r/L[ii,ii]
        s = x[ii]/L[ii,ii]
        L[ii,ii] = r
        for jj in range(ii + 1,L.shape[0]):
            L[jj,ii] = (L[jj,ii] + numpy.conj(s)*x[jj])/c
            x[jj] = c*x[jj] - s*L[jj,ii]
    L = numpy.swapaxes(L,0,-2)
    L = numpy.swapaxes(L,1,-1)
    return L

def cholesky_rank1_update(Li,xi,sign=1):
    """Computes the cholesky rank1 update. (Downdate may be unstable.)"""
    if numpy.ndim(sign) >= 2:
        sign = numpy.swapaxes(sign,0,-1)
        sign = numpy.swapaxes(sign,1,-2)
        sign = numpy.swapaxes(sign,-1,-2)
    Li = numpy.swapaxes(Li,0,-2)
    Li = numpy.swapaxes(Li,1,-1)
    xi = numpy.swapaxes(xi,0,-1)
    xi = numpy.swapaxes(xi,1,-2)
    xi = numpy.swapaxes(xi,-1,-2)
    L = numpy.empty(Li.shape[0:2] + tuple(numpy.maximum(Li.shape[2:],xi.shape[1:])),dtype=numpy.cdouble)
    L[:] = Li
    x = numpy.empty((xi.shape[0],) + tuple(numpy.maximum(Li.shape[2:],xi.shape[1:])),dtype=numpy.cdouble)
    x[:] = xi

    for ii in range(L.shape[0]):
        r = numpy.sqrt(L[ii,ii]**2 + sign*x[ii]*numpy.conj(x[ii]))
        c = r/L[ii,ii]
        s = x[ii]/L[ii,ii]
        L[ii,ii] = r
        for jj in range(ii + 1,L.shape[0]):
            L[jj,ii] = (L[jj,ii] + sign*numpy.conj(s)*x[jj])/c
            x[jj] = c*x[jj] - s*L[jj,ii]
    L = numpy.swapaxes(L,0,-2)
    L = numpy.swapaxes(L,1,-1)
    return L

def conj_sym_proj(x,axes):
    y = numpy.array(x,copy=True)
    for ii in axes:
        y = numpy.roll(numpy.flip(y,ii),1,ii)
    return (x + numpy.conj(y))/2

def vec_sqrnorm(x):
    return numpy.sum(numpy.conj(x)*x,axis=-1,keepdims=True)

def DfMatRep(a,axisu,axisv):
    """
    MatRep moves the matrices to the end of the representation.
    This is necessary for factorization updates, since 
    factorization code is currently not compatible with axisu != -2
    and axisv != -1.
    """
    a = addDim(a)
    a = numpy.swapaxes(a,axisv,-1)
    a = numpy.swapaxes(a,axisu,axisv)
    a = numpy.swapaxes(a,axisv,-2)
    return a

def uMatRep(u,axisu,axisv):
    """
    MatRep moves the matrices to the end of the representation.
    This is necessary for factorization updates, since 
    factorization code is currently not compatible with axisu != -2
    and axisv != -1.
    """
    u = numpy.swapaxes(u,axisu,axisv)
    u = numpy.swapaxes(u,axisv,-1)
    return u

def vMatRep(v,axisu,axisv):
    """
    MatRep moves the matrices to the end of the representation.
    This is necessary for factorization updates, since 
    factorization code is currently not compatible with axisu != -2
    and axisv != -1.
    """
    v = numpy.conj(numpy.swapaxes(v,axisv,-1))
    return v

def addDim(x):
    return x.reshape(x.shape + (1,))

def minusDim(x):
    if x.shape[-1] > 1:
        raise ValueError('Last diminsion is not singleton. Cannot remove.')
    return x.reshape(x.shape[0:-1])
