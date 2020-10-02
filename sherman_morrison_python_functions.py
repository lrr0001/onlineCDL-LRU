import sporco.linalg
import numpy
import numpy.linalg
import scipy.linalg
import sklearn_modified_complex_svd
import math



class factoredMatrix:
    """ Currently, I don't see what kind of functions would be necessary for all matrix decompositions, so I'll leave this empty for now.
    """
    pass

class factoredMatrix_aIpBhB(factoredMatrix):
    """
    This is a class of decompositions for matrices of the form aI + B^H B. Rank-1 B updates can be converted to 3 symmetric rank-1 updates for the decomposition.

    """
    def sym_update(self,x,sign):
        raise NotImplementedError

    def inv_mat(self,b,D):
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

    def inv_check_ls(self,D):
        """ Checks the left-side inverse property of object's mat_inv function.

        """
        M = D.shape[-2]
        N = D.shape[-1]
        matId = numpy.zeros(D.shape[0:-2] + (N,N,))
        for ii in range(N):
            s = [slice(0,D.shape[jj]) for jj in range(len(D.shape[0:-2]))] + [slice(ii,ii + 1),slice(ii,ii + 1)]
            matId[tuple(s)] = 1
        aIpDhD = self.rho*matId + numpy.matmul(conj_tp(D),D)
        return numpy.amax(numpy.abs(matId - self.inv_mat(aIpDhD,D)))

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
            idMat = numpy.reshape(idMat,(1,)*len(D.shape[:-2]) + (m,m,))
            self.L = numpy.linalg.cholesky(rho*idMat + numpy.matmul(D,conj_tp(D)))
            self.flipped=True
        else:
            idMat = numpy.identity(n,dtype= self.dtype)
            idMat = numpy.reshape(idMat,(1,)*len(D.shape[:-2]) + (n,n,))
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

class factoredMatrix_cloneSlices:
    def __init__(self, D=None,dtype=None,rho = 1.,dimN=None,clonedSlices=slice(0,1),clonedRhos = 2.):
        assert D is not None
        m = D.shape[-2]
        n = D.shape[-1]
        if dtype is None:
            dtype = D.dtype
        if dimN is None:
            dimN = len(numpy.squeeze(D[:-2]))
        if not isinstance(clonedSlices,list):
            clonedSlices = [clonedSlices,]*dimN
        if not isinstance(clonedRhos,list):
            clonedRhos = [clonedRhos,]*dimN

        self.dtype = dtype
        self.rho = rho
        self.dimN = dimN
        self.baseShape = D.shape[:-2]

        self.coreSlices = []
        self.numOfSlices = []
        for ii in range(dimN):
            # The problem is an incompatibility between the slice and range functions. To fix, need to add N to stop.
            self.coreSlices.append(iter2slices(range(*clonedSlices[ii].indices(D.shape[ii])),D.shape[ii]))
            self.numOfSlices.append(len(self.coreSlices[ii]))

        self.coreQ = []
        for inds in loop_magic(self.numOfSlices):
            #import pdb; pdb.set_trace()
            self.coreQ.append(factoredMatrix_chol(D[accessLoL(self.coreSlices,inds)],self.dtype,self.rho))


        self.edgeSlice = []
        self.edgeQ = []
        self.altEdgeQ = []
        altRho = []

        for ii in range(dimN):
            self.edgeSlice.append([slice(None),]*ii + [clonedSlices[ii],] + [slice(None),]*(dimN - ii - 1))
            altRho.append(clonedRhos[ii]*numpy.ones(D.shape[0:ii] + (1,) + D.shape[ii + 1:self.dimN] + (1,)*(len(D.shape) - dimN)))
        
        for ii in range(dimN):
            for jj in range(dimN):
                if ii != jj:
                    altRho[ii][tuple(self.edgeSlice[jj])] += clonedRhos[jj]

        #import pdb; pdb.set_trace()
        for ii in range(dimN):
            self.edgeQ.append(factoredMatrix_chol(D[tuple(self.edgeSlice[ii])],self.dtype,self.rho))
            self.altEdgeQ.append(factoredMatrix_chol(D[tuple(self.edgeSlice[ii])],self.dtype,altRho[ii]))


#    def __init__(self, D=None,dtype=None,rho = 1.,dimN=None,clonedSlices = slice(-1,0),cloneRhos = 2.):
#        if D is None:
#            raise NotImplementedError('Duplicate extension of Cholesky factorization requires input matrix D.')
#        m = D.shape[-2]
#        n = D.shape[-1]
#        if dtype is None:
#            dtype = D.dtype
#        if dimN is None:
#            dimN = len(D.shape) - 2
#        if not isinstance(clonedSlices,list):
#            clonedSlices = [clonedSlices,]*dimN
#        if not isinstance(cloneRhos,list):
#            cloneRhos = [cloneRhos,]*dimN

        # still need to work on duplicate rhos
#        self.dtype = dtype
#        self.rho = rho
#        self.dimN = dimN
#        self.baseShape = D.shape[:-2]

#        self.coreSlice = clonedSlices + [slice(None),]*(len(D.shape) - dimN)
#        self.coreQ = FactoredMatrix_chol(D[self.coreSlice],self.dtype,self.rho)
#        self.edgeSlice = []
#        self.edgeQ = []
#        self.altEdgeQ = []

#        for ii in range(dimN):
#            self.edgeSlice.append([splice(None),]*ii + [clonedSlices[ii],] + [splice(None),]*(dimN - ii - 1))
#            self.edgeQ.append(FactoredMatrix_chol(self.D[self.edgeSlice[ii]],self.dtype,self.rho)
#            altRho = self.cloneRhos[ii]*numpy.ones((1,)*ii + (D.shape[ii],) + (1,)*(dimN - ii - 1))
#            altRho[self.edgeSlice[ii]] = numpy.sum(duplicRhos)
#            self.altEdgeQ.append(FactoredMatrix_chol(D[self.edgeSlice[ii],self.dtype,altRho))

    def inv_check_ls(self,D):
        err = -math.inf
        ii = 0
        for inds in loop_magic(self.numOfSlices):
            err = max(err,self.coreQ[ii].inv_check_ls(D[accessLoL(self.coreSlices,inds)]))
            ii += 1
        for ii in range(self.dimN):
            err = max(err,self.edgeQ[ii].inv_check_ls(D[tuple(self.edgeSlice[ii])]))
            err = max(err,self.altEdgeQ[ii].inv_check_ls(D[tuple(self.edgeSlice[ii])]))
        return err

    def _inv_mat_(self,b,D,isalt):
        # alocate, select, solve
        
        xBaseShape = max(b.shape[:-2],self.baseShape)
        x = numpy.empty(xBaseShape + b.shape[-2:],self.dtype)
        bCoreSlices = self.coreSlices.copy()
        for ii in range(self.dimN):
            bEdgeSlice = self.edgeSlice[ii].copy()
            if b.shape[ii] == 1:
                bEdgeSlice[ii] = slice(None)
                bCoreSlices[ii] = [slice(None),]*self.numOfSlices[ii]
            if D is None:
                if isalt:
                    x[tuple(self.edgeSlice[ii])] = self.altEdgeQ[ii].inv_mat(b=b[tuple(bEdgeSlice)],D=D)
                else:
                    x[tuple(self.edgeSlice[ii])] = self.edgeQ[ii].inv_mat(b=b[tuple(bEdgeSlice)],D=D)
            else:
                if isalt:
                    x[tuple(self.edgeSlice[ii])] = self.altEdgeQ[ii].inv_mat(b=b[tuple(bEdgeSlice)],D=D[tuple(self.edgeSlice[ii])])
                else:
                    x[tuple(self.edgeSlice[ii])] = self.edgeQ[ii].inv_mat(b=b[tuple(bEdgeSlice)],D=D[tuple(self.edgeSlice[ii])])
        ii = 0
        for inds in loop_magic(self.numOfSlices):
            if D is None:
                x[accessLoL(self.coreSlices,inds)] = self.coreQ[ii].inv_mat(b=b[accessLoL(bCoreSlices,inds)],D=D)
            else:
                x[accessLoL(self.coreSlices,inds)] = self.coreQ[ii].inv_mat(b=b[accessLoL(bCoreSlices,inds)],D=D[accessLoL(self.coreSlices,inds)])
            ii += 1
        return x
    def inv_mat(self,b,D):
        return _inv_mat_(b,D,False)

    def inv_vec(self,b,D):
        return minusDim(self._inv_mat_(addDim(b),D,False))

    def inv_mat_duplc(self,b,D):
        return _inv_mat_(b,D,True)
    def inv_vec_duplc(self,b,D):
        return minusDim(self._inv_mat_(addDim(b),D,True))

    def update(self,u,v,D):
        # This code cannot handle broadcasting for singleton dimensions
        uCoreSlices = self.coreSlices.copy()
        vCoreSlices = self.coreSlices.copy()
        
        for ii in range(self.dimN):
            uEdgeSlice = self.edgeSlice[ii].copy()
            vEdgeSlice = self.edgeSlice[ii].copy()
            if u.shape[ii] == 1:
                uCoreSlices[ii] = [slice(None),]*self.numOfSlices[ii]
                uEdgeSlice[ii] = slice(None)
            if v.shape[ii] == 1:
                vCoreSlices[ii] = [slice(None),]*self.numOfSlices[ii]
                vEdgeSlice[ii] = slice(None)
            self.edgeQ[ii].update(u[tuple(uEdgeSlice)],v[tuple(vEdgeSlice)],D[tuple(self.edgeSlice[ii])])
            self.altEdgeQ[ii].update(u[tuple(uEdgeSlice)],v[tuple(vEdgeSlice)],D[tuple(self.edgeSlice[ii])])
        ii = 0
        for inds in loop_magic(self.numOfSlices):
            self.coreQ[ii].update(u[accessLoL(uCoreSlices,inds)],v[accessLoL(vCoreSlices,inds)],D[accessLoL(self.coreSlices,inds)])
            ii += 1


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


def lowRankApprox_vold(a, projIter=2,axisu=2,axisv=3,dimN=2):
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





def lowRankApprox(a, projIter=5, axisu=2,axisv=3,dimN=2):
    # This function forces the low-rank approximation to have conjugate symmetry (real after frequency transformation).
    numelN = a.shape[0:dimN]
    numelu = a.shape[axisu]
    numelv = a.shape[axisv]
    u = []
    v = []

    resid = external2mid_LRA(a,axisu,axisv,dimN)
    midshape = resid.shape

    x = mid2internaltp_LRA(resid, numelv, numelN)
    #import pdb; pdb.set_trace()
    vh2,s2,u2 = sklearn_modified_complex_svd.randomized_svd(x,n_components=1,n_iter=projIter)
    #import pdb; pdb.set_trace()
    u2 = s2*u2
    # These projections are necessary, though I don't understand why. If the input has conjugate symmetry across the N-axes, shouldn't the output as well?
    tempu = conj_sym_proj(internaltp2mid_u_LRA(u2, dimN, midshape),range(dimN))
    tempv = conj_sym_proj(internaltp2mid_v_LRA(vh2, dimN, midshape),range(dimN))
    u.append(mid2external_LRA(tempu,axisu,axisv,dimN))
    v.append(mid2external_LRA(tempv,axisu,axisv,dimN))
    approx = tempu*tempv
    resid = resid - approx

    #print(midshape)
    x = mid2internal_LRA(resid, numelu, numelN)
    u1,s1,vh1 = sklearn_modified_complex_svd.randomized_svd(x,n_components=1,n_iter=projIter)
    vh1 = s1*vh1
    #tempu = #conj_sym_proj(
    tempu = internal2mid_u_LRA(u1, dimN, midshape)#,range(dimN))
    #print(vh1.shape)
    #tempv = #conj_sym_proj(
    tempv = internal2mid_v_LRA(vh1, dimN, midshape)
    #,range(dimN))


    u.append(mid2external_LRA(tempu,axisu,axisv,dimN))
    v.append(mid2external_LRA(tempv,axisu,axisv,dimN))
    #print('First rank-one component, fractional error:')
    #print(numpy.sqrt(numpy.sum(numpy.conj(resid - approx)*(resid - approx)))/numpy.sqrt(numpy.sum(numpy.conj(resid)*resid)))


    #print('Second rank-one component, subsequent fractional error:')
    #print(numpy.sqrt(numpy.sum(numpy.conj(resid -tempu*tempv)*(resid -tempu*tempv)))/numpy.sqrt(numpy.sum(numpy.conj(resid)*resid)))
    approx = approx + tempu*tempv

    
    return (u,v,mid2external_LRA(approx,axisu,axisv,dimN))
def lowRankApprox_stackfilters(a, projIter=5,n_components=1,axisu=2,axisv=3,dimN=2):
    # This function forces the low-rank approximation to have conjugate symmetry (real after frequency transformation).
    numelN = a.shape[0:dimN]
    numelu = a.shape[axisu]
    numelv = a.shape[axisv]
    #u = []
    #v = []

    resid = external2mid_LRA(a,axisu,axisv,dimN)
    midshape = resid.shape

    x = mid2internaltp_LRA(resid, numelv, numelN)
    #import pdb; pdb.set_trace()
    vh2,s2,u2 = sklearn_modified_complex_svd.randomized_svd(x,n_components=n_components,n_iter=projIter)
    #import pdb; pdb.set_trace()
    u2 = s2*u2
    # These projections are necessary, though I don't understand why. If the input has conjugate symmetry across the N-axes, shouldn't the output as well?
    #tempu = conj_sym_proj(internaltp2mid_u_LRA(u2, dimN, midshape),range(dimN))
    #tempv = conj_sym_proj(internaltp2mid_v_LRA(vh2, dimN, midshape),range(dimN))
    u=numpy.split(mid2external_LRA(tempu,axisu,axisv,dimN),n_components,axis=axisu)
    v=numpy.split(mid2external_LRA(tempv,axisu,axisv,dimN),n_components,axis=axisv)
    approx = tempu*tempv

    #print(midshape)
    #x = mid2internal_LRA(resid, numelu, numelN)
    #u1,s1,vh1 = sklearn_modified_complex_svd.randomized_svd(x,n_components=1,n_iter=projIter)
    #vh1 = s1*vh1
    #tempu = #conj_sym_proj(
    #tempu = internal2mid_u_LRA(u1, dimN, midshape)#,range(dimN))
    #print(vh1.shape)
    #tempv = #conj_sym_proj(
    #tempv = internal2mid_v_LRA(vh1, dimN, midshape)
    #,range(dimN))


    #u.append(mid2external_LRA(tempu,axisu,axisv,dimN))
    #v.append(mid2external_LRA(tempv,axisu,axisv,dimN))
    #print('First rank-one component, fractional error:')
    #print(numpy.sqrt(numpy.sum(numpy.conj(resid - approx)*(resid - approx)))/numpy.sqrt(numpy.sum(numpy.conj(resid)*resid)))


    #print('Second rank-one component, subsequent fractional error:')
    #print(numpy.sqrt(numpy.sum(numpy.conj(resid -tempu*tempv)*(resid -tempu*tempv)))/numpy.sqrt(numpy.sum(numpy.conj(resid)*resid)))
    #approx = approx + tempu*tempv

    
    return (u,v,mid2external_LRA(approx,axisu,axisv,dimN))

def lowRankApprox_stackchannels(a, projIter=5,n_components=1,axisu=2,axisv=3,dimN=2):
    # This function forces the low-rank approximation to have conjugate symmetry (real after frequency transformation).
    numelN = a.shape[0:dimN]
    numelu = a.shape[axisu]
    numelv = a.shape[axisv]
    #u = []
    #v = []

    resid = external2mid_LRA(a,axisu,axisv,dimN)
    midshape = resid.shape

    #x = mid2internaltp_LRA(resid, numelv, numelN)
    #import pdb; pdb.set_trace()
    #vh2,s2,u2 = sklearn_modified_complex_svd.randomized_svd(x,n_components=n_components,n_iter=projIter)
    #import pdb; pdb.set_trace()
    #u2 = s2*u2
    # These projections are necessary, though I don't understand why. If the input has conjugate symmetry across the N-axes, shouldn't the output as well?
    #tempu = conj_sym_proj(internaltp2mid_u_LRA(u2, dimN, midshape),range(dimN))
    #tempv = conj_sym_proj(internaltp2mid_v_LRA(vh2, dimN, midshape),range(dimN))
    #u=numpy.split(mid2external_LRA(tempu,axisu,axisv,dimN),n_components,axis=axisu)
    #v=numpy.split(mid2external_LRA(tempv,axisu,axisv,dimN),n_components,axis=axisv)
    #approx = tempu*tempv

    #print(midshape)
    x = mid2internal_LRA(resid, numelu, numelN)
    u1,s1,vh1 = sklearn_modified_complex_svd.randomized_svd(x,n_components=1,n_iter=projIter)
    vh1 = s1*vh1
    #tempu = #conj_sym_proj(
    tempu = internal2mid_u_LRA(u1, dimN, midshape)#,range(dimN))
    #print(vh1.shape)
    #tempv = #conj_sym_proj(
    tempv = internal2mid_v_LRA(vh1, dimN, midshape)
    #,range(dimN))


    u = numpy.split(mid2external_LRA(tempu,axisu,axisv,dimN),n_components,axisu)
    v = numpy.split(mid2external_LRA(tempv,axisu,axisv,dimN),n_components,axisv)
    #print('First rank-one component, fractional error:')
    #print(numpy.sqrt(numpy.sum(numpy.conj(resid - approx)*(resid - approx)))/numpy.sqrt(numpy.sum(numpy.conj(resid)*resid)))


    #print('Second rank-one component, subsequent fractional error:')
    #print(numpy.sqrt(numpy.sum(numpy.conj(resid -tempu*tempv)*(resid -tempu*tempv)))/numpy.sqrt(numpy.sum(numpy.conj(resid)*resid)))
    approx = tempu*tempv

    
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







def randomized_svd(x,n_components=1,n_oversamples=10,n_iter=4):
    if x.shape[1] > x.shape[0]:
        x = numpy.swapaxes(x,0,1)
    p = numpy.random.randn(x.shape[1],n_components + n_oversamples)
    z = numpy.matmul(x,p)
    for ii in range(n_iter):
        z = numpy.matmul(x,numpy.matmul(numpy.conj(numpy.swapaxes(x,0,1)),z))
        
    q,_ = scipy.linalg.qr(z)
    u_small,s,vh = scipy.linalg.svd(numpy.matmul(numpy.conj(numpy.swapaxes(q,0,1)),x))
    u = numpy.matmul(q,u_small)
    if x.shape[1] > x.shape[0]:
        return (numpy.swapaxes(vh[slice(0,n_components),:],0,1),s[0:n_components],numpy.swapaxes(u[:,slice(0,n_components)],0,1))
    else:
        return (u[:,slice(0,n_components)],s[0:n_components],vh[slice(0,n_components),:])

def computeNorms(v,dimN=2):
    axisN = tuple(range(0,dimN)) + (dimN,)
    C = v.shape[dimN]
    vn2 = numpy.sum(numpy.conj(v) * v,axisN,keepdims=True)/C
    return numpy.sqrt(vn2)


class loop_magic:
    def __init__(self,b):
        self.b = b
        self.L = len(b)
        self.ind = [0,]*self.L
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

def highFreqSlice(N):
    if N % 2 == 0:
        return slice(int(N/2),int(N/2) + 1,1)
    else:
        return slice(int((N - 1)/2),int((N + 3)/2),1)


def iter2slices(k,N):
    pairs = []
    a = 0
    if not k:
        pairs.append(slice(None))
        return pairs
    for ki in k:
        if ki < 0:
            ki = ki + N
        if ki != a:
            pairs.append(slice(a,ki))
            a = ki + 1
    if k[-1] + 1 < N:
        pairs.append(slice(k[-1] + 1,N))
    return pairs

def accessLoL(listOfLists,tupleOfInds):
    return tuple([listOfLists[ii][tupleOfInds[ii]] for ii in range(len(tupleOfInds))])
