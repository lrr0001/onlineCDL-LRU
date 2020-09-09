import numpy

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

