import sherman_morrison_python_functions as sm
import numpy as np

idmat = np.zeros((4,5,1,1,3,3))
for inds in sm.loop_magic((4,5,1,1)):
    idmat[inds] = np.identity(3)




D = np.ones((4,5,1,1,5,3),dtype=np.cdouble)/2.
Q = sm.factoredMatrix_chol(D)


u = np.ones((4,5,1,1,5),dtype=np.cdouble)
u[:] = np.array([1.0j + 0.5,-1.0j + 0.5,2.0j,-2.0j,0.]).reshape((1,1,1,1,5))
v = np.ones((4,5,1,1,3),dtype=np.cdouble)
v[:] = np.array([-1.,-1.,1.0j]).reshape((1,1,1,1,3))


x1,x2,x3 = Q.get_update_vectors(u,v,D)

u = u.reshape(u.shape + (1,))
v = v.reshape(v.shape + (1,))

x1 = x1.reshape(x1.shape + (1,))
x2 = x2.reshape(x2.shape + (1,))
x3 = x3.reshape(x3.shape + (1,))

x1prod = np.matmul(x1,sm.conj_tp(x1))
x2prod = np.matmul(x2,sm.conj_tp(x2))
x3prod = np.matmul(x3,sm.conj_tp(x3))

dhuvh = np.matmul(np.matmul(sm.conj_tp(D),u),sm.conj_tp(v))

vuhd = np.matmul(v,np.matmul(sm.conj_tp(u),D))

print((dhuvh + vuhd)[0,0])

print((x1prod - x2prod)[0,0])








#D_new = D + u.reshape(u.shape + (1,))*np.conj(v.reshape((4,5,1,1,1,3)))


#print((idmat + np.matmul(np.conj(np.swapaxes(D_new,4,5)),D_new))[0,0])
