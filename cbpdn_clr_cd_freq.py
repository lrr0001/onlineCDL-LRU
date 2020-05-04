#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Multi-channel CSC
=================

This example demonstrates solving a convolutional sparse coding problem with a colour dictionary and a colour signal :cite:`wohlberg-2016-convolutional`

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \sum_c \left\| \sum_m \mathbf{d}_{c,m} * \mathbf{x}_m -\mathbf{s}_c \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 \;,$$

where $\mathbf{d}_{c,m}$ is channel $c$ of the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_m$ is the coefficient map corresponding to the $m^{\text{th}}$ dictionary filter, and $\mathbf{s}_c$ is channel $c$ of the input image.
"""


from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import plot
import sporco.metric as sm
from sporco.admm import cbpdn
import sporco.linalg
import sherman_morrison_python_functions
import cbpdn_factored_freq

"""
Load example image.
"""

img = util.ExampleImages().image('kodim23.png', scaled=True,
                                 idxexp=np.s_[160:416,60:316])


"""
Highpass filter example image.
"""

npd = 16
fltlmbd = 10
sl, sh = util.tikhonov_filter(img, fltlmbd, npd)


"""
Load colour dictionary and display it.
"""

D = util.convdicts()['RGB:8x8x3x64']
D = D[:,:,:,slice(0,32)]
plot.imview(util.tiledict(D), fgsz=(7, 7))


R = sherman_morrison_python_functions.computeNorms(v=D,dimN=2)
D = D/R
R = sherman_morrison_python_functions.computeNorms(v=D,dimN=2)

print(R)

# code for testing:
# compute A inverse
Df = sporco.linalg.fftn(D,sh.shape[0:2],(0,1))
[a,b,noc,nof]= Df.shape
#sh = np.zeros((2,1,1))
#Df = np.array([[1,2,1],[2,-1,2]])
#Df = Df.reshape((2,1,1,3))
#ainv = sherman_morrison_python_functions.woodburyIpUV(Df, sh.shape[0], sh.shape[1], Df.shape[2], Df.shape[3])
Q = sherman_morrison_python_functions.factoredMatrix_qr(Df.reshape((a,b,1,1,noc,nof)))


Df_internal = Df.reshape(a,b,1,1,noc,nof)
 
idmat = np.zeros((a,b,1,1,nof,nof),dtype=np.complex128)
for inds in sherman_morrison_python_functions.loop_magic((a,b,1,1)):
    idmat[inds] = np.identity(nof)
 
dhdpi = idmat + np.matmul(sherman_morrison_python_functions.conj_tp(Df_internal),Df_internal)
approx_idmat = Q.inv_mat(dhdpi,Df_internal)
print('inverse test:')
print(np.sum(np.abs(idmat - approx_idmat)))



#print(66*ainv)
#print(66*ainv2)

#print(np.sum(np.abs(ainv - ainv2)))

# compute A
#idmat = np.zeros((sh.shape[0],sh.shape[1],1,1,Df.shape[3],Df.shape[3]))
#for mm in range(Df.shape[3]):
#    idmat[:,:,:,:,slice(mm,mm+1),slice(mm,mm+1)] = 1
#d = Df.reshape((sh.shape[0],sh.shape[1],Df.shape[2],1,1,Df.shape[3]))
#dt = np.conj(Df.reshape((sh.shape[0],sh.shape[1],Df.shape[2],1,Df.shape[3],1)))
#a = idmat + sporco.linalg.inner(dt,d,axis=2)

# Is A inverse the inverse of A?
#print(np.sum(np.abs(idmat - np.matmul(a,ainv2))))
#print(np.sum(np.abs(idmat - np.matmul(ainv2,a))))

#idmat = np.zeros((sh.shape[0],sh.shape[1],Df.shape[3],Df.shape[3]))
#for mm in range(Df.shape[3]):
#    idmat[:,:,slice(mm,mm+1),slice(mm,mm+1)] = 1
#dt = np.conj(np.swapaxes(Df,2,3))
#a = idmat + np.matmul(dt,Df)
#print(np.max(np.abs(idmat - np.matmul(a,Q.inv(idmat,)))))
#print(np.max(np.abs(idmat - np.matmul(ainv,a))))
#ainv = ainv.reshape(ainv.shape[0:2] + (1,1,ainv.shape[2],ainv.shape[3]))



lmbda = 1e-1
#opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
 #                             'RelStopTol': 5e-3, 'AuxVarObj': False})
opt = cbpdn_factored_freq.CBPDN_FactoredScaledDict.Options({'Verbose': False, 'rho': 0.05, 'AutoRho': {'Enabled': True},
                    'RelaxParam': 1.0, 'RelStopTol': 1e-7, 'MaxMainIter': 50,
                    'FastSolve': False, 'DataType': np.complex128})


"""
Initialise and run CSC solver.
"""

#b = cbpdn.ConvBPDN(D, sh, lmbda, opt)
#X = b.solve()
#print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'))

W = np.ones((sh.shape[0],sh.shape[1],1,1,1))
W1 = np.ones((sh.shape[0],sh.shape[1],1,1,1))

Df = Df.reshape((Df.shape[0],Df.shape[1],Df.shape[2],Df.shape[3]))
# code for testing
#b2 = cbpdn_freq.CBPDN_ScaledDict(Ainv=ainv, DR=Df, R=R, S=sh, W=W, W1=W1, lmbda=lmbda, Ndim=2, opt=opt2)
b2 = cbpdn_factored_freq.CBPDN_FactoredScaledDict(Q=Q,DR=Df,S=sh,R=R,W=W,W1=W1,lmbda=lmbda,dimN=2,opt=opt)

x2 = b2.solve()



"""
Reconstruct image from sparse representation.
"""

#shr = b.reconstruct().squeeze()
#imgr = sl + shr
#print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(img, imgr))


#"""
#Display low pass component and sum of absolute values of coefficient maps of highpass component.
#"""

#fig = plot.figure(figsize=(14, 7))
#plot.subplot(1, 2, 1)
#plot.imview(sl, title='Lowpass component', fig=fig)
#plot.subplot(1, 2, 2)
#plot.imview(np.sum(abs(X), axis=b.cri.axisM).squeeze(), cmap=plot.cm.Blues,
#            title='Sparse representation', fig=fig)
#fig.show()


"""
Display original and reconstructed images.
"""

#fig = plot.figure(figsize=(14, 7))
#plot.subplot(1, 2, 1)
#plot.imview(img, title='Original', fig=fig)
#plot.subplot(1, 2, 2)
#plot.imview(imgr, title='Reconstructed', fig=fig)
#fig.show()


"""
Get iterations statistics from solver object and plot functional value, ADMM primary and dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
"""

#its = b.getitstat()
#fig = plot.figure(figsize=(20, 5))
#plot.subplot(1, 3, 1)
#plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
#plot.subplot(1, 3, 2)
#plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
#          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
#          lgnd=['Primal', 'Dual'], fig=fig)
#plot.subplot(1, 3, 3)
#plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
#fig.show()


# Wait for enter on keyboard
#input()
