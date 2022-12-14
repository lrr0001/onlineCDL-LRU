#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Online Convolutional Dictionary Learning
========================================

This example demonstrates the use of :class:`.dictlrn.onlinecdl.OnlineConvBPDNDictLearn` for learning a convolutional dictionary from a set of training images. The dictionary is learned using the online dictionary learning algorithm proposed in :cite:`liu-2018-first`.
"""


from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

import onlineCDL_lowrankupdates
import sporco.dictlrn.onlinecdl as onlinecdl
import sporco.linalg
from sporco import util
from sporco import plot
import sherman_morrison_python_functions as sm

"""
Load training images.
"""

exim = util.ExampleImages(scaled=True, zoom=0.25)

dimN = 2
npd = 16
fltlmbd = 5
increment = (64,64)
start = (10,100)
end = (522,612)
filterSz = (8,8)
noc = 3
nof = 64
noepochs = 1
mu = [1.0,1.0]

imagenames = ['barbara.png','kodim23.png','monarch.png','sail.png','tulips.png']

start = [(10,100),(0,60),(0,160),(0,210),(0,30)]

# S is a list of high-freqency images (512 x 512)
S = []
Sl = []
Sh = []
for imgnum in range(len(start)):
    temp = exim.image(imagenames[imgnum], idxexp=np.s_[start[imgnum][0]:start[imgnum][0] + 512,start[imgnum][1]:start[imgnum][1] + 512])
    sl,sh = util.tikhonov_filter(temp, fltlmbd, npd)
    Sh.append(sh)
    Sl.append(sl)
    S.append(sh + sl)

#S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
#S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
#S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
#S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
#S5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
#S = np.stack((S1, S2, S3, S4, S5), axis=3)


#"""
#Highpass filter training images.
#"""

#npd = 16
#fltlmbd = 5
#sl, sh = util.tikhonov_filter(S, fltlmbd, npd)


"""
Construct initial dictionary.
"""

np.random.seed(12345)
#D0 = np.random.randn(8, 8, noc, 1, nof)
#Df0 = sporco.linalg.fftn(D0,s=[78,78],axes=(0,1))
#u,vh,Df = sherman_morrison_python_functions.lowRankApprox(a = Df0, projIter=5,axisu = dimN,axisv=dimN + 2,dimN=dimN)

#Q = sherman_morrison_python_functions.factoredMatrix_chol(Df)

#R = sherman_morrison_python_functions.computeNorms(Df,dimN=2)
Dutil = util.convdicts()['RGB:8x8x3x64']
Drand = np.random.randn(8,8,3,64)
Rutil = sm.computeNorms(v=Dutil,dimN=2)
Rrand = sm.computeNorms(v=Drand,dimN=2)
D = Dutil/Rutil + 0.5*Drand/Rrand
R = sm.computeNorms(v=D,dimN=2)
D = D/R
R = sm.computeNorms(v=D,dimN=2)
rho = 1

Df = np.empty((increment[0] + 2*(filterSz[0] - 1),increment[1] + 2*(filterSz[1] - 1)) + D.shape[2:],dtype=np.complex128)

Df = sporco.linalg.fftn(D,(increment[0] + 2*(filterSz[0] - 1),increment[1] + 2*(filterSz[1] - 1)),(0,1))
Df = (Df + sm.conj_sym_proj(Df,range(2)))/2
[a,b,noc,nof]= Df.shape
slice0 = sm.highFreqSlice(a)
slice1 = sm.highFreqSlice(b)
print(slice0)
print(slice1)
Q = sm.factoredMatrix_cloneSlices(D = Df.reshape((a,b,1,1,noc,nof)),dtype=np.complex128,rho=rho,dimN=2,clonedSlices =[slice0,slice1,],clonedRhos=[mu[ii]/rho + rho for ii in range(len(mu))])
#Q = sherman_morrison_python_functions.factoredMatrix_chol(Df.reshape((a,b,1,1,noc,nof)),rho=rho)
Qold = sm.factoredMatrix_chol(Df.reshape((a,b,1,1,noc,nof)),rho=rho)

#idmat = np.zeros((78,78,1,1,nof,nof))
#for mm in range(nof):
#    idmat[:,:,:,:,slice(mm,mm+1),slice(mm,mm+1)] = 1
#ainv = np.copy(idmat)

#for ii in [0,1]:
#    for jj in [0,1]:
#        uhu = sporco.linalg.inner(np.conj(u[ii]),u[jj],dimN)
#        uhu = uhu.reshape(uhu.shape + (1,))
#        vH = np.reshape(vh[jj],vh[jj].shape + (1,))
#        v = np.reshape(np.conj(vh[ii]),vh[ii].shape[0:-1] + (1,) + (vh[ii].shape[-1],))
#        ainv = sherman_morrison_python_functions.mdbi_sm_r1update(ainv,u=v, vt=uhu*vH, dimN=dimN)

"""
Set regularization parameter and options for dictionary learning solver.
"""

lmbda = 1e-1
n_components=3
opt = onlinecdl.OnlineConvBPDNDictLearn.Options({
                'Verbose': False, 'ZeroMean': False, 'eta_a': 100.0,
                'eta_b': 200.0, 'DataType': np.complex128,
                'CBPDN': {'Verbose': False, 'rho': rho, 'AutoRho': {'Enabled': False},
                    'RelaxParam': 1.0, 'RelStopTol': 1e-7, 'MaxMainIter': 50,
                    'FastSolve': False, 'DataType': np.complex128}})


# need to define W and W1
W = np.full((increment[0] + 2*(filterSz[0] - 1), increment[1] + 2*(filterSz[1] - 1)), False, dtype='bool')
W1 = np.full((increment[0] + 2*(filterSz[0] - 1), increment[1] + 2*(filterSz[1] - 1)), False, dtype='bool')
W[slice(filterSz[0] - 1,2*filterSz[0] + increment[0] - 2), slice(filterSz[1] - 1,2*filterSz[1] + increment[1] - 2)] = True
W1[slice(filterSz[0] - 1,filterSz[0] + increment[0] - 1), slice(filterSz[1] - 1,filterSz[1] + increment[1] - 1)] = True
W = W.reshape(W.shape +  3*(1,))
W1 = W1.reshape(W1.shape + 3*(1,))

W[:] = True
#W1[:] = True

"""
Create solver object and solve.
"""
d = onlineCDL_lowrankupdates.OnlineConvBPDNDictLearnLRU(Q=Q, Df0=Df,W=W,W1=W1,dsz=filterSz + (noc,) + (nof,),lmbda=lmbda,projIter=5, n_components=n_components,opt=opt)

print('Is D real?')
print(np.amax(np.abs(Df - (Df + sm.conj_sym_proj(Df,range(2)))/2)))

iter = 50
d.display_start()
#for it in range(iter):
#    img_index = np.random.randint(0, sh.shape[-1])
#    print(sh[...,[img_index]].shape)
#    d.solve(sh[..., [img_index]])
for epoch in range(noepochs):
    for imgnum in range(len(S)):
        for ii in range(0,128,increment[0]):
            c = ii - filterSz[0] + 1 
            if c + increment[0] + 2*(filterSz[0] - 1) >= 128:
                cinds = 127 - abs(np.arange(c - 127,c+increment[0] + 2*(filterSz[0] - 1) - 127))
            else:
                cinds = abs(np.arange(c,c+increment[0] + 2*(filterSz[0] - 1)))
            for jj in range(0,128,increment[1]):
                r = jj - filterSz[1]
                if r + increment[1] + 2*(filterSz[1] - 1) >= 128:
                    rinds = 127 - abs(np.arange(r - 127,r+increment[1] + 2*(filterSz[1] - 1) - 127))
                else:
                    rinds = abs(np.arange(r,r+increment[1] + 2*(filterSz[1] - 1)))       
                Scurr = Sh[imgnum][np.ix_(cinds,rinds)]
                temp = d.solve(Scurr)





d.display_end()
D1 = d.getdict()
print("OnlineConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))


"""
Display initial and final dictionaries.
"""
D0 = D.squeeze()
D1 = D1.squeeze()
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1.real), title='D1', fig=fig)
fig.show()


"""
Get iterations statistics from solver object and plot functional value.
"""

its = d.getitstat()
fig = plot.figure(figsize=(7, 7))
plot.plot(np.vstack((its.DeltaD, its.Eta)).T, xlbl='Iterations',
          lgnd=('Delta D', 'Eta'), fig=fig)
fig.show()

fig2 = plot.figure(figsize=(7,7))
plot.plot(np.vstack((its.Cnstr,its.DeltaD)).T, xlbl='Iterations', ylbl='difference',lgnd=['approx','delta'],fig=fig2)

fig2.show()

import cbpdn_factoredInv
import sporco.metric as smet
from sporco import plot
Dfnew = d.Df.squeeze()
for imgnum in range(len(S)):
    for ii in range(0,128,increment[0]):
        c = ii - filterSz[0] + 1 
        if c + increment[0] + 2*(filterSz[0] - 1) >= 128:
            cinds = 127 - abs(np.arange(c - 127,c+increment[0] + 2*(filterSz[0] - 1) - 127))
        else:
            cinds = abs(np.arange(c,c+increment[0] + 2*(filterSz[0] - 1)))
        for jj in range(0,128,increment[1]):
            r = jj - filterSz[1]
            if r + increment[1] + 2*(filterSz[1] - 1) >= 128:
                rinds = 127 - abs(np.arange(r - 127,r+increment[1] + 2*(filterSz[1] - 1) - 127))
            else:
                rinds = abs(np.arange(r,r+increment[1] + 2*(filterSz[1] - 1)))       
            Shcurr = Sh[imgnum][np.ix_(cinds,rinds)]
            bold = cbpdn_factoredInv.CBPDN_Factored(Q=Qold,DR=Df,S=Shcurr,R=R,W=W,lmbda=lmbda,dimN=2,opt=opt['CBPDN'])
            xold = bold.solve()
            bnew = cbpdn_factoredInv.CBPDN_L1DF(Q=d.Q,DR=Dfnew.squeeze(),S=Shcurr,R=d.R,W=W,lmbda=lmbda,mu=mu,dimN=2,opt=opt['CBPDN'])
            #bnew = cbpdn_factoredInv.CBPDN_Factored(Q=d.Q,DR=d.Df.squeeze(),S=Shcurr,R=R,W=W,lmbda=lmbda,dimN=2,opt=opt['CBPDN'])
            xnew = bnew.solve()
            shr = bold.reconstruct().squeeze()
            sl = Sl[imgnum][np.ix_(cinds,rinds)]
            Scurr = S[imgnum][np.ix_(cinds,rinds)]
            imgrold = sl + shr
            print("Initial dictionary reconstruction PSNR: %.2fdB\n" % smet.psnr(Scurr, imgrold.real))
            shr = bnew.reconstruct().squeeze()
            imgrnew = sl + shr
            print("Final dictionary reconstruction PSNR: %.2fdB\n" % smet.psnr(Scurr, imgrnew.real))
            fig = plot.figure(figsize=(14, 14))
            plot.subplot(2,2,1)
            plot.imview(Scurr, title='Original', fig=fig)
            plot.subplot(2,2,2)
            plot.imview(sl, title='Low-pass', fig=fig)
            plot.subplot(2,2,3)
            plot.imview(imgrold.real, title='Reconstruction (initial dictionary)', fig=fig)
            plot.subplot(2,2,4)
            plot.imview(imgrnew.real, title='Reconstruction (final dictionary)', fig=fig)
            plot.subplot(2,2,4)
            fig.show()
            itsold = bold.getitstat()
            itsnew = bnew.getitstat()
            fig = plot.figure(figsize=(14,7))
            plot.subplot(1, 2, 1)
            plot.plot(np.vstack((itsold.ObjFun,itsnew.ObjFun)).T, xlbl='Iterations', ylbl='Functional',lgnd=['init dict','final dict'], fig=fig)
            plot.subplot(1, 2, 2)
            plot.plot(np.vstack((itsold.PrimalRsdl, itsold.DualRsdl,itsnew.PrimalRsdl,itsnew.DualRsdl)).T,
                ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
                lgnd=['Primal-init dict', 'Dual-init dict','Primal-final dict','Dual-final dict'], fig=fig)
            fig.show()
            print("Initial dictionary objective: %.2f\n" % itsold.ObjFun[-1])
            print("Final dictionary objective: %.2f\n" % itsnew.ObjFun[-1])
            
            input()



            
