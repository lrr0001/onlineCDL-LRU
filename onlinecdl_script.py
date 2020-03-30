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
import sherman_morrison_python_functions

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

imagenames = ['barbara.png','kodim23.png','monarch.png','sail.png','tulips.png']

start = [(10,100),(0,60),(0,160),(0,210),(0,30)]

# S is a list of high-freqency images (512 x 512)
S = []
Sl = []
for imgnum in range(len(start)):
    temp = exim.image(imagenames[imgnum], idxexp=np.s_[start[imgnum][0]:start[imgnum][0] + 512,start[imgnum][1]:start[imgnum][1] + 512])
    sl,sh = util.tikhonov_filter(temp, fltlmbd, npd)
    S.append(sh)
    Sl.append(sl)

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
D0 = np.random.randn(8, 8, noc, 1, nof)
Df0 = sporco.linalg.fftn(D0,s=[78,78],axes=(0,1))
u,vh,Df = sherman_morrison_python_functions.lowRankApprox(a = Df0, projIter=5,axisu = dimN,axisv=dimN + 2,dimN=dimN)

R = sherman_morrison_python_functions.computeNorms(Df,dimN=2)
idmat = np.zeros((78,78,1,1,nof,nof))
for mm in range(nof):
    idmat[:,:,:,:,slice(mm,mm+1),slice(mm,mm+1)] = 1
ainv = np.copy(idmat)

for ii in [0,1]:
    for jj in [0,1]:
        uhu = sporco.linalg.inner(np.conj(u[ii]),u[jj],dimN)
        uhu = uhu.reshape(uhu.shape + (1,))
        vH = np.reshape(vh[jj],vh[jj].shape + (1,))
        v = np.reshape(np.conj(vh[ii]),vh[ii].shape[0:-1] + (1,) + (vh[ii].shape[-1],))
        ainv = sherman_morrison_python_functions.mdbi_sm_r1update(ainv,u=v, vt=uhu*vH, dimN=dimN)

"""
Set regularization parameter and options for dictionary learning solver.
"""

lmbda = 0.002
opt = onlinecdl.OnlineConvBPDNDictLearn.Options({
                'Verbose': False, 'ZeroMean': False, 'eta_a': 10.0,
                'eta_b': 20.0, 'DataType': np.float32,
                'CBPDN': {'rho': 0.005, 'AutoRho': {'Enabled': True},
                    'RelaxParam': 1.0, 'RelStopTol': 1e-7, 'MaxMainIter': 50,
                    'FastSolve': False, 'DataType': np.complex128}})


# need to define W and W1
W = np.full((increment[0] + 2*(filterSz[0] - 1), increment[1] + 2*(filterSz[1] - 1)), False, dtype='bool')
W1 = np.full((increment[0] + 2*(filterSz[0] - 1), increment[1] + 2*(filterSz[1] - 1)), False, dtype='bool')
W[slice(filterSz[0] - 1,filterSz[0] + increment[0] - 1), slice(filterSz[1] - 1,filterSz[1] + increment[1] - 1)] = True
W1[slice(filterSz[0] - 1,filterSz[0] + increment[0] - 1), slice(filterSz[1] - 1,filterSz[1] + increment[1] - 1)] = True
W = W.reshape(W.shape +  3*(1,))
W1 = W1.reshape(W1.shape + 3*(1,))

"""
Create solver object and solve.
"""
d = onlineCDL_lowrankupdates.OnlineConvBPDNDictLearnLRU(Ainv=ainv, Df0=Df,W=W,W1=W1,dsz=filterSz + (noc,),lmbda=lmbda,projIter=5, opt=opt)

iter = 50
d.display_start()
#for it in range(iter):
#    img_index = np.random.randint(0, sh.shape[-1])
#    print(sh[...,[img_index]].shape)
#    d.solve(sh[..., [img_index]])
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
            Scurr = S[imgnum][np.ix_(cinds,rinds)]
            d.solve(Scurr)




d.display_end()
D1 = d.getdict()
print("OnlineConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))


"""
Display initial and final dictionaries.
"""
D0 = D0.squeeze()
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


# Wait for enter on keyboard
input()
