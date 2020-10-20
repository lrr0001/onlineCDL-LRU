

from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw
import optparse

parser = optparse.OptionParser()
parser.add_option("-i","--runnumber",action="store",type="int",dest="ii",help="selects which rho to use",metavar="RUNNUM")
parser.add_option("-N","--numberofruns",action="store",type="int",dest="N",help="number of rho values in sweep",metavar="NUMOFRUN")
parser.add_option("-o","--outputfile",action="store",type="string",dest="outputfile",help="name of output file",metavar="OUT")

#parser.add_option("-m","--mu",action="store",type="float",dest="mu",help="hyperparameter mu for salt-and-pepper denoising",metavar="MU")

(options,args) = parser.parse_args()

minrho = 1/100
maxrho = 100
rho = minrho*(maxrho/minrho)**(float(options.ii)/(options.N - 1))
noepochs = 2

#mu = options.mu
N = options.N
outputfile = options.outputfile

import pickle
import numpy as np
fid = open('initial_dictionary.pckl','rb')
D = pickle.load(fid)
Df = pickle.load(fid)
fid.close()

C = D.shape[-2]
nof = D.shape[-1]
fltrsz = D.shape[:-2]
framesz = Df.shape[:-2]
dimN = len(fltrsz)

import sherman_morrison_python_functions as sm
Q = sm.factoredMatrix_chol(sm.DfMatRep(Df.reshape(framesz + (C,) + (1,) + (nof,)),axisu = dimN,axisv = dimN + 2),dtype=np.complex128,rho = rho)

import onlineCDL_lowrankupdates
import sporco.dictlrn.onlinecdl



lmbda = 1e-1
n_components=3
opt = sporco.dictlrn.onlinecdl.OnlineConvBPDNDictLearn.Options({
                'Verbose': False, 'ZeroMean': False, 'eta_a': 100.0,
                'eta_b': 200.0, 'DataType': np.complex128,
                'CBPDN': {'Verbose': False, 'rho': rho, 'AutoRho': {'Enabled': False},
                    'RelaxParam': 1.0, 'RelStopTol': 1e-7, 'MaxMainIter': 50,
                    'FastSolve': False, 'DataType': np.complex128}})


# need to define W and W1
W = np.full(framesz, False, dtype='bool')
W1 = np.full(framesz, False, dtype='bool')
W1[slice(fltrsz[0] - 1,framesz[0] - fltrsz[0] + 1), slice(fltrsz[1] - 1,framesz[1] - fltrsz[1] + 1)] = True
W = W.reshape(W.shape +  3*(1,))
W1 = W1.reshape(W1.shape + 3*(1,))

W[:] = True
#W1[:] = True

"""
Create solver object and solve.
"""
d = onlineCDL_lowrankupdates.OnlineConvBPDNDictLearnLRU(Q=Q, Df0=Df,W=W,W1=W1,dsz=fltrsz + (C,) + (nof,),lmbda=lmbda,projIter=5, n_components=n_components,opt=opt)



d.display_start()
#for it in range(iter):
#    img_index = np.random.randint(0, sh.shape[-1])
#    print(sh[...,[img_index]].shape)
#    d.solve(sh[..., [img_index]])
for epoch in range(noepochs):
    fid = open('scene6_frames.pckl','rb')
    while 1:
        try:
            temp = pickle.load(fid)
        except (EOFError):
            break
        print(temp.dtype)
        print(np.amin(temp))
        print(np.amax(temp))
        print(temp.shape)
        d.solve(temp)
    fid.close()


