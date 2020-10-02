import optparse

parser = optparse.OptionParser()
parser.add_option("-i","--runnumber",action="store",type="int",dest="ii",help="selects which rho to use",metavar="RUNNUM")
parser.add_option("-N","--numberofruns",action="store",type="int",dest="N",help="number of rho values in sweep",metavar="NUMOFRUN")
parser.add_option("-m","--mu",action="store",type="float",dest="mu",help="hyperparameter mu for salt-and-pepper denoising",metavar="MU")
parser.add_option("-o","--outputfile",action="store",type="string",dest="outputfile",help="name of output file",metavar="OUT")

(options,args) = parser.parse_args()

minrho = 1/100
maxrho = 100
rho = minrho*(maxrho/minrho)**(float(options.ii)/(options.N - 1))


mu = options.mu
N = options.N
outputfile = options.outputfile

import pickle

fid = open(outputfile,'wb')
pickle.dump(mu,fid)
pickle.dump(rho,fid)
pickle.dump(N,fid)
fid.close()
