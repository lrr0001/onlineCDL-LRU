import sporco
import sporco.dictlrn.onlinecdl
#import cbpdn_fixed_rho
import cbpdn_factoredInv
from sporco import util
from sporco import common
from sporco.util import u
import sporco.linalg as sl
import sporco.cnvrep as cr
from sporco.admm import cbpdn
from sporco import cuda
from sporco.dictlrn import dictlrn
import sherman_morrison_python_functions as sm
import numpy




class OnlineConvBPDNDictLearnLRU(sporco.dictlrn.onlinecdl.OnlineConvBPDNDictLearn):
    r"""
    Stochastic gradient descent (SGD) based online convolutional
    dictionary learning
    """

    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    """Fields in IterationStats associated with the objective function"""
    itstat_fields_alg = ('PrimalRsdl', 'DualRsdl', 'Rho', 'Cnstr',
                         'DeltaD', 'Eta')
    """Fields in IterationStats associated with the specific solver
    algorithm"""
    itstat_fields_extra = ()
    """Non-standard fields in IterationStats; see :meth:`itstat_extra`"""




    def __init__(self, Q, Df0, W, W1, dsz=None, lmbda=None, projIter=5, opt=None, dimK=None, dimN=2):
        """
        Parameters
        ----------
        D0 : array_like
          Initial dictionary array
        lmbda : float
          Regularisation parameter
        opt : :class:`OnlineConvBPDNDictLearn.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of signal dimensions in signal array passed to
          :meth:`solve`. If there will only be a single input signal
          (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = OnlineConvBPDNDictLearnLRU.Options()
        if not isinstance(opt, OnlineConvBPDNDictLearnLRU.Options):
            raise TypeError('Parameter opt must be an instance of '
                            'OnlineConvBPDNDictLearnLRU.Options')
        self.opt = opt

        if dimN != 2 and opt['CUDA_CBPDN']:
            raise ValueError('CUDA CBPDN solver can only be used when dimN=2')

        if opt['CUDA_CBPDN'] and cuda.device_count() == 0:
            raise ValueError('SPORCO-CUDA not installed or no GPU available')

        self.Q = Q
        self.W = W
        self.W1 = W1
        self.projIter = projIter
        self.dimK = dimK
        self.dimN = dimN

        # DataType option overrides data type inferred from __init__
        # parameters of derived class
        self.set_dtype(opt, Df0.dtype)

        # Initialise attributes representing algorithm parameter
        self.lmbda = lmbda
        self.eta_a = opt['eta_a']
        self.eta_b = opt['eta_b']
        self.set_attr('eta', opt['eta_a'] / opt['eta_b'],
                      dval=2.0, dtype=self.dtype)

        # Get dictionary size
        if self.opt['DictSize'] is None:
            self.dsz = dsz
        else:
            self.dsz = self.opt['DictSize']

        # Construct object representing problem dimensions
        self.cri = None

        # Normalise dictionary
        ds = sporco.cnvrep.DictionarySize(self.dsz, dimN)
        dimCd = ds.ndim - dimN - 1
        #D0 = sporco.cnvrep.stdformD(D0, ds.nchn, ds.nflt, dimN).astype(self.dtype)
        #self.D = cr.Pcn(D0, self.dsz, (), dimN, dimCd, crp=True,
                        #zm=opt['ZeroMean'])
        #self.Dprv = self.D.copy()

        # Need to decide whether the expected input dictionary shape.
        self.Dfshape = Df0.shape
        cri = cr.CSC_ConvRepIndexing(Df0,Df0[0:dimN + 1])
        self.Df = Df0.reshape(cri.shpD)
        self.Gf = self.Df
        temp = sporco.linalg.ifftn(self.Df,[self.Dfshape[ii] for ii in range(0,dimN)],tuple(range(0,dimN)))
        self.Gprv = temp[0:self.dsz[0],0:self.dsz[1]]
        self.G = self.Gprv.copy()
        self.R = sm.computeNorms(self.G)
        print('Is D0 real?')
        complexGf = self.Gf - sm.conj_sym_proj(self.Gf,range(self.dimN))
        print(numpy.amax(numpy.abs(complexGf)))

        # Create constraint set projection function
        self.Pcn = sporco.cnvrep.getPcn(self.dsz, (), dimN, dimCd, crp=True,
                             zm=opt['ZeroMean'])

        # Initalise iterations stats list and iteration index
        self.itstat = []
        self.j = 0

        # Configure status display
        self.display_config()



    def __new__(cls, *args, **kwargs):
        """Create an OnlineConvBPDNDictLearn object and start its
        initialisation timer."""

        instance = super(OnlineConvBPDNDictLearnLRU, cls).__new__(cls)
        instance.timer = util.Timer(['init', 'solve', 'solve_wo_eval'])
        instance.timer.start('init')
        return instance

    def init_vars(self, S, dimK):
        """Initalise variables required for sparse coding and dictionary
        update for training data `S`."""

        Nv = S.shape[0:self.dimN]
        if self.cri is None or Nv != self.cri.Nv:
            self.cri = cr.CDU_ConvRepIndexing(self.dsz, S, dimK, self.dimN)
            if self.opt['CUDA_CBPDN']:
                if self.cri.Cd > 1 or self.cri.Cx > 1:
                    raise ValueError('CUDA CBPDN solver can only be used for '
                                     'single channel problems')
                if self.cri.K > 1:
                    raise ValueError('CUDA CBPDN solver can not be used with '
                                     'mini-batches')
                else:
                    raise NotImplementedError()
            #self.Df = sl.pyfftw_byte_aligned(sl.fftn(self.D, self.cri.Nv,
                                                      #self.cri.axisN))
            #self.Gf = sl.pyfftw_empty_aligned(self.Df.shape, self.Df.dtype)
            #self.Z = sl.pyfftw_empty_aligned(self.cri.shpX, self.dtype)
        #else:
        #    self.Df[:] = sl.fftn(self.D, self.cri.Nv, self.cri.axisN)



    def xstep(self, S, lmbda, dimK):
        """Solve CSC problem for training data `S`."""

        if self.opt['CUDA_CBPDN']:
            raise NotImplementedError()
        else:
            Sf = sporco.linalg.fftn(S,s=self.cri.Nv,axes=self.cri.axisN).reshape(S.shape[0:self.dimN + 1]  + 2*(1,))
            if S.ndim == self.dimN + 1:
                self.Sf = Sf.reshape(S.shape[0:self.dimN + 1] + 2*(1,))
                #S = S.reshape(S.shape[0:self.dimN + 1] + 2*(1,))
            elif S.ndim == self.dimN + 2:
                self.Sf = Sf.reshape(S.shape[0:self.dimN + 2] + (1,))
                #S = S.reshape(S.shape[0:self.dimN + 2] + (1,))
            else:
                raise TypeError('Signal array must have Ndim + 1 or Ndim + 2 dimensions.')
            #print('Is the signal real?')
            #complexSf = self.Sf - sm.conj_sym_proj(self.Sf,self.cri.axisN)
            #print(numpy.amax(numpy.abs(complexSf)))

            #print('What if we realize the signal, haha?')
            #self.Sf = sm.conj_sym_proj(self.Sf,self.cri.axisN)
            #complexSf = self.Sf - sm.conj_sym_proj(self.Sf,self.cri.axisN)
            #print(numpy.amax(numpy.abs(complexSf)))

            #print('Hold on, just a sanity check here... Does our conjegate symmetric projection actually work?')
            #S2 = sporco.linalg.ifftn(self.Sf,s=self.cri.Nv,axes=self.cri.axisN)
            #print(numpy.amax(numpy.abs(S - numpy.squeeze(S2))))

            # apparently, W isn't supposed to be boolean.
            #xstep = cbpdn_fixed_rho.CBPDN_FactoredFixedRho(Q=self.Q, DR=self.Df.reshape(self.Dfshape),S=S, R=self.R, W=self.W, W1=self.W1, lmbda=lmbda, dimN=self.dimN, opt=self.opt['CBPDN'])
            xstep = cbpdn_factoredInv.CBPDN_Factored(self.Q, DR=self.Df.reshape(self.Dfshape),S=S,R=self.R,W=self.W,lmbda=lmbda,dimN = self.dimN,opt=self.opt['CBPDN'])
            temp = xstep.solve()
            #import pdb; pdb.set_trace()
            self.Zf = xstep.getcoef()
            self.Zf = self.Zf.reshape(self.cri.shpX)
            #complexZf = self.Zf - sm.conj_sym_proj(self.Zf,range(self.dimN))
            #print('Are the coefficients real?')
            #print(numpy.amax(numpy.abs(complexZf)))

            #print('How large are these coeficients?')
            #print('frequency:')
            #print(numpy.amax(numpy.abs(self.Zf)))
            #print('spatial:')
            #print(numpy.amax(numpy.abs(sporco.linalg.ifftn(self.Zf,self.cri.Nv, self.cri.axisN))))
            self.xstep_itstat = xstep.itstat[-1] if xstep.itstat else None
            #print('xstep complete.')



#    def setcoef(self, Z):
#        """Set coefficient array."""

        # If the dictionary has a single channel but the input (and
        # therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        #if self.cri.Cd == 1 and self.cri.C > 1:
        #    Z = Z.reshape(self.cri.Nv + (1,) + (self.cri.Cx*self.cri.K,) +
        #                  (self.cri.M,))
        #self.Z[:] = np.asarray(Z, dtype=self.dtype)
        #self.Zf = sl.fftn(self.Z, self.cri.Nv, self.cri.axisN)
#        self.Zf = Z

    def dstep(self):
        """Compute dictionary update for training data of preceding
        :meth:`xstep`.
        """
        print(self.j)
        # Compute X D - S
        Ryf = sl.inner(self.Zf, self.Gf, axis=self.cri.axisM) - self.Sf

        #print('Maximum coefficient magnitude:')
        #print(numpy.amax(numpy.abs(self.Zf)))

        # Filter out elements that do not contribute to error.
        Ry = sl.ifftn(Ryf,self.cri.Nv, self.cri.axisN)
        Ry = self.W*Ry
        Ryf = sl.fftn(Ry, self.cri.Nv, self.cri.axisN)

        # Compute gradient

        gradf = sl.inner(numpy.conj(self.Zf), Ryf, axis=self.cri.axisK)
        print('Is grad real?')
        realgradf = sm.conj_sym_proj(gradf,range(self.dimN))
        complexgrad = gradf - realgradf
        print(numpy.amax(numpy.abs(complexgrad)))
        gradf = realgradf
        print('Maximum value gradf:')
        print(numpy.amax(numpy.abs(gradf)))

        
        # If multiple channel signal, single channel dictionary
        #if self.cri.C > 1 and self.cri.Cd == 1:
        #    gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        # Update gradient step
        self.eta = self.eta_a / (self.j + self.eta_b)
        
        #gradfr = sl.ifftn(gradf, self.cri.dsz[0:-2], self.cri.axisN) # something weird is going on here. Result should be real, but it is not.
        gradfr = sl.ifftn(gradf,self.cri.Nv, self.cri.axisN)
        self.Gprv = self.G

        #self.G = self.Pcn(self.G - self.eta*gradfr)
        newG = self.G - self.eta*gradfr[0:self.cri.dsz[0],0:self.cri.dsz[1]] # assumes dimN = 2
        newG = newG - numpy.mean(newG,axis=self.cri.axisN,keepdims=True)
        self.G = newG/sm.computeNorms(newG)
        #print('Maximum value G magnitude:')
        #print(numpy.amax(numpy.abs(self.G)))

        self.Gf = sl.fftn(self.G, self.cri.Nv,self.cri.axisN)
        #print('Is G real?')
        realGf = sm.conj_sym_proj(self.Gf,range(self.dimN))
        complexGf = self.Gf - realGf
        #print(numpy.amax(numpy.abs(complexGf)))
        self.Gf = realGf
        #import pdb; pdb.set_trace()

        realgfmdf = sm.conj_sym_proj(self.Gf - self.Df,range(self.dimN))
        complexgfmdf = self.Gf - self.Df - realgfmdf
        #print('Is G - D real?')
        #print(numpy.amax(numpy.abs(complexgfmdf)))
        
        (u,vH,dupdate) = sm.lowRankApprox(a=self.Gf -self.Df,projIter=self.projIter,axisu=self.cri.axisC,axisv=self.cri.axisM,dimN=self.dimN)

        #print('Is dupdate real?')
        #realdupdate = sm.conj_sym_proj(dupdate,range(self.dimN))
        #complexdupdate = dupdate - realdupdate
        #print(numpy.amax(numpy.abs(complexdupdate)))

        #print(u[1].shape)
        #print('Is u[0] real?')
        #realu = sm.conj_sym_proj(u[0],range(self.dimN))
        #complexu = u[0] - realu
        #print(numpy.amax(numpy.abs(complexdupdate)))

        #print(vH[0].shape)
        #print('Is vH[1] real?')
        #realvh = sm.conj_sym_proj(vH[1],range(self.dimN))
        #complexvh = vH[1] - realvh
        #print(numpy.amax(numpy.abs(complexvh)))

        print('Low-rank update fractional error:')
        print(numpy.sqrt(numpy.sum(numpy.conj(dupdate - self.Gf + self.Df)*(dupdate - self.Gf + self.Df)))/numpy.sqrt(numpy.sum(numpy.conj(self.Gf - self.Df)*(self.Gf - self.Df))))

        for ii in range(0,2):
            #Dftemp = Df
            #Dftemp = sm.addDim(Dftemp)
            #dftemp = numpy.swapaxes(Dftemp,self.cri.axisM,-1)
            #dftemp = numpy.swapaxes(Dftemp,self.cri.axisC,-2)
            self.Q.update(sm.uMatRep(u[ii],self.cri.axisC,self.cri.axisM),sm.vMatRep(vH[ii],self.cri.axisC,self.cri.axisM),sm.DfMatRep(self.Df,self.cri.axisC,self.cri.axisM))
            #self.Q.update(numpy.swapaxes(u[ii],self.cri.axisC,-1),numpy.conj(numpy.swapaxes(vH[ii],self.cri.axisM,-1)),self.Dftemp)
            self.Df = self.Df + u[ii]*vH[ii]
        
            print('Is D real?')
            realDf = sm.conj_sym_proj(self.Df,range(self.dimN))
            complexDf = self.Df - realDf
            print(numpy.amax(numpy.abs(complexDf)))
            self.Df = realDf
            print('Inverse check:')
            print(self.Q.inv_check_ls(sm.DfMatRep(self.Df,self.cri.axisC,self.cri.axisM)))
            
        
        self.R = sm.computeNorms(self.Df.reshape(self.Dfshape)/numpy.sqrt(numpy.prod(self.cri.Nv)))
        print('minimum element R:')
        print(numpy.amin(numpy.abs(self.R)))
        print('maximum element R:')
        print(numpy.amax(numpy.abs(self.R)))
        #D = sporco.linalg.ifftn(self.Df, self.cri.Nv,self.cri.axisN)
        #R = sm.computeNorms(D[0:self.dsz[0],0:self.dsz[1]])
        #tempD = self.getdict()
        #print('Maximum dictionary magnitude:')
        #print(numpy.amax(numpy.abs(tempD)))
        #input()
        #if self.j == 1:
         #   import pickle
          #  fid = open('badDictionary.pkl','wb')
           # pickle.dump(self.Df,fid)
            #pickle.dump(self.R,fid)
            #pickle.dump(self.Q,fid)
            #pickle.dump(self.Gf,fid)
            #fid.close()

    def getdict(self):
        """Get final dictionary."""

        dict = sporco.linalg.ifftn(self.Df, self.cri.Nv,self.cri.axisN) / self.R
        return dict[0:self.dsz[0],0:self.dsz[1]]
    def iteration_stats(self):
        """Construct iteration stats record tuple."""
        tk = self.timer.elapsed(self.opt['IterTimer'])
        if self.xstep_itstat is None:
            objfn = (0.0,) * 3
            rsdl = (0.0,) * 2
            rho = (0.0,)
        else:
            # Had to change RegL1 to Reg, because of CBPDN class
            objfn = (self.xstep_itstat.ObjFun, self.xstep_itstat.DFid,
                     self.xstep_itstat.Reg)
            rsdl = (self.xstep_itstat.PrimalRsdl,
                    self.xstep_itstat.DualRsdl)
            rho = (self.xstep_itstat.Rho,)


        # These next two lines are specific to this class, which is why the parent method is not used.
        cnstr = numpy.linalg.norm(self.Df - self.Gf) / numpy.sqrt(numpy.prod(self.cri.Nv))
        dltd = numpy.linalg.norm(self.G - self.Gprv)

        tpl = (self.j,) + objfn + rsdl + rho + (cnstr, dltd, self.eta) + \
              self.itstat_extra() + (tk,)
        return type(self).IterationStats(*tpl)


    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return ()



