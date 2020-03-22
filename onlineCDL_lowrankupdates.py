import sporco
import sporco.dictlrn.onlinecdl
import cbpdn_freq
from sporco import util
from sporco import common
from sporco.util import u
import sporco.linalg as sl
import sporco.cnvrep as cr
from sporco.admm import cbpdn
from sporco import cuda
from sporco.dictlrn import dictlrn
import sherman_morrison_python_functions
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




    def __init__(self, Ainv, Df0, W, W1, dsz=None, lmbda=None, projIter=5, opt=None, dimK=None, dimN=2):
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

        self.Ainv = Ainv
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
        self.Gf = Df0
        self.Gprv = sporco.linalg.ifftn(Df0,self.dsz[0:-1],tuple(range(0,dimN)))
        self.G = self.Gprv
        self.Df = Df0
        self.R = sherman_morrison_python_functions.computeNorms(self.Df)

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


    def solve(self, S, dimK=None):
        """Compute sparse coding and dictionary update for training
        data `S`."""

        # Use dimK specified in __init__ as default
        if dimK is None and self.dimK is not None:
            dimK = self.dimK

        # Start solve timer
        self.timer.start(['solve', 'solve_wo_eval'])

        # Solve CSC problem on S and do dictionary step
        self.init_vars(S, dimK)
        self.xstep(S, self.lmbda, dimK)
        self.dstep()

        # Stop solve timer
        self.timer.stop('solve_wo_eval')

        # Extract and record iteration stats
        self.manage_itstat()

        # Increment iteration count
        self.j += 1

        # Stop solve timer
        self.timer.stop('solve')

        # Return current dictionary
        return self.getdict()



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
            self.Gf = sl.pyfftw_empty_aligned(self.Df.shape, self.Df.dtype)
            #self.Z = sl.pyfftw_empty_aligned(self.cri.shpX, self.dtype)
        #else:
        #    self.Df[:] = sl.fftn(self.D, self.cri.Nv, self.cri.axisN)



    def xstep(self, S, lmbda, dimK):
        """Solve CSC problem for training data `S`."""

        if self.opt['CUDA_CBPDN']:
            raise NotImplementedError()
        else:
            # Create X update object (external representation is expected!)
            Sf = sporco.linalg.fftn(S,s=self.cri.Nv,axes=self.cri.axisN).reshape(S.shape[0:self.dimN + 1]  + 2*(1,))
            if S.ndim == self.dimN + 1:
                self.Sf = Sf.reshape(S.shape[0:self.dimN + 1] + 2*(1,))
                S = S.reshape(S.shape[0:self.dimN + 1] + 2*(1,))
            elif S.ndim == self.dimN + 2:
                self.Sf = Sf.reshape(S.shape[0:self.dimN + 2] + (1,))
                S = S.reshape(S.shape[0:self.dimN + 2] + (1,))
            else:
                raise TypeError('Signal array must have Ndim + 1 or Ndim + 2 dimensions.')
            xstep = cbpdn_freq.CBPDN_ScaledDict(Ainv=self.Ainv, DR=self.Df, R=self.R, S=S, W=self.W, W1=self.W1, lmbda=lmbda, Ndim=self.dimN, opt=self.opt['CBPDN'])
            xstep.solve()
            self.Zf = xstep.getcoef()
            self.xstep_itstat = xstep.itstat[-1] if xstep.itstat else None



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

        # Compute X D - S
        Ryf = sl.inner(self.Zf, self.Gf, axis=self.cri.axisM) - self.Sf
        # Compute gradient
        gradf = sl.inner(numpy.conj(self.Zf), Ryf, axis=self.cri.axisK)

        # If multiple channel signal, single channel dictionary
        #if self.cri.C > 1 and self.cri.Cd == 1:
        #    gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        # Update gradient step
        self.eta = self.eta_a / (self.j + self.eta_b)
        
        gradfr = sl.ifftn(gradf, self.cri.dsz[0:-1], self.cri.axisN)

        self.Gprv = self.G

        self.G = self.Pcn(self.G - self.eta*gradfr)
        self.Gf = sl.fftn(self.G, self.cri.Nv,self.cri.axisN)
        (u,vH,dupdate) = sherman_morrison_python_functions.lowRankApprox(a=self.Gf -self.Df,projIter=self.projIter,axisu=self.cri.axisC,axisv=self.cri.axisM,dimN=self.dimN)
        for ii in range(0,2):
            dhu = sporco.linalg.inner(numpy.conj(self.Df),u[ii], axis=self.cri.axisC)
            uhu = sporco.linalg.inner(numpy.conj(u[ii]),u[ii],axis=self.cri.axisC)
            uhu = uhu.reshape(uhu.shape + (1,))
            dhu = numpy.reshape(dhu,dhu.shape[0:self.dimN] + (1,) + (1,) + (self.Df.shape[self.dimN + 2],) + (1,))
            vh = vH[ii].reshape(vH[ii].shape[0:self.dimN] + (1,) + (1,) + (self.Df.shape[self.dimN + 2],) + (1,))
            self.Ainv = sherman_morrison_python_functions.mdbi_dict_sm_r1update(self.Ainv, dhu, uhu, vh, self.dimN)
        self.Df = self.Df + dupdate
        self.R = sherman_morrison_python_functions.computeNorms(self.Df)

    def getdict(self):
        """Get final dictionary."""

        return sporco.linalg.ifftn(self.Df, self.cri.dsz[0:-1],self.cri.axisN) / self.R
    def iteration_stats(self):
        """Construct iteration stats record tuple."""

        tk = self.timer.elapsed(self.opt['IterTimer'])
        if self.xstep_itstat is None:
            objfn = (0.0,) * 3
            rsdl = (0.0,) * 2
            rho = (0.0,)
        else:
            objfn = (self.xstep_itstat.ObjFun, self.xstep_itstat.DFid,
                     self.xstep_itstat.RegL1)
            rsdl = (self.xstep_itstat.PrimalRsdl,
                    self.xstep_itstat.DualRsdl)
            rho = (self.xstep_itstat.Rho,)

        cnstr = numpy.linalg.norm(self.Df - self.Gf) / numpy.sqrt(numpy.prod(self.cri.Nv))
        dltd = numpy.linalg.norm(self.G - self.Gprv)

        tpl = (self.j,) + objfn + rsdl + rho + (cnstr, dltd, self.eta) + \
              self.itstat_extra() + (tk,)
        return type(self).IterationStats(*tpl)


    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return ()



