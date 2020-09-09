import sporco
import sporco.linalg
import sporco.admm.admm
import sporco.admm.cbpdn
import sporco.cnvrep
import sporco.prox
import sporco.prox
import numpy as np
import copy
import sherman_morrison_python_functions as sm

class CBPDN_Factored(sporco.admm.admm.ADMM):
    class Options(sporco.admm.admm.ADMM.Options):
        r"""
        """
        defaults = copy.deepcopy(sporco.admm.admm.ADMM.Options.defaults)
        defaults.update({'AuxVarObj': False, 'ReturnX': False,
                         'RelaxParam': 1.8})
        defaults['AutoRho'].update({'Enabled': False, 'Period': 1,
                                    'AutoScaling': True, 'Scaling': 1000.0,
                                    'RsdlRatio': 1.2})


        def __init__(self, opt=None):
            """
          #  Parameters
          #  ----------
          #  opt : dict or None, optional (default None)
          #    GenericConvBPDN algorithm options
          #  """

            if opt is None:
                opt = {}
            sporco.admm.admm.ADMM.Options.__init__(self, opt)


    itstat_fields_objfn = ('ObjFun', 'DFid', 'Reg')
    #itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'DFid', 'Reg')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', 'Reg': 'Reg'}


    def __init__(self,Q,DR,S,R,W,lmbda,dimN=2,opt=None):
        dimK = None
        self.cri = sporco.cnvrep.CSC_ConvRepIndexing(DR, S, dimK=dimK, dimN=dimN)

        Nx = np.prod(self.cri.shpX)
        yshape = self.cri.Nv + (1,) + (self.cri.K,) + (self.cri.M + self.cri.C,)
        ushape = yshape
        self.DR = np.asarray(DR.reshape(self.cri.shpD))
        self.S = np.asarray(S.reshape(self.cri.shpS))
        super(CBPDN_Factored, self).__init__(Nx, yshape, ushape, DR.dtype, opt)
        self.Q = Q
        self.R = R
        self.W = W
        self.lmbda = lmbda
        self.Sf = fft(S)

        # for testing purposes:
        #self.X = np.zeros(self.cri.Nv + (self.cri.C,self.cri.K,1,))


    def getmin(self):
        return self.X if self.opt['ReturnX'] else self.block_sep1(self.Y)

    def getcoef(self):
        return self.R*self.getmin()

    def var_x(self):
        return self.R*self.X

    def var_y(self):
        return self.block_sep0(self.Y)

    def var_z(self):
        return self.R*self.block_sep1(self.Y)

    def var_eta(self):
        return self.block_sep0(self.U)

    def var_gamma(self):
        return self.block_sep1(self.U)

    #def relax_AX(self):
    #    """Implement relaxation if option ``RelaxParam`` != 1.0.
    #    This code was copied verbatim from ADMM in SPORCO. This is necessary because ADMMEqual overwrites it."""

        # We need to keep the non-relaxed version of AX since it is
        # required for computation of primal residual r
 #       self.AXnr = self.cnst_A(self.X)
  #      if self.rlx == 1.0:
   #         # If RelaxParam option is 1.0 there is no relaxation
    #        self.AX = self.AXnr
     #   else:
      #      # Avoid calling cnst_c() more than once
       #     # (e.g. due to allocation of a large block of memory)
        #    if not hasattr(self, '_cnst_c'):
         #       self._cnst_c = self.cnst_c()
          #  # Compute relaxed version of AX
           # alpha = self.rlx
            #self.AX = alpha*self.AXnr - (1 - alpha)*(self.cnst_B(self.Y) -
             #                                        self._cnst_c)

    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`.
        """

        dhy = sporco.linalg.inner(np.conj(self.DR),self.block_sep0(self.Y) - self.Sf + self.block_sep0(self.U)/self.rho,self.cri.axisC)
        zpg = self.block_sep1(self.Y) + self.block_sep1(self.U)/self.rho
        #print((dhypu + zpg).shape)
        self.X = self.Q.inv_vec(zpg - dhy,sm.DfMatRep(self.DR,self.cri.axisC,self.cri.axisM))

    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`.
        self.AX should be used where the constraints affect the ystep, but not in cases where x appears in the objective. Currently, this ystep needs to be fixed. The overrelaxed AX can be nonzero where nonrelaxed AX is zero. This will affect Y0S. I will need to rederive the equations to find the correct solution here.
        """
        
        #idftDX = self.ifft(sporco.linalg.inner(self.DR,self.X,self.cri.axisM))
        idftU = self.ifft(self.block_sep0(self.U))
        idftAX = self.ifft(self.block_sep0(self.AX))
        Y0S = self.rho/(self.rho + self.W)*(idftAX - self.S + idftU/self.rho)
        idftnAXmU = -self.ifft(self.block_sep1(self.AX) + self.block_sep1(self.U)/self.rho)
        Y1S = self.W1*sporco.prox.prox_l1(idftnAXmU, self.lmbda*self.R/self.rho)

        self.Ys = self.block_cat(Y0S,Y1S)
        self.Y = self.fft(self.Ys)
        

    def cnst_A(self,X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """        
        return self.block_cat(sporco.linalg.inner(self.DR,X,self.cri.axisM),-X)

    def cnst_AT(self,X):
        return sporco.linalg.inner(self.DR,self.block_sep0(X),self.cri.axisM) - self.block_sep1(X)

    def cnst_B(self,X):
        return X

    def cnst_c(self):
        return self.block_cat(-self.Sf,np.zeros(self.X.shape))

    def reconstruct(self):
        return self.ifft(sporco.linalg.inner(self.DR, self.X, axis=self.cri.axisM)) if self.opt['ReturnX'] else \
            self.ifft(sporco.linalg.inner(self.DR, self.block_sep1(self.Y), axis=self.cri.axisM))
        pass

    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on ``fEvalX`` option value.
        """

        return self.block_sep1(self.Y) if self.opt['AuxVarObj'] else \
            self.X


    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m \mathbf{d}_m *
        \mathbf{x}_m - \mathbf{s} \|_2^2`.
        """

        Ef = self.W*(self.ifft(sporco.linalg.inner(self.DR, self.obfn_fvarf(), axis=self.cri.axisM)) - \
            self.S)
        return np.linalg.norm(Ef)**2 / 2.0



    def obfn_reg(self):
        """Compute regularisation term(s) and contribution to objective
        function.
        """
        l1_penalty = self.lmbda*np.sum(np.abs(self.ifft(self.obfn_fvarf())))
        return (l1_penalty,l1_penalty)

    def fft(self,x):
        return sporco.linalg.fftn(x,self.cri.Nv,self.cri.axisN)

    def ifft(self,xf):
        return sporco.linalg.ifftn(xf,self.cri.Nv,self.cri.axisN)

    def block_sep0(self,Y):
        return np.swapaxes(Y[(slice(None),)*(self.cri.axisM) + (slice(0, self.cri.C),)],self.cri.axisC,self.cri.axisM)

    def block_sep1(self,Y):
        return Y[(slice(None),)*(self.cri.axisM) + (slice(self.cri.C,self.cri.C + self.cri.M),)]

    def block_cat(self,Y0,Y1):
        return np.concatenate((np.swapaxes(Y0, self.cri.axisC, self.cri.axisM), Y1), axis=self.cri.axisM)

#    def data_fid_term(self):
#        return 1/2*np.linalg.norm(self.W*self.S - self.W*self.ifft(self.block_sep0(self.Y)))**2

#    def sparsity_term(self):
#        return self.lmbda*sporco.prox.norm_l1(x=self.ifft(self.R*self.block_sep1(self.Y)))

#    def zx_constraint(self,X):
#        return self.rho/2*(np.linalg.norm(self.block_sep1(self.Y) - X + self.block_sep1(self.U)/self.rho)/np.sqrt(np.prod(self.cri.Nv)))**2

#    def ydx_constraint(self,DX):
#        return self.rho/2*(np.linalg.norm(self.block_sep0(self.Y) - DX + self.block_sep0(self.U)/self.rho)/np.sqrt(np.prod(self.cri.Nv)))**2

