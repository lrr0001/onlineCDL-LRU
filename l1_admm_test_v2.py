import sporco
import sporco.linalg
import sporco.admm.admm
import sporco.admm.cbpdn
import sporco.cnvrep
import sporco.prox
import numpy
import copy

class constraint_test(sporco.admm.admm.ADMM):
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


    def __init__(self,a,b,lmbda,opt):
        self.a = a
        self.b = b
        self.lmbda = lmbda
        Nx = 1
        yshape = (2,1)
        ushape = yshape
        super(constraint_test, self).__init__(Nx, yshape, ushape, numpy.dtype('complex128'), opt)


    def getmin(self):
        return self.X if self.opt['ReturnX'] else self.Y[0,0]

    def getcoef(self):
        return self.getmin()

    def var_x(self):
        return self.X

    def var_y(self):
        return self.Y[1,0]

    def var_z(self):
        return self.Y[0,0]

    def var_eta(self):
        return self.U[1,0]

    def var_gamma(self):
        return self.U[0,0]

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
        self.X = self.cnst_AT(self.cnst_B(self.Y) - self.cnst_c() + self.U)/(1 + self.a**2)
 
        
    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`.
        self.AX should be used where the constraints affect the ystep, but not in cases where x appears in the objective.
        """
        
        self.Y =  sporco.prox.prox_l1(-self.AX + self.cnst_c() - self.U, numpy.array([[1/self.rho],[self.lmbda/self.rho]]))
        

    def cnst_A(self,X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """    
        #import pdb; pdb.set_trace()
        #print(X)    
        return numpy.array([[-X],[-self.a*X]])

    def cnst_AT(self,X):
        #import pdb; pdb.set_trace()
        #print(X)


        return  -X[0,0] - self.a*X[1,0]

    def cnst_B(self,X):
        return X

    def cnst_c(self):
        return numpy.array([[0],[-self.b]])

    def reconstruct(self):
        return self.a*self.getmin() - self.b

    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on ``fEvalX`` option value.
        """

        return self.X if self.opt['AuxVarObj'] else \
            self.Y[0,0]


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

        Ef = self.reconstruct()
        return numpy.sum(numpy.abs(Ef))



    def obfn_reg(self):
        """Compute regularisation term(s) and contribution to objective
        function.
        """
        l1_penalty = self.lmbda*numpy.sum(numpy.abs(self.obfn_fvarf()))
        return (l1_penalty,l1_penalty)


