import sporco
import sporco.linalg
import sporco.admm
import sporco.admm.cbpdn
import sporco.prox
import numpy as np

class CBPDN_ScaledDict(sporco.admm.cbpdn.GenericConvBPDN):
    r"""

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The default fields of the named tuple
    ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``FVal`` :  Value of objective function component :math:`f`

       ``GVal`` : Value of objective function component :math:`g`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual Residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}` (see Sec. 3.3.1 of
       :cite:`boyd-2010-distributed`)

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}` (see Sec. 3.3.1 of
       :cite:`boyd-2010-distributed`)

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """




    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    itstat_fields_objfn = ('ObjFun', 'FVal', 'GVal')
    """Fields in IterationStats associated with the objective function;
    see :meth:`eval_objfn`"""
    itstat_fields_alg = ('PrimalRsdl', 'DualRsdl', 'EpsPrimal', 'EpsDual',
                         'Rho')
    """Fields in IterationStats associated with the specific solver
    algorithm"""
    itstat_fields_extra = ()
    """Non-standard fields in IterationStats; see :meth:`itstat_extra`"""

    hdrtxt_objfn = ('Fnc', 'f', 'g')
    """Display column headers associated with the objective function;
    see :meth:`eval_objfn`"""
    hdrval_objfun = {'Fnc': 'ObjFun', 'f': 'FVal', 'g': 'GVal'}
    """Dictionary mapping display column headers in :attr:`hdrtxt_objfn`
    to IterationStats entries"""




    def __init__(self, Ainv, DR, R, S, W, W1, lmbda, Ndim=2, dtype='complex128', opt=None):
        r"""
        Parameters
        ----------
        Nx : int
          Size of variable :math:`\mathbf{x}` in objective function
        yshape : tuple of ints
          Shape of working variable Y (the auxiliary variable)
        ushape : tuple of ints
          Shape of working variable U (the scaled dual variable)
        dtype : data-type
          Data type for working variables (overridden by 'DataType' option)
        opt : :class:`ADMM.Options` object
          Algorithm options
        """

        if opt is None:
            opt = CBPDN_ScaledDict.Options()
        if not isinstance(opt, CBPDN_ScaledDict.Options):
            raise TypeError('Parameter opt must be an instance of '
                            'CBPDN_ScaledDict.Options')

        self.opt = opt
        self.lmbda = lmbda
        self.M = Ainv.shape[-1]
        self.Ainv = Ainv
        self.R = R.reshape(Ndim*(1,) + (1,) + (1,) + (self.M,))
        self.DR = DR
        # This could be placed in a method
        if S.ndim == Ndim + 1:
            self.K = 1
            self.C = S.shape[-1]
            self.S = S.reshape(S.shape[0:Ndim] + (self.C,) + 2*(1,))
        elif S.ndim == Ndim + 2:
            self.K = S.shape[-1]
            self.C = S.shape[-2]
            self.S = S.reshape(S.shape[0:Ndim] + (self.C,) + (self.K,) + (1,))
        elif S.ndim == Ndim + 3:
            self.K = S.shape[-2]
            self.C = S.shape[-3]
            self.S = S
        else:
            raise TypeError('Signal array must have Ndim + 1 or Ndim + 2 dimensions.')

        self.W = W
        self.W1 = W1
        self.Maxis = Ndim + 2
        self.Kaxis = Ndim + 1
        self.Caxis = Ndim
        self.normCorrection = np.sqrt(np.prod(S.shape[0:Ndim]))
        self.Ndim = Ndim
        
        self.Nx = S.shape[0:Ndim] + (1,) + (self.K,) + (self.M,)
        yshape = S.shape[0:Ndim] + (1,) + (self.K,) + (self.M + self.C,)
        # Working variable U has the same dimensionality as constant c
        # in the constraint Ax + By = c
        ushape = yshape
        self.Nc = np.product(np.array(ushape))

        # DataType option overrides data type inferred from __init__
        # parameters of derived class
        self.set_dtype(opt, dtype)

        # Initialise attributes representing penalty parameter and other
        # parameters
        self.set_attr('rho', opt['rho'], dval=1.0, dtype=self.dtype)
        self.set_attr('rho_tau', opt['AutoRho', 'Scaling'], dval=2.0,
                      dtype=self.dtype)
        self.set_attr('rho_mu', opt['AutoRho', 'RsdlRatio'], dval=10.0,
                      dtype=self.dtype)
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=1.0,
                      dtype=self.dtype)
        self.set_attr('rlx', opt['RelaxParam'], dval=1.0, dtype=self.dtype)


        # Initialise working variable X
        if not hasattr(self, 'X'):
            self.X = None

        # Initialise working variable Y
        if self.opt['Y0'] is None:
            self.Y = self.yinit(yshape)
        else:
            self.Y = self.opt['Y0'].astype(self.dtype, copy=True)
        self.Yprev = self.Y.copy()
        self.Ys = self.ifft(self.Y)

        # Initialise working variable U
        if self.opt['U0'] is None:
            self.U = self.uinit(ushape)
        else:
            self.U = self.opt['U0'].astype(self.dtype, copy=True)

        self.itstat = []
        self.k = 0


    def getmin(self):
        """Get minimiser after optimisation."""

        return self.R * self.block_sep1(self.Y)

    def getcoef(self):
        return self.R * self.block_sep1(self.Y)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`.
        """

        dhypu = sporco.linalg.inner(np.conj(self.DR),(self.block_sep0(self.Y) + self.block_sep0(self.U)),self.Caxis)
        zpg = self.block_sep1(self.Y) + self.block_sep1(self.U)

        X = self.matMul(self.Ainv,dhypu + zpg)
        #X = sporco.linalg.inner(self.Ainv,np.swapaxes((dhypu + zpg).reshape(dhypu.shape + (1,)), self.Maxis, self.Maxis + 1))
        #X = np.swapaxes(X,self.Maxis + 1,self.Maxis).reshape(self.S.shape[0:self.Ndim] + (1,) + (self.K,) + (self.M,))

        # need method for fft and ifft
        self.X = self.fft(self.W1*self.ifft(X))
        


    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`.

        """
        # need method for ifft
        Dxmg = self.ifft(-self.nDX - self.block_sep0(self.U))
        self.Yprev = self.Y
        Y0S = np.logical_not(self.W)*Dxmg + self.W*(1/(1 + self.rho)*self.S + self.rho*Dxmg)
        #temp = -self.nX
        #temp = -self.nX - self.block_sep1(self.U)
        #temp = tuple(range(0,self.Ndim))
        #temp = self.lmbda*self.R/self.rho
        #temp = sporco.linalg.rifftn(-self.nX - self.block_sep1(self.U),axes=tuple(range(0,self.Ndim)))
        Y1S = sporco.prox.prox_l1(self.ifft(-self.nX - self.block_sep1(self.U)), self.lmbda*self.R/self.rho)
        self.Ys = self.block_cat(Y0S,Y1S)
        self.Y = self.fft(self.Ys)



    def ustep(self):
        """Dual variable update."""
        self.U += self.rho*self.rsdl_r(self.nDX, self.nX, self.Y)



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        # We need to keep the non-relaxed version of AX since it is
        # required for computation of primal residual r
        self.nDXnr = -sporco.linalg.inner(self.DR,self.X,self.Maxis)
        if self.rlx == 1.0:
            # If RelaxParam option is 1.0 there is no relaxation
            self.nDX = self.nDXnr
            self.nX = -self.X
        else:
            # Avoid calling cnst_c() more than once in case it is expensive
            # (e.g. due to allocation of a large block of memory)
            #if not hasattr(self, '_cnst_c'):
            #    self._cnst_c = self.cnst_c()
            # Compute relaxed version of AX
            alpha = self.rlx
            self.nDX = alpha*self.nDXnr - (1 - alpha)*(self.block_sep0(self.Y))
            self.nX = -alpha*self.X - (1 - alpha)*(self.block_sep1(self.Y))



    def compute_residuals(self):
        """Compute residuals and stopping thresholds."""
        
        # what is different about this vs. the original method?
        if self.opt['AutoRho', 'StdResiduals']:
            r = np.linalg.norm(self.rsdl_r(self.nDXnr,-self.X, self.Y)[:])/self.normCorrection
            s = np.linalg.norm(self.rsdl_s(self.Yprev, self.Y)[:])/self.normCorrection
            epri = np.sqrt(self.Nc) * self.opt['AbsStopTol'] + \
                self.rsdl_rn(self.nDXnr,-self.X, self.Y) * self.opt['RelStopTol']
            edua = np.sqrt(self.Nx) * self.opt['AbsStopTol'] + \
                self.rsdl_sn(self.U) * self.opt['RelStopTol']
        else:
            rn = self.rsdl_rn(self.nDXnr,-self.X, self.Y)
            if rn == 0.0:
                rn = 1.0
            sn = self.rsdl_sn(self.U)
            if sn == 0.0:
                sn = 1.0
            r = np.linalg.norm(self.rsdl_r(self.nDXnr,-self.X, self.Y)[:]) / rn / self.normCorrection
            s = np.linalg.norm(self.rsdl_s(self.Yprev, self.Y)[:]) / sn / self.normCorrection
            epri = np.sqrt(self.Nc) * self.opt['AbsStopTol'] / rn + \
                self.opt['RelStopTol']
            edua = np.sqrt(self.Nx) * self.opt['AbsStopTol'] / sn + \
                self.opt['RelStopTol']

        return r, s, epri, edua

    def block_sep0(self,Y):
        return np.swapaxes(Y[(slice(None),)*(self.Maxis) + (slice(0, self.C),)],self.Caxis,self.Maxis)

    def block_sep1(self,Y):
        return Y[(slice(None),)*(self.Maxis) + (slice(self.C,self.C + self.M),)]

    def block_cat(self,Y0,Y1):
        return np.concatenate((np.swapaxes(Y0, self.Caxis, self.Maxis), Y1), axis=self.Maxis)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return ()



    def var_x(self):
        r"""Get :math:`\mathbf{x}` variable."""

        return self.R*self.X



    def var_y(self):
        r"""Get :math:`\mathbf{y}` variable."""

        return self.block_sep0(self.Y)

    def var_z(self):
        r"""Get :math:'mathbf{z}' variable."""

        return self.R*self.block_sep1(self.Y)



    def var_u(self):
        r"""Get :math:`\mathbf{u}` variable."""

        return self.U



#    def obfn_f(self, X):
#        r"""Compute :math:`f(\mathbf{x})` component of ADMM objective function.
#
#        """
#
#        return 0
    def obfn_dfd(self):
        return (1/(2*self.normCorrection**2))*np.linalg.norm(self.block_sep0(self.Y)[:] - self.S)**2

    def obfn_reg(self):
        return (self.lmbda*np.linalg.norm(x=sporco.linalg.ifftn(self.block_sep1(self.Y)).reshape((-1,1)),ord=1),)



#    def obfn_g(self, Y):
#        r"""Compute :math:`g(\mathbf{y})` component of ADMM objective function.
#
#        """

#        return (1/(2*self.normCorrection**2))*np.linalg.norm(self.block_sep0(Y)[:])**2 + self.lmbda*np.linalg.norm(sporco.linalg.rifft(self.block_sep1(Y))[:],1)



    def cnst_A(self, X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def cnst_AT(self, X):
        r"""Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """
        # I removed the scaling of the second term by 1/R because I think it was there in error.
        return -sporco.linalg.inner(np.conj(self.DR),self.block_sep0(X),self.Caxis) - self.block_sep1(X)



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def rsdl_r(self, nDX, nX, Y):
        """Compute primal residual vector.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        #print(np.linalg.norm((self.block_cat(nDX,nX) + Y)[:]))
        return self.block_cat(nDX,nX) + Y



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        It is likely this method should include an R factor for block_sep1(Y - Yprev).

        I was correct on this in my earlier thoughts. block_sep1(Y - Yprev) should be rescaled by 1/R.

        Intuitively, one might expect it necessary to scale the result by R in a similar way to the entire equation being scaled by rho. Intuition fails us here, however. If we evaluate the gradients, we can clearly see such an alteration would not be correct.

        I made this adjustment in cnst_AT. The only other function call this changes is rsdl_sn, and I suspect this adjustment is correct there too.
        """

        return self.rho * self.cnst_AT(Y - Yprev)



    def rsdl_rn(self, nDX, nX, Y):
        """Compute primal residual normalisation term.
        """

        return max((np.linalg.norm(self.block_cat(nDX,nX)[:]), np.linalg.norm(Y[:])))/self.normCorrection



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term.
        This residual may not be correct. I'm not sure.
        """
        #print(np.linalg.norm(self.cnst_AT(U)[:]))
        return self.rho * np.linalg.norm(self.cnst_AT(U)[:])/self.normCorrection



    def rhochange(self):
        """Action to be taken, if any, when rho parameter is changed.

        Overriding this method is optional.
        """

        pass

    def reconstruct(self):
        DX = sporco.linalg.inner(self.DR,self.X,self.Maxis)
        # need ifft mehod
        return self.ifft(X)




    def solve(self):
        """Start (or re-start) optimisation. This method implements the
        framework for the iterations of an ADMM algorithm. There is
        sufficient flexibility in overriding the component methods that
        it calls that it is usually not necessary to override this method
        in derived clases.

        If option ``Verbose`` is ``True``, the progress of the
        optimisation is displayed at every iteration. At termination
        of this method, attribute :attr:`itstat` is a list of tuples
        representing statistics of each iteration, unless option
        ``FastSolve`` is ``True`` and option ``Verbose`` is ``False``.

        Attribute :attr:`timer` is an instance of :class:`.util.Timer`
        that provides the following labelled timers:

          ``init``: Time taken for object initialisation by
          :meth:`__init__`

          ``solve``: Total time taken by call(s) to :meth:`solve`

          ``solve_wo_func``: Total time taken by call(s) to
          :meth:`solve`, excluding time taken to compute functional
          value and related iteration statistics

          ``solve_wo_rsdl`` : Total time taken by call(s) to
          :meth:`solve`, excluding time taken to compute functional
          value and related iteration statistics as well as time take
          to compute residuals and implemented ``AutoRho`` mechanism
        """

        # Open status display
        #fmtstr, nsep = self.display_start()

        # Start solve timer
        #self.timer.start(['solve', 'solve_wo_func', 'solve_wo_rsdl'])
        temp = 0
        # Main optimisation iterations
        for self.k in range(self.k, self.k + self.opt['MaxMainIter']):

            # Update record of Y from previous iteration
            self.Yprev = self.Y.copy()
            print('Lagrangian x-terms before update')
            if temp==1:
                print(np.linalg.norm(self.rsdl_r(self.nDX, self.nX, self.Y) + self.U)) # added + U
            temp=1

            # X update
            self.xstep()



            # Implement relaxation if RelaxParam != 1.0
            self.relax_AX()
            # all these residuals should be in methods.
            print('Lagrangian x-terms after update.')
            print(np.linalg.norm(self.rsdl_r(self.nDX, self.nX, self.Y) + self.U)) # added + U

            # Y update
            print('Lagrangian y-terms before update')
            print(np.linalg.norm(self.W*(self.S -self.block_sep0(self.Ys)))**2/2 + self.rho*np.linalg.norm(self.block_sep0(self.Y) + self.nDX + self.block_sep0(self.U))**2/2)
            self.ystep()
            print('Langrangian y-terms after update')
            print(np.linalg.norm(self.W*(self.S -self.block_sep0(self.Ys)))**2/2 + self.rho*np.linalg.norm(self.block_sep0(self.Y) + self.nDX + self.block_sep0(self.U))**2/2)
            # U update
            self.ustep()

            # Compute residuals and stopping thresholds
            #self.timer.stop('solve_wo_rsdl')
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                r, s, epri, edua = self.compute_residuals()
            #self.timer.start('solve_wo_rsdl')

            # Compute and record other iteration statistics and
            # display iteration stats if Verbose option enabled
            #self.timer.stop(['solve_wo_func', 'solve_wo_rsdl'])
            #if not self.opt['FastSolve']:
            #    itst = self.iteration_stats(self.k, r, s, epri, edua)
            #    self.itstat.append(itst)
            #    self.display_status(fmtstr, itst)
            #self.timer.start(['solve_wo_func', 'solve_wo_rsdl'])

            # Automatic rho adjustment
            #self.timer.stop('solve_wo_rsdl')
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                self.update_rho(self.k, r, s)
            #self.timer.start('solve_wo_rsdl')

            # Call callback function if defined
            if self.opt['Callback'] is not None:
                if self.opt['Callback'](self):
                    break

            # Stop if residual-based stopping tolerances reached
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                if r < epri and s < edua:
                    break


        # Increment iteration count
        self.k += 1

        # Record solve time
        #self.timer.stop(['solve', 'solve_wo_func', 'solve_wo_rsdl'])

        # Print final separator string if Verbose option enabled
        #self.display_end(nsep)

        return self.getmin()

    def matMul(self,A,x):
        temp = sporco.linalg.inner(A,np.swapaxes(x.reshape(x.shape + (1,)), -1, -2),-1)
        return temp.reshape(x.shape[0:-1] + (A.shape[-2],))

    def fft(self,x):
        return sporco.linalg.fftn(x,self.Nx[0:self.Ndim],tuple(range(0,self.Ndim)))

    def ifft(self,xf):
        return sporco.linalg.ifftn(xf,self.Nx[0:self.Ndim],tuple(range(0,self.Ndim)))
        

