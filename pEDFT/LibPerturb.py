import psi4
import numpy as np
import scipy.linalg as la

from LibDegen import *

eV = 27.211

np.set_printoptions(precision=4, suppress=False, floatmode="fixed")

##### This is a hack to convert a UKS superfunctional to its RKS equivalent
# Internal routine
# https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/dft/dft_builder.py#L251
sf_from_dict =  psi4.driver.dft.build_superfunctional_from_dictionary
# My very hacky mask
def sf_RKS_to_UKS(DFA):
    #print(DFA.name())
    DFA_Dict = { 'name':DFA.name()+'_u'}
    DFA_Dict['x_functionals']={}
    DFA_Dict['c_functionals']={}
    for x in DFA.x_functionals():
        Name = x.name()[3:]
        alpha = x.alpha()
        DFA_Dict['x_functionals'][Name] = {"alpha": alpha,}
    for c in DFA.c_functionals():
        Name = c.name()[3:]
        alpha = c.alpha()
        DFA_Dict['c_functionals'][Name] = {"alpha": alpha,}
    #print(DFA_Dict)
    npoints = psi4.core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
    DFAU, _ = sf_from_dict(DFA_Dict,npoints,1,False)
    return DFAU
##### End hack

# For nice debug printing
def NiceArr(X):
    return "[ %s ]"%(",".join(["%8.3f"%(x) for x in X]))
def NiceArrInt(X):
    return "[ %s ]"%(",".join(["%5d"%(x) for x in X]))
def NiceMat(X):
    N = X.shape[0]
    if N==0:
        return "[]"
    elif N==1:
        return "["+NiceArr(X[0,:])+"]"
    elif N==2:
        return "["+NiceArr(X[0,:])+",\n "+NiceArr(X[1,:])+"]"
    else:
        R = "["
        for K in range(N-1):
            R+=NiceArr(X[K,:])+",\n "
        R+=NiceArr(X[N-1,:])+"]"
        return R

# Handle PBE_XX calculations
def GetDFA(DFA):
    if DFA[:5].lower()=="pbe0_":
        X = DFA.split('_')
        alpha = float(X[1])/100.
        if len(X)>2: f_c = max(float(X[2])/100.,1e-5)
        else: f_c = 1.
        return {
            'name':DFA,
            'x_functionals': {"GGA_X_PBE": {"alpha":1.-alpha, }},
            'c_functionals': {"GGA_C_PBE": {"alpha":f_c, }},
            'x_hf': {"alpha":alpha, },
            }
    else:
        return DFA

# Get the degeneracy of each orbital
def GetDegen(epsilon, eta=1e-5):
    Degen = np.zeros((len(epsilon),),dtype=int)
    for k in range(len(epsilon)):
        ii =  np.argwhere(np.abs(epsilon-epsilon[k])<eta).reshape((-1,))
        Degen[k] = len(ii)
    return Degen
        

class ExcitationHelper:
    def __init__(self, wfn, RKS=True, wfnSym=None):
        self.wfn = wfn
        self.RKS = RKS

        if wfnSym is None:
            wfnSym = wfn
            UseDegen = False
        else:
            UseDegen = True
            
        self.Da= wfn.Da().to_array(dense=True)
        self.epsilon = wfn.epsilon_a().to_array(dense=True)
        self.C = wfn.Ca().to_array(dense=True)
        self.F = wfn.Fa().to_array(dense=True)

        basis = wfn.basisset()
        self.basis = basis
        self.nbf = self.wfn.nmo() # Number of basis functions

        
        self.mints = psi4.core.MintsHelper(self.basis)
        self.S_ao = self.mints.ao_overlap().to_array(dense=True)
        self.T_ao = self.mints.ao_kinetic().to_array(dense=True)
        self.V_ao = self.mints.ao_potential().to_array(dense=True)
        self.H_ao = self.T_ao + self.V_ao

        self.UseDegen = UseDegen
        # Call degeneracy helper if we need to use more complex degeneracies
        if self.UseDegen:
            self.DegenHelp = psi4DegenHelper(wfn, wfnSym)
            self.Symh = self.DegenHelp.SymH[0]
            self.Syml = self.DegenHelp.SymL[0]
        else:
            self.Symh = 0
            self.Syml = 0            
            
        aux_basis = psi4.core.BasisSet.build\
                    (self.wfn.molecule(), "DF_BASIS_SCF", "",
                     "RIFIT", self.basis.name())
        zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
        self.aux_basis, self.zero_basis = aux_basis, zero_basis
        self.SAB = np.squeeze(self.mints.ao_eri(aux_basis, zero_basis, basis, basis))
        metric = self.mints.ao_eri(aux_basis, zero_basis, aux_basis, zero_basis)
        metric.power(-0.5, 1e-14)
        metric = np.squeeze(metric)
        self.ERIA = np.tensordot(metric, self.SAB, axes=[(1,),(0,)])
            
        self.NBas = wfn.nmo()
        self.NOcc = wfn.nalpha()
        self.kh = wfn.nalpha()-1
        self.kl = self.kh+1
        self.Degen = GetDegen(self.epsilon)

        self.kFrom = self.kh
        self.kTo = self.kl
        
        print("eps = %s eV"%(NiceArr(self.epsilon[max(0,self.kh-2):min(self.NBas,self.kl+3)]*eV)))
        print("eps_h = %8.3f/%8.2f, eps_l = %8.3f/%8.2f [Ha/eV]"\
              %(self.epsilon[self.kh], self.epsilon[self.kh]*eV,
                self.epsilon[self.kl], self.epsilon[self.kl]*eV,))

        self.FHF = self.H_ao*1.
        for I in range(self.kh+1):
            CI = self.C[:,I] 
            self.FHF += 2.*self.GetFJ(CI) - self.GetFK(CI)

        # These are used for DFA calculations
        self.VPot = wfn.V_potential() # Note, this is a VBase class
        try:
            self.DFA = self.VPot.functional()
        except:
            self.DFA = None
            self.VPot = None

        # Ensure we have UKS for the excitations
        if not(self.DFA is None) and \
           not(self.RKS):
            # Convert DFA from RKS to UKS
            self.DFAU = sf_RKS_to_UKS(self.DFA)
            # Make a new VPot for the UKS DFA
            self.VPot = psi4.core.VBase.build(self.VPot.basis(), self.DFAU, "UV")
            self.VPot.initialize()

        self.xDFA = 0.
        self.omega = 0.
        self.beta = 0.
        if not(self.DFA is None):
            if self.DFA.is_x_hybrid():
                self.xDFA = 1. - self.DFA.x_alpha()
                # Not implemented yet
                if self.DFA.is_x_lrc():
                    self.omega = self.DFA.x_omega()
                    self.beta = self.DFA.x_beta()
                if not self.omega == 0.:
                    print("Range-separated hybrids not implemented yet")
                    quit()
            else:
                self.xDFA = 1.

            # psi4 matrices for DFA evaluation
            self.DMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.DMb = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMb = psi4.core.Matrix(self.nbf, self.nbf)
        else:
            self.xDFA = 0.

        print("# xDFA = %.2f, beta = %.2f, omega = %.2f"\
              %(self.xDFA, self.beta, self.omega))

    # Choose the orbital to promote from using degeneracy and shift (=0 for highest)
    def SetFrom(self, Degen=0, Shift=0):
        if Degen<1:
            self.kFrom=self.kh-Shift
            return self.kFrom
        
        Count = 0
        self.kFrom = -1
        for k in range(self.kh,-1,-1):
            if self.Degen[k]==Degen:
                if Count==Shift:
                    self.kFrom=k
                    break
                Count+=1
                
        if self.kFrom<0:
            print("# There is no orbital compatible with Degen=%d and Shift=%d"%(Degen, Shift))
            quit()

        if Degen>1:
            self.kFromAll = np.argwhere(np.abs(
                self.epsilon-self.epsilon[self.kFrom])<1e-6).reshape((-1,))
            
        return self.kFrom
    
    # Choose the orbital to promote to using degeneracy and shift (=0 for lowest)
    def SetTo(self, Degen=0, Shift=0):
        if Degen<1:
            self.kTo=self.kl+Shift
            return self.kTo
        
        Count = 0
        self.kTo = -1
        for k in range(self.kl,self.NBas):
            if self.Degen[k]==Degen:
                if Count==Shift:
                    self.kTo=k
                    break
                Count+=1
        if self.kTo<0:
            print("# There is no orbital compatible with Degen=%d and Shift=%d"%(Degen, Shift))
            quit()

        return self.kTo

    def GetEJ(self, D):
        A = np.tensordot(self.ERIA,D,axes=((1,2),(1,0)))
        return 0.5*np.dot(A,A)

    def GetEK(self, D):
        B = np.tensordot(self.ERIA,D,axes=((1,),(0,)))
        B = np.tensordot(self.ERIA,B,axes=((0,1),(0,2)))
        return 0.5*np.tensordot(D,B)

    def GetFJ(self, CI):
        A = np.tensordot(self.ERIA,CI,axes=((2,),(0,)))
        A = np.tensordot(A,CI,axes=((1,),(0,)))
        return np.tensordot(A,self.ERIA,axes=((0,),(0,)))

    def GetFK(self, CI):
        A = np.tensordot(self.ERIA,CI,axes=((2,),(0,)))
        return np.tensordot(A,A,axes=((0,),(0,)))


    def GetFDFA(self, C0, C1, Return="DV"):
        Dgs = self.Da
        # Singlet gs
        if not(self.RKS):
            self.DMa.np[:,:] = Dgs
            self.DMb.np[:,:] = Dgs
            self.VPot.set_D([self.DMa,self.DMb])
            self.VPot.compute_V([self.VMa,self.VMb])
            Egs = self.VPot.quadrature_values()["FUNCTIONAL"]
            Vgs = self.VMa.to_array(dense=True)
            self.DMa.np[:,:] = Dgs + np.outer(C1,C1)
            self.DMb.np[:,:] = Dgs - np.outer(C0,C0)
            self.VPot.set_D([self.DMa,self.DMb])
            self.VPot.compute_V([self.VMa,self.VMb])
            Ets = self.VPot.quadrature_values()["FUNCTIONAL"]
            Vts = self.VMa.to_array(dense=True)
        else:
            self.DMa.np[:,:] = Dgs - np.outer(C0,C0)
            self.VPot.set_D([self.DMa,])
            self.VPot.compute_V([self.VMa,])
            Egs = self.VPot.quadrature_values()["FUNCTIONAL"]
            Vgs = self.VMa.to_array(dense=True)
            self.DMa.np[:,:] = Dgs + np.outer(C1,C1)
            self.VPot.set_D([self.DMa,])
            self.VPot.compute_V([self.VMa,])
            Ets = self.VPot.quadrature_values()["FUNCTIONAL"]
            Vts = self.VMa.to_array(dense=True)

        if Return.upper()=="E":
            return Egs, Ets
        elif Return.upper()=="V":
            return Vgs, Vts
        else:
            return Vts-Vgs

    def GuessLowestTriple(self, Range=3):
        MinGap = 10000
        for kFrom in range(self.kh, max(0,self.kh-Range)-1, -1):
            for kTo in range(self.kl, min(self.NBas, self.kl+Range)):
                Gap = self.SolveTriple(k0=kFrom, k1=kTo, Silent=True, MaxStep=0)
                #print(kTo, kFrom, Gap, MinGap)
                if (Gap<MinGap):
                    MinGap = Gap
                    self.kFrom = kFrom
                    self.kTo = kTo
        print("# Lowest gap from %d to %d"%(self.kFrom, self.kTo))
        return self.kFrom, self.kTo

    def GuessLowestSingle(self, Range=3):
        MinGap = 10000
        for kFrom in range(self.kh, max(0,self.kh-Range)-1, -1):
            for kTo in range(self.kl, min(self.kl+Range, self.NBas)):
                Gap = self.SolveSingle(k0=kFrom, k1=kTo, Silent=True, MaxStep=0)
                #print(kTo, kFrom, Gap, MinGap)
                if (Gap<MinGap):
                    MinGap = Gap
                    self.kFrom = kFrom
                    self.kTo = kTo
        print("# Lowest gap from %d to %d"%(self.kFrom, self.kTo))
        return self.kFrom, self.kTo

    def SolveTriple(self, *args, **kwargs):
        return self.SolveGeneral(J1 = [-1, 0], K1 = [ 0, 0],
                                 **kwargs)
    def SolveSingle(self, *args, **kwargs):
        return self.SolveGeneral(J1 = [-1, 0], K1 = [ 2, 0],
                                 **kwargs)
    def SolveDouble(self, *args, **kwargs):
        return self.SolveGeneral(J1 = [-2, 2], K1 = [ 1,-1],
                                 **kwargs)

    def SolveGeneral(self, k0=None, k1=None,
                     J1 = [-1, 0], K1 = [ 2, 0],
                     UseHF = False,
                     GapFactor = 1.,
                     ErrCut = 1e-3, MaxStep=50,
                     MOM = True,
                     Report = False,
                     Silent = False,
                     Debug = False, # This should be set to false normally
    ):
        # Handle the new From and To
        if not(k0 is None) or not(k1 is None):
            if not(Silent):
                print("# The direct use of k0 and k1 is not recommended")
                print("# - use SetFrom and SetTo before calling for excitations")
            if not(k0 is None): k0=k0
            else: k0=self.kFrom
            if not(k1 is None): k1=k1
            else: k1=self.kTo
            Nk0 = 1
        else:
            k0 = self.kFrom
            k1 = self.kTo
            if not(Silent):
                if self.UseDegen:
                    print("# From %3d [Sym%d] to %3d [Sym%d]"\
                          %(k0, self.DegenHelp.SymOrb(k0),
                            k1, self.DegenHelp.SymOrb(k1)))
                else:
                    print("# From %3d to %3d"%(k0, k1))

            # Setting Nk0>1 will trigger a more complicated method that should be
            # strictly symmetry - but it doesn't appear to be necessary
            Nk0 = 1 # self.Degen[k0]
            if Nk0>1:
                k0All = self.kFromAll
            
        dk1 = k1-self.kl

        if Debug:
            C_In = self.C*1.

        if Nk0 == 1:
            C0All = [self.C[:,k0]]
        else:
            C0All = []
            for k in k0All:
                C0All += [self.C[:,k]]
            
        C1 = self.C[:,k1]
        CX = self.C[:,self.kl:]

        epsilon_old = self.epsilon*1.
        epsilon_new = 1.*epsilon_old

        #####################################################################
        # *** NEW HERE ***
        # Correct for DFA exchange
        # Note, the exchange acts only on the Fock term for l-l interactions
        # and is negative
        # But we _subtract_ the Fock in any DFA/hybrid so we have a positive
        # correction
        if not(UseHF or self.DFA is None):
            K1[1]+=self.xDFA # This is a -xDFA times the -1 term
        #####################################################################

        # Do a perturbative correction
        if not(Silent):
            print("# HF correction = %.2f v_{J,h} + %.2f v_{J,l} + %.2f v_{K,h} + %.2f v_{K,l}"\
                  %(J1[0],J1[1],K1[0],K1[1]))

        DF1 = 0.
        for C0 in C0All:
            # Convert to operator
            DF1 += J1[0]*self.GetFJ(C0) + K1[0]*self.GetFK(C0) \
                  + (J1[1]+K1[1])*self.GetFJ(C1)
            # Add DFA if appropriate
            if not(UseHF or self.DFA is None):
                DF1 += self.GetFDFA(C0,C1)
        DF1 /= len(C0All)
            
        # Project onto C1
        DEP = C1.dot(DF1).dot(C1)
        self.EGapPerturb = GapFactor* (self.epsilon[k1]-self.epsilon[k0]+DEP)
        if not(Silent):
            print("# DFA initial correction = %8.3f Ha = %8.2f eV"\
                  %(DEP, DEP*eV))
            print("# Perturbative gap = %8.2f eV"%(self.EGapPerturb*eV))
        if MaxStep==0:
            self.CE = 1.*self.C
            self.epsilonE = self.epsilon*1.
            self.epsilonE[self.kl:]+=DEP
            return self.EGapPerturb
        
        # Introduce some smoothing
        FOld = self.F * 1.
        FOld2 = self.F * 1.

        Err = 1e3 # Make sure large error if no iterations
        Mix = 0.3+0.3*self.xDFA
        Mix2 = 0.3*self.xDFA
        #Mix, Mix2 = 0.1,0.
        eps1Old = 0.
        for step in range(MaxStep): # If it's not done in MaxStep it's probably oscillating
            F1 = 0.
            for C0 in C0All:
                DF1 = J1[0]*self.GetFJ(C0) + K1[0]*self.GetFK(C0) \
                       + (J1[1]+K1[1])*self.GetFJ(C1) # Can use the same term for both

                if UseHF or self.DFA is None:
                    F1 += self.FHF + DF1
                else:
                    #####################################################################
                    # Here, form:
                    # D_avg = 2\sum_{i<=h}outer(C_i,C_i)
                    # D_up = \sum_{i<=l}outer(C_i,C_i)
                    # D_down = \sum_{i<h}outer(C_i,C_i)
                    # and use the code from LibEnsemble ==> GetFDFA to calculate
                    # DFDFA = V(D_up,D_down) - V(D_avg,D_avg)

                    DFDFA = self.GetFDFA(C0,C1, Return="DV")
                    F1 += self.F + DF1 + DFDFA
            F1 /= len(C0All)
            
            FNew = (1.-Mix2)*( (1.-Mix)*F1 + Mix*FOld ) + Mix2*FOld2
            if self.UseDegen:
                w,CX = self.DegenHelp.eighVir(FNew)
            else:
                w,U = la.eigh((CX.T).dot(FNew).dot(CX))
                CX = CX.dot(U)

            FOld2 = FOld*1.
            FOld = FNew*1.

            i1 = dk1
            
            C1 = CX[:,i1]

            epsilon_new[self.kl:]=w

            Err = eV*GapFactor*np.abs((epsilon_new[k1]-epsilon_new[k0])-(epsilon_old[k1]-epsilon_old[k0]))
            if Report or (step>(MaxStep*2/3)):
                # Show that the index shifted if it did
                if not(i1==dk1):
                    if not(Silent):
                        print("MOM shift @ %d : %d to %d"%(step, dk1, i1))
                if not(Silent):
                    print("%3d: epsilon(%d) = %.3f, epsilon(%d) = %.3f, Gap = %.3f [%.5f] eV"\
                          %(step,
                            k0, eV*epsilon_new[k0],
                            k1, eV*epsilon_new[k1],
                            eV*GapFactor*(epsilon_new[k1]-epsilon_new[k0]),
                            Err))
            epsilon_old = 1.*epsilon_new
            if Err<ErrCut:
                break
        
        if Err<(ErrCut*10.):
            if not(Silent):
                print("# Took %d steps to get to sc Err = %.6f eV"%(step, Err))
            # Now that we have successfully converged things, we need to
            # compute the other virtual orbitals using the correct operator.
            # This replaces J_1 with K_1, where appropriate.
            DF1 = J1[0]*self.GetFJ(C0) + K1[0]*self.GetFK(C0) \
                  + J1[1]*self.GetFJ(C1) + K1[1]*self.GetFK(C1)
            # Add DFA if appropriate
            if not(UseHF or self.DFA is None):
                DF1 += self.GetFDFA(C0,C1)
            FFinal = self.F + DF1

            if self.UseDegen:
                w,CX = self.DegenHelp.eighVir(FFinal)
            else:
                w,U = la.eigh((CX.T).dot(FFinal).dot(CX))
                CX = CX.dot(U)

            self.CE = 1.*self.C
            self.CE[:,self.kl:] = CX
            self.epsilonE = epsilon_new
        else:
            if not(Silent):
                print("# Took %d steps to get Err = %.6f eV - using perturbation"%(step, Err))
            self.CE = 1.*self.C
            self.epsilonE = self.epsilon*1.
            self.epsilonE[self.kl:]+=DEP


        if Debug:
            O = (C_In.T).dot(self.S_ao).dot(self.CE)
            print("Overlaps")
            print(NiceMat(O[k0:(k1+3),k0:(k1+3)]**2))

        # Get EST = 2(hl|lh) -- the singlet/triplet splitting
        self.EST = 2.*np.einsum('p,q,pq',C0,C0,self.GetFK(C1))

        # Get E01 = <S0|H|S1>
        # 1-RDM part
        self.E01 = (C0).dot(self.H_ao.dot(C1))
        # 2-RDM part
        for i in range(self.kh):
            A = 2.*self.GetFJ(self.CE[:,i]) - self.GetFK(self.CE[:,i])
            self.E01 += C0.dot(A.dot(C1))
        self.E01 = np.abs(self.E01) # real orbitals so may always take as +ve
        if not(Silent):
            print("# EST = %5.2f E01 = %7.4f"%(self.EST*eV, self.E01*eV))
            
        return GapFactor*(self.epsilonE[k1]-self.epsilonE[k0])
        
if __name__ == "__main__":
    psi4.set_output_file("__perturb.dat")
    psi4.set_options({
        'basis' : 'def2-tzvp',
        'reference': 'rhf',
    })
    psi4.geometry("""
0 1
Be
symmetry c1""")

    E, wfn = psi4.energy("scf", return_wfn=True)
    
    XHelp = ExcitationHelper(wfn)
    Gap1 = XHelp.SolveSingle()
    Gap2 = XHelp.SolveDouble()

    
    print("Gap(sx) = %.2f, Gap(dx) = %.2f"%(Gap1*eV, Gap2*eV))
