#!/home/timgould/psi4conda/bin/python

import psi4
import numpy as np
import scipy.linalg as la

class psi4DegenHelper:
    def __init__(self, wfnNoSym, wfnSym, epsCut=1e-5,
                 SetOrder=True):
        # Set the number of basis functions
        self.NBas = wfnNoSym.S().np.shape[0]

        # Work out the occupancies
        if wfnNoSym.nalpha()==wfnNoSym.nbeta():
            self.NDOcc = wfnNoSym.nalpha()
        else:
            self.NDOcc = wfnNoSym.nbeta()
        self.kH = wfnNoSym.nalpha()-1
        self.kL = self.kH+1

        # Does not have an order at first
        self.HasOrder = False

        # Set the S matrix in ao
        self.S_ao = wfnNoSym.S().to_array(dense=True)


        
        if wfnSym.nirrep()==1:
            # No symmetries - so no map needed
            self.Indx = {0:(0,self.NBas)}
            self.CMap = None
        else:
            # Symmetris means we need to define CMap
            
            # See the below examples for key information:
            # https://github.com/psi4/psi4numpy/blob/master/Self-Consistent-Field/RHF_symmetry.py
            # https://github.com/psi4/psi4numpy/blob/master/Self-Consistent-Field/helper_HF.py

            # First set up the indices and so->ao mappings
            self.Indx = {}
            T = {}
            I0=0
            for Sym,X in enumerate(wfnSym.aotoso().to_array()):
                if X.shape[1]==0: continue
                I1 = I0 + X.shape[1]
                T[Sym] = X
                self.Indx[Sym]= (I0,I1)
                I0 = I1

            # Then convert the alpha orbitals into the full ao basis
            # This is done via CMap
            self.CMap = {}
            for Sym,X in enumerate(wfnSym.Ca().to_array()):
                if len(X)==0: continue

                I0,I1 = self.Indx[Sym]
                self.CMap[Sym] = T[Sym].dot(X)

        # If required, set an order based on the current Fock matrix
        if SetOrder:
            self.eigh(wfnNoSym.Fa(), wfnNoSym.S(), UpdateOrder=True)
            
    def eigh(self, a, b, UpdateOrder=False, epsCut=1e-5):
        if not(self.CMap is None):
            # Solve the eigenvalue problem in the CMap bases per sym
            w = np.zeros((self.NBas,))
            v = np.zeros((self.NBas,self.NBas))
            for Sym in self.CMap:
                aSym = (self.CMap[Sym].T).dot(a).dot(self.CMap[Sym])
                if b is None: bSym = None
                else: bSym = (self.CMap[Sym].T).dot(b).dot(self.CMap[Sym])

                wSym,vSym = la.eigh(aSym, b=bSym)

                I0,I1 = self.Indx[Sym]

                # Then convert back to ao basis
                w[I0:I1]=wSym
                v[:,I0:I1]=(self.CMap[Sym]).dot(vSym)
        else:
            w, v = la.eigh(a, b=b)
            
        if UpdateOrder or not(self.HasOrder):
            # If asked, set the order and orbital<->mapping details
            self.OrderIndx = np.argsort(w)
            self.HasOrder = True

            self.ReverseIndx = np.zeros_like(self.OrderIndx)
            self.ReverseIndx[self.OrderIndx] = np.arange(len(w))
            
            self.SymIndx = {}
            for Sym in self.Indx:
                I0,I1 = self.Indx[Sym]
                self.SymIndx[Sym] = self.ReverseIndx[I0:I1]

            self.epsilon = w[self.OrderIndx]
            self.epsilonH = self.epsilon[self.kH]
            self.epsilonL = self.epsilon[self.kL]

            self.SymH = []
            for k in np.argwhere(np.abs(self.epsilon-self.epsilonH)<epsCut).reshape((-1,)):
                for Sym in self.SymIndx:
                    if k in self.SymIndx[Sym]: self.SymH+=[Sym,]
            self.SymL = []
            for k in np.argwhere(np.abs(self.epsilon-self.epsilonL)<epsCut).reshape((-1,)):
                for Sym in self.SymIndx:
                    if k in self.SymIndx[Sym]: self.SymL+=[Sym,]

            self.epsilon, self.C = w[self.OrderIndx], v[:,self.OrderIndx]
            
        return w[self.OrderIndx], v[:,self.OrderIndx] 

    def eighVir(self, F):
        # Diagonalise F in the basis of virtual orbitals only
        
        if self.CMap is None:
            CV = self.C[:,self.kL:]
            w,U = la.eigh((CV.T).dot(F).dot(CV))
            return w, CV.dot(U)
        else:
            NVir = self.NBas-self.kL
            wVir = np.zeros((NVir,))
            CVir = np.zeros((self.NBas,NVir))
            for Sym in self.ListSym():
                kk = self.VirSym(Sym)
                if len(kk)==0: continue
                
                kkL = kk - self.kL
                CV = self.C[:,kk]
                wT,UT = la.eigh((CV.T).dot(F).dot(CV))

                wVir[kkL] = wT
                CVir[:,kkL] = CV.dot(UT)

            return wVir, CVir
    
    def SymOrb(self, k):
        # Return the symmetry group of orbital k
        if not(self.HasOrder):
            print("Must define an order first!")
            quit()

        for Sym in self.SymIndx:
            if k in self.SymIndx[Sym]:
                return Sym

        print("Failed to find a symmetry")
        print(self.SymIndx)
        quit()

    def ListSym(self): return np.sort(list(self.Indx))
        
    def AllSym(self, Sym):
        # Return the orbitals in symmetry Sym
        return self.SymIndx[Sym]

    def OccSym(self, Sym):
        # Return the occupied orbitals of Sym
        kk = self.SymIndx[Sym]
        return np.sort(kk[kk<=self.kH])

    def VirSym(self, Sym):
        # Return the virtual orbitals of Sym
        kk = self.SymIndx[Sym]
        return np.sort(kk[kk>=self.kL])
