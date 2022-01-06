import psi4
import numpy as np
import scipy.linalg as la

from LibPerturb import *

eV = 27.211

np.set_printoptions(precision=5, suppress=True)



# Get the degeneracy of each orbital
def GetDegen(epsilon, eta=1e-5):
    Degen = np.zeros((len(epsilon),),dtype=int)
    for k in range(len(epsilon)):
        ii =  np.argwhere(np.abs(epsilon-epsilon[k])<eta).reshape((-1,))
        Degen[k] = len(ii)
    return Degen
        
    
class DegenHelper:
    def __init__(self, wfn, kh=None, eta=1e-6):
        C = wfn.Ca().to_array(dense=True)
        epsilon = wfn.epsilon_a().to_array(dense=True)
        H = wfn.H().to_array(dense=True)
        S = wfn.S().to_array(dense=True)
        C = wfn.Ca().to_array(dense=True)

        self.eta = eta
        self.Degen = GetDegen(epsilon, eta)
        if kh is None:
            self.kh = wfn.nalpha()-1
        else: self.kh = kh
        self.NBas = S.shape[0]

        self.Q = (C.T).dot(H).dot(C)
        self.Q = self.Q**2
        
        #FS = F + np.diag(np.arange(self.NBas))/self.NBas*self.eta
        #w,C = la.eigh(FS, b=S)
        ##self.Q = np.abs((C.T).dot(FS).dot(C))
        #self.Q = (C.T)**2
        
        # Initialise the decoupled blocks
        Maps = {}
        for k in range(self.NBas):
            ii = np.argwhere(self.Q[k,:]>self.eta).reshape((-1,))
            i0 = ii.min()
            if i0 in Maps:
                Maps[i0] = set(list(ii) + list(Maps[i0]))
            else:
                Maps[i0] = set(ii)

        # Find any accidental repeats
        Same = []
        for i0 in Maps:
            for i1 in Maps:
                if i0>=i1: continue # Only merge bigger to smaller
                # If dupe prepare merge
                if (i0 in Maps[i1]) or (i1 in Maps[i0]):
                    Same+=[(i0,i1)]

        # Eliminate any accidental repeats
        if len(Same)>0:
            for (i0,i1) in Same:
                if i1 in Maps:
                    Maps[i0] = set(list(Maps[i0])+list(Maps[i1]))
                    Maps.pop(i1)

        # Finally, check that every number is accounted for
        All = []
        for i0 in Maps:
            All += list(Maps[i0])
        if len(All)>self.NBas:
            print("Doubling up - quitting")
            quit()
        elif len(All)<self.NBas:
            print("Some elements missed - quitting")
            quit()
        else:
            M = np.diff(np.sort(All))
            if M.min()<1 or M.max()>1:
                print("Some elements missed - quitting")
                quit()

        # Convert back into sorted lists
        for i0 in Maps:
            Maps[i0] = np.array(list(Maps[i0]), dtype=int)
            Maps[i0] = np.sort(Maps[i0])
            #print("%3d :"%(i0), NiceArrInt(Maps[i0]))

        self.Maps = Maps
        self.SymList = sorted(list(Maps))

        self.SymID = {i0:"Sym%d"%(k) for k,i0 in enumerate(self.SymList)}
        self.IDSym = {"Sym%d"%(k):i0 for k,i0 in enumerate(self.SymList)}

        # Ensure it gets an order
        self.iOut = None

    # Report key information
    def Report(self):
        for ID in self.IDSym:
            i0 = self.IDSym[ID]
            kk = self.Maps[i0]
            if self.kh is None:
                print("%-6s :"%(ID) + ", ".join(["%3d"%(k) for k in kk]))
            else:
                kOcc = kk[kk<=self.kh]
                kVir = kk[kk>self.kh]

                if len(kOcc)>0:
                    StrOcc = ", ".join(["%3d"%(k) for k in kOcc])
                else: StrOcc=""
                
                if len(kVir)>0:
                    StrVir = ", ".join(["%3d"%(k) for k in kVir])
                else: StrVir=""

                print("%-6s :"%(ID) + StrOcc + " | " + StrVir)
                
    # Identify which symmetry group
    def GetSym(self, i): return self.GetID(i)
    def GetID(self, i):
        for i0 in self.SymList:
            if i in self.Maps[i0]:
                return self.SymID[i0]
        print("# Could not ID - quitting")
        quit()

    # Extract using symmetry group
    def Getk(self, Sym, k=0, Occ=None, Vir=None):
        i0 = self.IDSym[Sym]
        if not(Occ is None) and not(self.kh is None):
            kk = self.Maps[i0][self.Maps[i0]<=self.kh]
            return kk[len(kk)-Occ-1]
        elif not(Vir is None) and not(self.kh is None):
            kk = self.Maps[i0][self.Maps[i0]>self.kh]
            return kk[Vir]
        else:
            return self.Maps[i0][k]
            
    # Replaces the usual eigh routine
    def eigh(self, X, b=None, FreezeOrder=False):
        w_All = np.zeros((self.NBas))
        v_All = np.zeros((self.NBas, self.NBas))
        k0 = 0
        for i0 in self.Maps:
            i_M = self.Maps[i0]
            X_M = X[i_M,:][:,i_M]
            if not(b is None):
                b_M = b[i_M,:][:,i_M]
            else: b_M = None

            w_M,v_M = la.eigh(X_M, b=b_M)

            k1 = k0 + w_M.shape[0]

            w_All[k0:k1] = w_M
            v_All[i_M,k0:k1] = v_M

            k0 = k1

        if FreezeOrder or self.iOut is None:
            self.iOut = np.argsort(w_All)
           
        return w_All[self.iOut],v_All[:,self.iOut]
    
    # Replaces eigh((C.T).dot(X).dot(C))
    def eigh_p(self, X, C):
        # Screen the C
        CP = C[:,:5]
        #print(NiceMat((CP.T).dot(X).dot(CP)))
        k_Map = {} # contains the elements that contain i0 and its Map
        for i0 in self.Maps:
            i_M = self.Maps[i0]
            P_M = np.sum(C[i_M,:]**2, axis=0)

            k_Map[i0] = np.argwhere(P_M>self.eta).reshape((-1,))

            C_M = C[:,k_Map[i0]]
            #print(NiceArrInt(k_Map[i0][:5]))
            #print(NiceMat((C_M[:,:5].T).dot(X).dot(C_M[:,:5])))


        #quit()
        NRet = C.shape[1]
        w = np.zeros((NRet))
        U = np.zeros((NRet,NRet))
        for i0 in self.Maps:
            i_M = self.Maps[i0]
            k_M = k_Map[i0]
            X_M = X[i_M,:][:,i_M]
            C_M = C[i_M,:][:,k_M]
            #C_M = C[:,k_M]
            #X_M = X

            w_M,U_M = la.eigh((C_M.T).dot(X_M).dot(C_M))
            #print(NiceArr(w_M))

            w[k_M]=w_M
            U[(k_M[:,None],k_M)]=U_M

        if False:
            print("=")
            print(NiceArr(w[:5]))
            wp,Up=la.eigh((C.T).dot(X).dot(C))
            print(NiceArr(wp[:5]))

        return w,U

if __name__ == "__main__":
    import psi4
    psi4.set_output_file("__temp.dat")
    psi4.set_options({
        "basis":'6-31g', 'reference':'rhf',})
    psi4.geometry("""
0 1
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
""")
    E, wfn = psi4.energy("pbe0", return_wfn=True)

    kh = wfn.nalpha()
    DH = DegenHelper(wfn)
    DH.Report()
    IDh = DH.GetID(kh)
    IDl = DH.GetID(kh+1)
    print(DH.Getk(IDh,Occ=0))
    print(DH.Getk(IDh,Vir=0))
    print(DH.Getk(IDl,Occ=0))
    print(DH.Getk(IDl,Vir=0))
