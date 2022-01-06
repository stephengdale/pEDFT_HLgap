#!/home/timgould/psi4conda/bin/python

import psi4
import numpy as np

from LibPerturb import *

from  optparse import OptionParser

parser = OptionParser()
parser.add_option('-M', type="string", default="Be")
parser.add_option('--DFA', type="string", default="pbe0")
parser.add_option('--RKS', default=False, action="store_true")
parser.add_option('--UKS', dest="RKS", action="store_false")
parser.add_option('--Basis', type="string", default="cc-pvdz")
parser.add_option('--dk0', type="int", default=0)
parser.add_option('--dk1', type="int", default=0)

parser.add_option('--SaveOrbs', type="string", default=None,
                  help="Save orbitals and related quantities to this file, if specified")
parser.add_option('--SaveERI', default=False, action="store_true",
                  help="Also save the ERI quantities")


(Opts, args) = parser.parse_args()

eV = 27.211

if Opts.M in ("He", "Be", "Ne", "Mg", "Ar"):
    MolStr = "%s\nsymmetry c1"%(Opts.M)
else:
    MolStr = "".join(list(open(Opts.M)))

MolStrSym = MolStr.replace("symmetry c1","")
#MolStrSym = MolStr

print("="*72)
print("Geometry")
print(MolStrSym)
print("="*72)
    
psi4.set_output_file("__EGKS-LUMO.out")

psi4.set_options({
    "basis": Opts.Basis,
    "reference": "rhf",
})

ActualDFA = GetDFA(Opts.DFA)

psi4.geometry(MolStrSym)
ESym, wfnSym = psi4.energy("scf", dft_functional=ActualDFA, return_wfn=True)
#print(wfnSym.sobasisset().petite_list().aotoso().to_array())
psi4.core.clean()

psi4.geometry(MolStr)
EDFA, wfnDFA = psi4.energy("scf", dft_functional=ActualDFA, return_wfn=True)
#print(wfnDFA.sobasisset().petite_list().aotoso().to_array())

print("# NBasis = %5d"%(wfnDFA.nmo()))

if np.abs(ESym-EDFA)>1e-6:
    print("Energies are different:")
    print("E(Sym) = %10.5f, E(NoS) = %10.5f"%(ESym, EDFA))


DH = psi4DegenHelper(wfnDFA, wfnSym)
F = wfnDFA.Fa().to_array(dense=True)
S = wfnDFA.S().to_array(dense=True)
w, C = DH.eigh(F, S)

dw = wfnDFA.epsilon_a().np-w
if np.mean(np.abs(dw))>1e-5:
    print("Difference in eigs:")
    print(NiceArr(dw))


CVir = C[:,DH.kL:]
DH.eigh_p(F, CVir)


if False:
    for k in range(14):
        print("Sym of orbital %d is %d"%(k, DH.GetSym(k)))
        
        print(np.sum(wfnDFA.Ca().to_array(dense=True)[:,k]**2))
        print(np.sum(C[:,k]**2))

if True:
    print("HOMO [%2d] symmetries: %s"%(DH.kH, str(DH.SymH)))
    print("LUMO [%2d] symmetries: %s"%(DH.kL, str(DH.SymL)))
    for Sym in DH.ListSym():
        print("%s : %s, %s"%(Sym, str(DH.OccSym(Sym)), str(DH.VirSym(Sym))))
