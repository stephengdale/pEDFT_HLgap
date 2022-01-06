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
parser.add_option('--MaxStep', type="int", default=50)

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

print("="*72)
print("Geometry")
print(MolStr)
print("="*72)
    
psi4.set_output_file("__EGKS-LUMO.out")

psi4.set_options({
    "basis": Opts.Basis,
    "reference": "rhf",
})


ActualDFA =GetDFA(Opts.DFA)

# First, do a run with full psi4 symmetries
psi4.geometry(MolStr.replace("symmetry c1",""))
ESym, wfnSym = psi4.energy("scf", dft_functional=ActualDFA, return_wfn=True)
psi4.core.clean()

# Then switch off symmetries and do another run
psi4.geometry(MolStr)
EDFA, wfnDFA = psi4.energy("scf", dft_functional=ActualDFA, return_wfn=True)

print("# NBasis = %5d"%(wfnDFA.nmo()))

# By passing in wfnSym we trigger symmetries mode
XHelp = ExcitationHelper(wfnDFA, wfnSym=wfnSym, RKS=Opts.RKS)

if Opts.dk0<0:
    # Use the guessed lowest triplet gap
    k0, k1 = XHelp.GuessLowestTriple()
elif Opts.dk1<0:
    # Use the guessed lowest singlet gap
    k0, k1 = XHelp.GuessLowestSingle()
else:
    # Use the specified options
    k0 = XHelp.SetFrom(Shift = Opts.dk0)
    k1 = XHelp.SetTo(Shift = Opts.dk1)

eps_DFA = wfnDFA.epsilon_a().to_array(dense=True)
print("Using %s:"%(Opts.DFA))
print("Gap(DFA) = %.2f eV"%((eps_DFA[k1]-eps_DFA[k0])*eV))
if not(Opts.SaveOrbs is None):
    C0 = 1.*wfnDFA.Ca().to_array(dense=True)
    epsilon0 = 1.*wfnDFA.epsilon_a().to_array(dense=True)

GapTriple = XHelp.SolveTriple(MaxStep=Opts.MaxStep)
if not(Opts.SaveOrbs is None):
    CT = 1.*XHelp.CE
    epsilonT = 1.*XHelp.epsilonE

GapSingle = XHelp.SolveSingle(MaxStep=Opts.MaxStep)
if not(Opts.SaveOrbs is None):
    CS = 1.*XHelp.CE
    epsilonS = 1.*XHelp.epsilonE

print("Using perturbative E%s:"%(Opts.DFA))
print("Gap(ts) = %.2f eV"%(GapTriple*eV))
print("Gap(sx) = %.2f eV"%(GapSingle*eV))
print("E(ST)   = %.2f eV"%((GapSingle-GapTriple)*eV))
print("E0 = %10.4f ET = %10.4f ES = %10.4f"%(EDFA, EDFA+GapTriple, EDFA+GapSingle))

if not(Opts.SaveOrbs is None):
    SysProps = {'T_ao':XHelp.T_ao, 'V_ao':XHelp.V_ao, 'S_ao':XHelp.S_ao,
                'kh':XHelp.kh, 'kl':XHelp.kl, 'kFrom':XHelp.kFrom, 'kTo':XHelp.kTo}

    if Opts.SaveERI:
        SysProps['ERIA']=XHelp.ERIA

    np.savez(Opts.SaveOrbs,
             C0=C0, epsilon0=epsilon0,
             CT=CT, epsilonT=epsilonT,
             CS=CS, epsilonS=epsilonS,
             SysProps=SysProps,
        )
