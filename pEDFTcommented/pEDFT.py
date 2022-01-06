#!/home/timgould/psi4conda/bin/python

import psi4
import numpy as np

from LibPerturb import *

from  optparse import OptionParser

parser = OptionParser()
parser.add_option('-M', type="string", default="Be",
                  help="A molecule file in psi4 format -- be sure to set c1 symmetry or risk nonsense")
parser.add_option('--DFA', type="string", default="pbe0",
                  help="Specify the DFA")
parser.add_option('--Basis', type="string", default="cc-pvdz",
                  help="Specify the basis set")

parser.add_option('--dk0', type="int", default=0, help="Use HOMO-dk0")
parser.add_option('--dk1', type="int", default=0, help="Use LUMO+dk1")

parser.add_option('--ShowSpectra', default=False, action="store_true",
                  help="Show additional spectra - only the chosen excitation is \"exact\"")

parser.add_option('--UseDegen', default=True, action="store_true",
                  help="Use the internal degeneracy helper - useful for small molecules")
parser.add_option('--NoDegen', action="store_false",
                  help="Don't use the internal degeneracy helper - mostly unwise")

parser.add_option('--Mix', type="float", default=None,
                  help="Mixing in iterations - increase if failure to converge")

#parser.add_option('--RKS', default=False, action="store_true",
#                  help="Expert level: Don't use spin in the DFA (unwise)")
#parser.add_option('--UKS', dest="RKS", action="store_false",
#                  help="Expert level: Force spin in the DFA (wise and default)")
parser.add_option('--SaveOrbs', type="string", default=None,
                  help="Expert level: Save orbitals and related quantities to this file, if specified")
parser.add_option('--SaveERI', default=False, action="store_true",
                  help="Expert level: If SaveOrbs is set, also save the ERI quantities")


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
    "reference": "rks",
})

psi4.geometry(MolStr)

ActualDFA = GetDFA(Opts.DFA)
EDFA, wfnDFA = psi4.energy("scf", dft_functional=ActualDFA, return_wfn=True)

print("# NBasis = %5d"%(wfnDFA.nmo()))

XHelp = ExcitationHelper(wfnDFA, RKS=False, # RKS=Opts.RKS,
                         UseDegen=Opts.UseDegen)
if Opts.UseDegen:
    XHelp.DegenHelp.Report()

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

if not(Opts.Mix is None):
    XHelp.SetMix(Opts.Mix, Opts.Mix)

eps_DFA = wfnDFA.epsilon_a().to_array(dense=True)
print("Using %s:"%(Opts.DFA))
print("Gap(DFA) = %.2f eV"%((eps_DFA[k1]-eps_DFA[k0])*eV))
if not(Opts.SaveOrbs is None):
    C0 = 1.*wfnDFA.Ca().to_array(dense=True)
    epsilon0 = 1.*wfnDFA.epsilon_a().to_array(dense=True)

GapTriple = XHelp.SolveTriple()
epsilonT = 1.*XHelp.epsilonE
if not(Opts.SaveOrbs is None):
    CT = 1.*XHelp.CE

GapSingle = XHelp.SolveSingle()
epsilonS = 1.*XHelp.epsilonE
if not(Opts.SaveOrbs is None):
    CS = 1.*XHelp.CE

print("Using perturbative E%s:"%(Opts.DFA))
print("Gap(ts) = %.2f eV"%(GapTriple*eV))
print("Gap(sx) = %.2f eV"%(GapSingle*eV))
print("E(ST)   = %.2f eV"%((GapSingle-GapTriple)*eV))
print("E0 = %10.4f ET = %10.4f ES = %10.4f"%(EDFA, EDFA+GapTriple, EDFA+GapSingle))

def NiceSpectra(epsilon, kh):
    X = (epsilon-epsilon[kh])*eV
    print("-- Occupied to %d"%(kh))
    for k in range(0,kh+1,8):
        km = min(k+8,kh+1)
        print(" ".join(["%8.1f"%(x) for x in X[k:km]]))
    print("-- Virtual from %d"%(kh+1))
    for k in range(kh+1,len(X),8):
        km = min(k+8,len(X))
        print(" ".join(["%8.1f"%(x) for x in X[k:km]]))

if Opts.ShowSpectra:
    print("Triplet specta")
    NiceSpectra(epsilonT, XHelp.kh)
    print("Singlet specta")
    NiceSpectra(epsilonS, XHelp.kh)

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
