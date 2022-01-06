import psi4
import sys
import os
sys.path.append(os.path.abspath("./pEDFT"))
from LibPerturb import *
from operator import itemgetter
import numpy as np

# Record Character Tables
charTables = []

charTable = [('Ap', 'App'), 
np.array([
 [1, 1],
 [1,-1]
 ])
]

charTables.append(charTable)

charTable = [('Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u'),
np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1,-1,-1, 1, 1,-1,-1],
    [1,-1, 1,-1, 1,-1, 1,-1],
    [1,-1,-1, 1, 1,-1,-1, 1],
    [1, 1, 1, 1, 1,-1,-1,-1],
    [1, 1,-1,-1, 1,-1, 1, 1],
    [1,-1, 1,-1, 1, 1,-1, 1],
    [1,-1,-1, 1, 1, 1, 1,-1]
 ])
]

charTables.append(charTable)

charTable = [('A1', 'A2', 'B1', 'B2'),
np.array([
    [1, 1, 1, 1],
    [1, 1,-1,-1],
    [1,-1, 1,-1],
    [1,-1,-1, 1]
    ])
]

charTables.append(charTable)

eV = 27.2114

def orbitalSymmetries(wfn):
    symbols = wfn.molecule().irrep_labels()
    orbs = wfn.epsilon_a().nph
    orbitals = []
    i = 0
    for s in symbols:
        for o in orbs[i]:
            orbitals.append((o, s))
        i += 1
    orbitals.sort(key=itemgetter(0))
    return(orbitals)

def writeData(file_name, mol_names, data, symm):
    data_file = open(file_name, 'w')
    i = 0
    while i < len(mol_names):
        data_file.write("{:20s} {:20.5f} {:40s}\n".format(mol_names[i], data[i], symm[i]))
        i += 1

def genData(func, basis):

    def transitionSymmetry(occ, virt):
        print(charTable[0], occ, virt)
        o = charTable[0].index(occ)
        v = charTable[0].index(virt)
        mult = charTable[1][o]*charTable[1][v]
        tIndex = [tuple(i) for i in charTable[1]].index(tuple(mult))
        return charTable[0][tIndex]
    
    names = ['acetaldehyde', 'acetylene', 'ammonia', 'carbon_monoxide', 'cyclopropene', 'diazomethane', 'dinitrogen', 'ethylene', 'formaldehyde', 'formamide', 'hydrogen_chloride', 'hydrogen_sulfide', 'ketene', 'methanimine', 'nitrosomethane', 'thioformaldehyde', 'water']
    trans = [          'App',      'B2g',     'App',              'B1',           'B1',           'A2',        'B2g',      'B3u',           'A2',           'App',            'B1',               'A2',     'A2',         'App',            'App',               'A2',    'B1']
    
    minimum = []
    symm = []
    ks = []
    for n in names:
        print(n)
        psi4.set_output_file(f"{n}.out")
        psi4.set_options({'basis': basis,
                          'scf_type': 'df',
                          'reference' : 'rhf',
                          'save_jk' : True,
                          'df_ints_io' : 'save',
                          'e_convergence': 1e-8,
                          'd_convergence': 1e-8})
        
        xyz = '0 1 \n' + "".join(open(f'./QUEST1/{n}.xyz', 'r').readlines()[2:])
        xyz = psi4.geometry(xyz)
        e, wfnSym = psi4.energy(func, molecule=xyz, return_wfn=True)
        print(tuple(wfnSym.molecule().irrep_labels()))
        print([i[0] for i in charTables])
        print([i[0] for i in charTables].index(tuple(wfnSym.molecule().irrep_labels())))
        charTable = charTables[[i[0] for i in charTables].index(tuple(wfnSym.molecule().irrep_labels()))]
        kh = wfnSym.nalpha()-1
        orbitals = orbitalSymmetries(wfnSym)
        xyz = '0 1 \n' + "".join(open(f'./QUEST1/{n}.xyz', 'r').readlines()[2:]) + '\nsymmetry c1'
        xyz = psi4.geometry(xyz)
        e, wfn = psi4.energy(func, molecule=xyz, return_wfn=True)
        
        Nocc = 4
        Nvirt = 4
        Tocc = np.arange(kh-Nocc+1, kh+1)
        Tvirt = np.arange(kh+1, kh+Nvirt+1)
        excitations = []
        ks_excitations = []
        for i in Tocc:
            print(i)
            XHelp = ExcitationHelper(wfn, wfnSym=wfnSym, RKS=False)
            XHelp.SetFrom(Shift=kh-i)
            for j in Tvirt:
                print(j)
                result = [(orbitals[j][0]-orbitals[i][0])*eV, f'HOMO-{kh-i}', f'LUMO+{j-kh-1}', orbitals[i][1], orbitals[j][1]]
                result.append(transitionSymmetry(result[-2], result[-1]))
                ks_excitations.append(result)
                XHelp.SetTo(Shift=j-(kh+1))    
                E = XHelp.SolveSingle()*eV
                result = [E, f'HOMO-{kh-i}', f'LUMO+{j-kh-1}', orbitals[i][1], orbitals[j][1]]
                print(result)
                print(charTable[0])
                print(charTable[0].index(result[-2]))
                result.append(transitionSymmetry(result[-2], result[-1]))
                excitations.append(result)
        excitations.sort(key=itemgetter(0))
        print(excitations)
        ks_excitations.sort(key=itemgetter(0))
        print(ks_excitations)
        sym = [i[-1] for i in ks_excitations]
        ks.append(ks_excitations[sym.index(trans[names.index(n)])])
        
        minimum.append(excitations[0])
        sym = [i[-1] for i in excitations]
        print(n)
        print(trans[names.index(n)])
        print(sym)
        symm.append(excitations[sym.index(trans[names.index(n)])])

    dat = [i[0] for i in minimum]
    rec = ['_'.join(i[1:]) for i in minimum]
    writeData(f'ignore_E__{func}_{basis}.txt', names, dat, rec)
    print(symm)
    dat = [i[0] for i in symm]
    rec = ['_'.join(i[1:]) for i in symm]
    writeData(f'E_{func}_{basis}.txt', names, dat, rec)
    print(ks)
    dat = [i[0] for i in ks]
    rec = ['_'.join(i[1:]) for i in ks]
    writeData(f'{func}_{basis}.txt', names, dat, rec)

    return excitations

genData('hf', 'cc-pvdz')    
    





