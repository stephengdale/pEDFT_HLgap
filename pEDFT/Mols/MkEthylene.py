#!/home/timgould/psi4conda/bin/python3

import numpy as np
import sys

if len(sys.argv)<2:
    Ang=0
else:
    Ang=np.pi/180.*float(sys.argv[1])

zC = 1.2603
zH = 2.3235
xH = 1.7429

c = np.cos(Ang)
s = np.sin(Ang)

Mask = "%s %8.4f %8.4f %8.4f\n"

Str = "0 1\n\n"
Str += Mask%('C',0,0,-zC)
Str += Mask%('H', xH,0,-zH)
Str += Mask%('H',-xH,0,-zH)
Str += Mask%('C',0,0, zC)
Str += Mask%('H', c*xH, s*xH,zH)
Str += Mask%('H',-c*xH,-s*xH,zH)
Str += "units bohr\nsymmetry c1\n"

print(Str)
