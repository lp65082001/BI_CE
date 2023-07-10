import MDAnalysis as mda
from MDAnalysis import transformations
import numpy as np

import warnings
# suppress some MDAnalysis warnings when writing PDB files
warnings.filterwarnings('ignore')

print("Using MDAnalysis version", mda.__version__)

pos = np.load("./position.npy")

n_residues = 960
n_atoms = n_residues

# create resindex list
resindices = np.repeat(range(n_residues), 1)
assert len(resindices) == n_atoms

# all water molecules belong to 1 segment
segindices = [0] * n_residues

# create system
structure = mda.Universe.empty(n_atoms,
                         n_residues=n_residues,
                         atom_resindex=resindices,
                         residue_segindex=segindices,
                         trajectory=True) # necessary for adding coordinates
                         
# add name
structure.add_TopologyAttr('name',['ru']*n_residues)

# add type
structure.add_TopologyAttr('type',['P']*n_residues)

print(structure.atoms.types)

# add resname
'''
resname_list = []
for i in range(16):
    for j in range(60):
        resname_list.append(f"P{j}")
'''
structure.add_TopologyAttr('resname',['P']*n_residues)

# add resid
structure.add_TopologyAttr('resid', list(range(1, n_residues+1)))

# add segid
#structure.add_TopologyAttr('segid', ["P"])


# add positions
structure.atoms.positions = pos

# add volume
transform = transformations.boxdimensions.set_dimensions([30.00, 30.00, 30.00, 90.00, 90.00, 90.00])
structure.trajectory.add_transformations(transform)

structure_pdb = structure.select_atoms("all")
structure_pdb.write("test.pdb")








