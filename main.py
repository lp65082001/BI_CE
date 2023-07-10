from mapping import Iteractive_boltzmann_inversion
import MDAnalysis as mda
import time
from pdb_make import output_cg
from xml_make import output_xml
from simulation import sample_simulation
from CEM import CEM




# calculate FA distribtion and CG initial parameter #
psf = mda.Universe("./FA_data/pe_l.psf","./FA_data/1nptts1.dcd")
data_all = Iteractive_boltzmann_inversion(psf)
data_all.cal_distribution()
    
# get some parameter #
initial_par = data_all.get_init_parameter()
ref_structure = data_all.get_ref_fa()
initial_position = data_all.get_position()

# output required file for simulation (pdb, xml) #
output_cg(initial_position,initial_par[1])
output_xml(initial_par[1])

#sample_simulation(initial_par[0],initial_par[1],temp=300,timestep=0.02)
optimizer = CEM(initial_par[0],initial_par[1],ref_structure,pressure=1,temperature=300)
optimizer.run()




