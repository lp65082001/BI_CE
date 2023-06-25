from mapping import Iteractive_boltzmann_inversion
from MD import MD_simulation
import MDAnalysis as mda

import time
import copy


class CEM:
    
    def __init__(self,data):
        pass

if __name__ == '__main__':
    psf = mda.Universe("./FA_data/pe_l.psf","./FA_data/1nptts1.dcd")
    data_all = Iteractive_boltzmann_inversion(psf)
    data_all.mapping()
    data_all.cal_distribution(-1)
    #data_all.xyz2data()

    #psf2 = mda.Universe("./test.data","./PE_bead_npt29.dcd")
    #data_all2 = xyz_dis(psf2,60,16)
    #data_all2.IBM_tr_cg()

    #compare_iter(data_all,data_all2).compare_plot()
    initial_parameter = data_all.get_parameter()
    #initial_state = copy.deepcopy(initial_parameter[0])

    print("MD run")
    initial_state = copy.deepcopy(initial_parameter[0])
    sim = MD_simulation(initial_state,initial_parameter[1],initial_parameter[2],initial_parameter[3],initial_parameter[4],28,310,1)
    sim.InitVelDis()
    output = sim.run()

