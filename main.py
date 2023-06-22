from mapping import Iteractive_boltzmann_inversion
from MD import MD_simulation
import MDAnalysis as mda

import time
import copy


class CEM:
    
    def __init__(self,data):
        pass

class print_output:

    def __init__(self,r):
        self.out = r
    def dump(self):
        print('%8s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t' % ('Timestep','KE','Temp','Press','evdw','ebond','eangle','edihedral'))
        for i in range(0,len(self.out)):
            print('%8d\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f\t' % (self.out[i][0],self.out[i][1],self.out[i][2],self.out[i][3],self.out[i][4],self.out[i][5],self.out[i][6],self.out[i][7]))


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
    start = time.time()
    initial_state = copy.deepcopy(initial_parameter[0])
    sim = MD_simulation(initial_state,initial_parameter[1],initial_parameter[2],initial_parameter[3],initial_parameter[4],28,310,1)
    output = sim.run()
    print_output(output).dump()
    end = time.time()
    print("Cost: "+str(end - start)) 
    print("Done")
