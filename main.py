""" Create by Amborse hui from M^5 lab """

from mapping import FA2CG 
from MD import MD_Universe
import warnings

warnings.filterwarnings("ignore")

# load config #

def process():
    data_all = FA2CG("./FA/pe_l.psf","./FA/1nptts1.dcd")
    data_all.mapping()
    data_all.cal_distribution()
    #data_all.xyz2data()

    initial_setting, initial_potential = data_all.get_parameter()

    sim = MD_Universe(initial_setting, initial_potential,28,temperature=300,pressure=1)
    sim.run()
   
if __name__ == '__main__':
   process()