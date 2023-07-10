import multiprocessing as mp
from simulation import sample_simulation 
import time
import numpy as np
import os

class CEM:
    def __init__(self,par,sys,conformation,pressure=1,temperature=300):
        self.init_par = par
        self.env = sys
        self.con = conformation

    # multi-process #
    def multi_process(self,task):
        num_process = 4 
        pool = mp.Pool(processes=num_process)
        result_list = pool.map(task,[0,1,2,3])
        pool.close()
        pool.join()
        return result_list

    # calculate properties #
    def do_multi(self,job):  
        press = sample_simulation(self.init_par,self.env,temp=300,timestep=0.02)
        time.sleep(1)
        return press,job
        
    # run simulation #
    def run(self,nloop=3):
        start = time.time()
        for i in range(nloop):
            pr = self.multi_process(self.do_multi)
        end = time.time()
        print("Cost: "+str(end-start))
        

    

    