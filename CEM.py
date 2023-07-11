import multiprocessing as mp
from simulation import sample_simulation 
import time
import numpy as np
import os
np.random.seed(0)

class CEM:
    def __init__(self,par,sys,conformation,pressure=1,temperature=300,n_s = 20):
        self.init_par = par
        self.env = sys
        self.con = conformation
        self.num_sample = n_s
        self.dis_sigma=0.01
        self.pressure_ref = pressure

    """ multi-process """
    def multi_process(self,task,sample_l):
        num_process = 4 
        pool = mp.Pool(processes=num_process)
        result_list = pool.map(task,sample_l)
        pool.close()
        pool.join()
        return result_list

    """ calculate properties """
    def do_multi(self,job):  
        ''' sample table input '''
        input_par = self.sample_table[job,:]
        press,bond,angle,rdf = sample_simulation(input_par,self.env,temp=300,timestep=0.02)
        time.sleep(1)
        return job,press,bond,angle,rdf
        
    """ run simulation """
    def run(self,nloop=5):
        pressure_table = np.zeros((self.num_sample,1))
        bond_table = np.zeros((self.num_sample,2))
        angle_table = np.zeros((self.num_sample,2))
        rdf_table = np.zeros((self.num_sample,100))

        start = time.time()
        for i in range(nloop):
            pr = self.multi_process(self.do_multi,[0+i*4,1+i*4,2+i*4,3+i*4])
            for i in pr:
                pressure_table[i[0],:] = i[1]
                bond_table[i[0],:] = i[2]
                angle_table[i[0],:] = i[3]
                rdf_table[i[0],:] = i[4]

        end = time.time()
        print("Cost: "+str(end-start))
        return [pressure_table,bond_table,angle_table,rdf_table]

    
    """ sample angle and angle stiffness """
    def create_sample_a(self,x):
        y = np.zeros((self.num_sample,len(self.init_par)))
        y[:,0] = x[0]
        y[:,1] = x[1]
        y[:,2] = np.random.normal(x[2],self.dis_sigma*100,self.num_sample)
        y[:,3] = np.random.normal(x[3],self.dis_sigma*2,self.num_sample)
        y[:,4] = np.random.normal(x[4],self.dis_sigma,self.num_sample)
        y[:,5] = np.random.normal(x[5],self.dis_sigma,self.num_sample)

        for i in range(0,y[:,2].shape[0]):
            if y[i,2] > 180:
                y[i,2] = 180

        self.sample_table = y

    """ sample epsilon and sigma """
    def create_sample_se(self,x):
        y = np.zeros((self.sample,self.par))
        y[:,0] = np.random.normal(x[0],self.dis_sigma*2,self.sample)
        y[:,1] = np.random.normal(x[1],self.dis_sigma*2,self.sample)
        y[:,2] = x[2]
        y[:,3] = x[3]
        y[:,4] = x[4]
        y[:,5] = x[5]
        return y
    
    """ sample pressure """
    def create_sample_p(self,x,xx):
        y = np.zeros((self.sample,self.par))
        thresholdx1 = 0.01
        thresholdx2 = 0.5

        if (x[0] > xx[0]+thresholdx1):
            x11 = xx[0]+thresholdx1
        elif (x[0]<xx[0]-thresholdx1):

            x11 = xx[0]-thresholdx1
        else:
            x11 = x[0]

        if (x[1] > xx[1]+thresholdx2):
            x12 = xx[1]+thresholdx2
        elif (x[1]<xx[1]-thresholdx2):
            x12 = xx[1]-thresholdx2
        else:
            x12 = x[1]
        y[:,0] = np.random.normal(x11,self.dis_sigma,self.sample)
        y[:,1] = np.random.normal(x12,self.dis_sigma,self.sample)
        y[:,2] = x[2]
        y[:,3] = x[3]
        y[:,4] = x[4]
        y[:,5] = x[5]

        return y

    """ select angle """
    def select_sample_a(self,x1,x2,x3,x4):
        self.pp = np.mean(x4[1:-1])
        y1 = x1.argsort()
        return np.array(y1)[0:6]
    
    """ select epsilon and sigma"""    
    def select_sample_se(self,x1,x2,x3,x4):
        self.pp = np.mean(x4[1:-1])

        y31 = x3[:,0].argsort()
        y32 = x3[:,1].argsort()

        z1 = np.intersect1d(y31[0:8],y32[0:8])

        return np.array(z1)
        
    """ select pressure """
    def select_sample_p(self,x1,x2,x3,x4):
        self.pp = np.mean(x4[1:-1])

        y31 = x3[:,0].argsort()
        y32 = x3[:,1].argsort()
        y4 = x4.argsort()

        z1 = np.intersect1d(y31[0:6],y4[0:6])
        z2 = np.intersect1d(y32[0:6],y4[0:6])
        
        zz = np.concatenate((z1,z2))

        if zz.shape[0]==0:
            zz = y4[0:6]

        return zz
    
    """ generation next sample average"""
    def next_generation(self,x1,x2):
        newlist = x2[x1,:]
        new_gen = np.mean(newlist,axis=0)
        self.eps = new_gen[0]
        self.sig = new_gen[1]
        self.be = new_gen[2]
        self.bl = new_gen[3]
        self.ae = new_gen[4]
        self.al = new_gen[5]
        return new_gen
    

    """ loss calculation"""
    def loss_calculation(self,x):
        loss_table = np.zeros((self.num_sample,5))
        press_t = x[0]
        angle_t = x[2]
        rdf_t = x[3]

        loss_table[:,0] = (np.absolute(press_t-self.pressure_ref)).reshape(-1)
        loss_table[:,[1,2]] = np.array([np.absolute(angle_t[:,0]-np.mean(self.con[0][2])),np.absolute(angle_t[:,1]-np.std(self.con[0][3]))]).reshape(-1,2)


        return loss_table

    """ dump sample average """
    def dump_parameter(self,x,t,stage):

        f = open(self.dump,"a+")

        f.write("epoch: {0:3d}, bond_length: {1:5f}, bond_energy: {2:5f}, angle: {3:5f}, angle_energy: {4:5f}, sigma: {5:5f}, epsilon: {6:5f}, press: {7:5f}, stage: {8:5f}, times: {9:5f}\n".format(x,self.bl,self.be,self.al,self.ae,self.sig,self.eps,self.pp,stage,time.time()-t))

        f.close()

    """ dump sample , loss ,pressure """
    def dump_search(self,x,press,error,error2,error3):
        f = open(self.dump2,"a+")
        for i in range(0,x.shape[0]):
            f.write("{0:5f}\t{1:5f}\t{2:5f}\t{3:5f}\t{4:5f}\t{5:5f}\t{6:5f}\t{7:5f}\t{8:5f}\t{9:5f}\t{10:5f}\n".format(x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],press[i],error[i,0],error[i,1],error2[i],error3[i]))
        f.close()
    
    """ converagence angle """
    def convergence_a(self,x1,x2,threshold):
        newlist = x2[x1]
        if np.mean(newlist<=threshold):
            retursn -1
        else: 
            return 1

    """ converagence epsilon and sigma"""
    def convergence_se(self,x1,x2,threshold1,threshold2,t):
        eps = x2[x1,0]
        sig = x2[x1,1]
        eps2 = t[x1,0]
        sig2 = t[x1,1]

        if (np.mean(eps)<=threshold1 and np.mean(sig)<=threshold2):
            return -1,[np.mean(eps2),np.mean(sig2)]
        else: 
            return 1,[np.mean(eps2),np.mean(sig2)]

    """ epsilon and simga calculation"""
    def cal_rdf(self,x):
        #eps = np.max(x)
        eps = np.max(x[0:50])
        s1 = np.where(x>=1)[0][0]

        s1p = 0.05+s1*0.1
        sig = 0.1*((1-x[s1])/(x[s1+1]-x[s1]))+s1p
        return np.array([eps, sig])

    def CEM_optimizer_run(self):
        self.create_sample_a(self.init_par)
        properties = self.run(nloop=int(self.num_sample/4))
        loss = self.loss_calculation(properties)
        print(loss)
        pass


        

    

    