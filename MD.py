import numpy as np
import math
from scipy.spatial import distance_matrix
#from numba import jit
#import jax.numpy as jnp
import warnings
import time
warnings.filterwarnings("ignore")


gamma = 3
k = 0.0019872067
dt = 4
cutoff = 20
#nrun = 100
ndump = 10
mass = 28
temp = 310
press = 1

class MD_simulation:

    def __init__(self,position,bond_table,angle_table,volume,potential,mass,temp,press,nrun=10):
        self.mass = mass
        self.a = np.array([volume[0],0,0])
        self.b = np.array([0,volume[1],0])
        self.c = np.array([0,0,volume[2]])
        self.a_len = math.sqrt(np.dot(np.array([volume[0],0,0]),np.array([volume[0],0,0])))
        self.b_len = math.sqrt(np.dot(np.array([0,volume[1],0]),np.array([0,volume[1],0])))
        self.c_len = math.sqrt(np.dot(np.array([0,0,volume[2]]),np.array([0,0,volume[2]])))
        self.volume = volume[0]*volume[1]*volume[2]
        self.bond_init = potential[0]
        self.bond_energy = potential[1]
        self.angle_init = potential[2]
        self.angle_energy = potential[3]
        self.sigma = potential[4]
        self.epsilon = potential[5]
        self.temp = temp
        self.press = press
        self.bond_table = bond_table[:,[1,2]]-1
        self.angle_table = angle_table[:,[1,2,3]]-1
        self.bond_dis = []
        self.angle_dis = []
        self.torsion_dis = []
        self.system = np.hstack((position,np.zeros([position.shape[0],3])))
        self.nrun = nrun
        self.nei_table = np.vstack((bond_table[:,[1,2]]-1,angle_table[:,[1,3]]-1))

    def InitVelDis(self):
        self.system[:,3] = np.random.randn(self.system.shape[0])
        self.system[:,4] = np.random.randn(self.system.shape[0])
        self.system[:,5] = np.random.randn(self.system.shape[0])
        
        self.system[:,3] -= np.sum(self.system[:,3])/self.system.shape[0]/math.sqrt(self.mass)
        self.system[:,4] -= np.sum(self.system[:,4])/self.system.shape[0]/math.sqrt(self.mass)
        self.system[:,5] -= np.sum(self.system[:,5])/self.system.shape[0]/math.sqrt(self.mass)
        
        scale = math.sqrt(self.temp/self.GetTemp())
        self.system[:,3:6] = np.multiply(self.system[:,3:6],scale)
    #@jit
    def SetTemp(self):
        t_s = self.GetTemp() - self.temp
        if (abs(t_s) > 0.02):
            t_t = self.GetTemp() - 0.5*t_s
            scale = math.sqrt(t_t/self.GetTemp())
            self.system[:,3:6] = self.system[:,3:6]*scale
    #@jit
    def GetTemp(self):
        return (self.mass*np.sum(np.power(self.system[:,3:6],2))*0.239005736*10000)/(gamma*(self.system.shape[0]-1)*k)
    #@jit
    def IncrementalPos(self,increments):
        self.system[:,0:3] = self.system[:,0:3] + increments
    #@jit
    def IncrementalVel(self,increments):
        self.system[:,3:6] = self.system[:,3:6] + increments
    #@jit   
    def KineticEnergy(self):
        return 0.5*self.mass*np.sum(np.power(self.system[:,3:6],2))*0.239005736*10000
    
    def wrap_box(self,r,size):
        new_a = np.tile(self.a,(size,1))
        new_b = np.tile(self.b,(size,1))
        new_c = np.tile(self.c,(size,1))
        return new_a*np.round(np.dot(r,self.a)/self.a_len**2).reshape(-1,1) + new_b*np.round(np.dot(r,self.b)/self.b_len**2).reshape(-1,1) + new_c*np.round(np.dot(r,self.c)/self.c_len**2).reshape(-1,1)

    def wrap(self):
        pos_table = distance_matrix(self.system[:,0:3],self.system[:,0:3])
        out_off = np.where((np.triu(pos_table,1) >= cutoff) & (np.triu(pos_table,1)!=0))
        rij = self.system[out_off[0],0:3]-self.system[out_off[1],0:3]
        rij = rij - self.wrap_box(rij,rij.shape[0])
        rij_len = np.linalg.norm(rij,axis=1)
        pos_table[out_off[0],out_off[1]] = rij_len
        end = time.time()
        return pos_table
    # calcaule potential
    #@jit
    def CalculateForces(self):
        forces = np.zeros((self.system.shape[0],3))
        
        pos_table = self.wrap()
        
        cutoff_dis_list = np.where((np.triu(pos_table,1) <= cutoff) & (np.triu(pos_table,1)!=0))
        cutoff_dis_list_ = np.sort(np.vstack((cutoff_dis_list[0][:],cutoff_dis_list[1][:])).T,axis=1)
        '''
        no_neighber = np.array(list(set(map(tuple, cutoff_dis_list_))- 
                                    set(map(tuple, self.nei_table))
                                    ))
        '''
        
        nt = np.zeros((self.system.shape[0],self.system.shape[0]))
        nt[cutoff_dis_list_[:,0],cutoff_dis_list_[:,1]] += 1
        nt[self.nei_table[:,0],self.nei_table[:,1]] += 2
        no_neighber = np.where((nt==1) & (np.triu(nt,1)!=0))
        self.vdw_index = np.hstack((no_neighber[0].reshape(-1,1),no_neighber[1].reshape(-1,1)))
        
        ## nonbond term ##
        nonbonded_rij = self.system[self.vdw_index[:,0],0:3]-self.system[self.vdw_index[:,1],0:3]
        rij_len = np.linalg.norm(nonbonded_rij,axis=1)
        sr6 = np.power(self.sigma**2/np.power(rij_len,2),3)
        sr12 = np.power(sr6,2)
        fij = 4*self.epsilon/rij_len*(-12*sr12+6*sr6)
        #forces[np.ix_(self.vdw_index[:,0],np.array([0,1,2]))]  += (fij.reshape(-1,1)*nonbonded_rij/rij_len.reshape(-1,1))
        #forces[np.ix_(self.vdw_index[:,1],np.array([0,1,2]))]  -= (fij.reshape(-1,1)*nonbonded_rij/rij_len.reshape(-1,1))
        forces[self.vdw_index[:,0],:]  += (fij.reshape(-1,1)*nonbonded_rij/rij_len.reshape(-1,1))
        forces[self.vdw_index[:,1],:]  -= (fij.reshape(-1,1)*nonbonded_rij/rij_len.reshape(-1,1))

        ## bond term ##
        bond_rij =self.system[self.bond_table[:,0],0:3]-self.system[self.bond_table[:,1],0:3]
        bond_rij_len = np.linalg.norm(bond_rij,axis=1)
        fbond = -2*self.bond_energy*(bond_rij_len-self.bond_init)
        #forces[np.ix_(self.bond_table[:,0],np.array([0,1,2]))]  += (fbond.reshape(-1,1)*bond_rij/bond_rij_len.reshape(-1,1))
        #forces[np.ix_(self.bond_table[:,1],np.array([0,1,2]))]  -= (fbond.reshape(-1,1)*bond_rij/bond_rij_len.reshape(-1,1))
        forces[self.bond_table[:,0],:]  += (fbond.reshape(-1,1)*bond_rij/bond_rij_len.reshape(-1,1))
        forces[self.bond_table[:,1],:]  -= (fbond.reshape(-1,1)*bond_rij/bond_rij_len.reshape(-1,1))

        ## angle term ##
        angle_lijk_1 = self.system[self.angle_table[:,0],0:3]-self.system[self.angle_table[:,1],0:3]
        delx1 = angle_lijk_1[:,0]
        dely1 = angle_lijk_1[:,1]
        delz1 = angle_lijk_1[:,2]
        r1 = np.linalg.norm(angle_lijk_1,axis=1)

        angle_lijk_2 = self.system[self.angle_table[:,2],0:3]-self.system[self.angle_table[:,1],0:3]
        delx2 = angle_lijk_2[:,0]
        dely2 = angle_lijk_2[:,1]
        delz2 = angle_lijk_2[:,2]
        r2 = np.linalg.norm(angle_lijk_2,axis=1)
        
        # angle
        c = delx1*delx2 + dely1*dely2 + delz1*delz2
        c /= r1*r2   

        cup = np.where(c>1.0)[0]
        clow = np.where(c<-1.0)[0]
        for i in range(cup.shape[0]):
            c[i] = 1
        for i in range(clow.shape[0]):
            c[i] = -1
        
        s = np.power(1.0-np.power(c,2),0.5)
        slow = np.where(s<0.001)[0]
        for i in range(slow.shape[0]):
             s[i] = 0.001
        s = 1.0/s
        
        dtheta = np.arccos(c)-math.radians(self.angle_init)
        tk = self.angle_energy * dtheta
        a = -2.0 * tk * s
        a11 = a*c / r1**2
        a12 = -a / (r1*r2)
        a22 = a*c / r2**2

        f1x = (a11*delx1 + a12*delx2).reshape(-1,1)
        f1y = (a11*dely1 + a12*dely2).reshape(-1,1)
        f1z = (a11*delz1 + a12*delz2).reshape(-1,1)
        f3x = (a22*delx2 + a12*delx1).reshape(-1,1)
        f3y = (a22*dely2 + a12*dely1).reshape(-1,1)
        f3z = (a22*delz2 + a12*delz1).reshape(-1,1)

        f1_ = np.hstack((f1x,f1y))
        f1a = np.hstack((f1_,f1z))
        f2_ = np.hstack((f1x + f3x,f1y + f3y))
        f2a = np.hstack((f2_,f1z + f3z))
        f3_ = np.hstack((f3x,f3y))
        f3a = np.hstack((f3_,f3z))

        forces[np.ix_(self.angle_table[:,0],np.array([0,1,2]))] += f1a
        forces[np.ix_(self.angle_table[:,1],np.array([0,1,2]))] -= f2a
        forces[np.ix_(self.angle_table[:,2],np.array([0,1,2]))] += f3a


        return forces
    #@jit
    def CalculateEnergy(self):
        # nonbond #
        nonbonded_rij = self.system[self.vdw_index[:,0],0:3]-self.system[self.vdw_index[:,1],0:3]
        rij_len = np.linalg.norm(nonbonded_rij,axis=1)
        sr6 = np.power(self.sigma**2/np.power(rij_len,2),3)
        sr12 = np.power(sr6,2)
        energy = 4*self.epsilon*(sr12-sr6)

        # bond #
        bond_rij =self.system[self.bond_table[:,0],0:3]-self.system[self.bond_table[:,1],0:3]
        bond_rij_len = np.linalg.norm(bond_rij,axis=1)
        bond_energy = self.bond_energy*(bond_rij_len-self.bond_init)**2
        self.bond_dis = bond_rij_len

        # angle #
        angle_lijk_1 = self.system[self.angle_table[:,0],0:3]-self.system[self.angle_table[:,1],0:3]
        delx1 = angle_lijk_1[:,0]
        dely1 = angle_lijk_1[:,1]
        delz1 = angle_lijk_1[:,2]
        r1 = np.linalg.norm(angle_lijk_1,axis=1)

        angle_lijk_2 = self.system[self.angle_table[:,2],0:3]-self.system[self.angle_table[:,1],0:3]
        delx2 = angle_lijk_2[:,0]
        dely2 = angle_lijk_2[:,1]
        delz2 = angle_lijk_2[:,2]
        r2 = np.linalg.norm(angle_lijk_2,axis=1)
        
        # angle
        c = delx1*delx2 + dely1*dely2 + delz1*delz2
        c /= r1*r2   

        cup = np.where(c>1.0)[0]
        clow = np.where(c<-1.0)[0]
        for i in range(cup.shape[0]):
            c[i] = 1
        for i in range(clow.shape[0]):
            c[i] = -1
        
        s = np.power(1.0-np.power(c,2),0.5)
        slow = np.where(s<0.001)[0]
        for i in range(slow.shape[0]):
             s[i] = 0.001
        s = 1.0/s
        
        dtheta = np.arccos(c)-math.radians(self.angle_init)
        angle_energy = self.angle_energy* np.power(dtheta,2)
        self.angle_dis = dtheta

        return np.sum(energy),np.sum(bond_energy),np.sum(angle_energy)

        angle_lijk_1 = self.system[self.angle_table[:,0],0:3]-self.system[self.angle_table[:,1],0:3]
        delx1 = angle_lijk_1[:,0]
        dely1 = angle_lijk_1[:,1]
        delz1 = angle_lijk_1[:,2]
        r1 = np.linalg.norm(angle_lijk_1,axis=1)

        angle_lijk_2 = self.system[self.angle_table[:,2],0:3]-self.system[self.angle_table[:,1],0:3]
        delx2 = angle_lijk_2[:,0]
        dely2 = angle_lijk_2[:,1]
        delz2 = angle_lijk_2[:,2]
        r2 = np.linalg.norm(angle_lijk_2,axis=1)
        
        # angle
        c = delx1*delx2 + dely1*dely2 + delz1*delz2
        c /= r1*r2   

        cup = np.where(c>1.0)[0]
        clow = np.where(c<-1.0)[0]
        for i in range(cup.shape[0]):
            c[i] = 1
        for i in range(clow.shape[0]):
            c[i] = -1
        
        s = np.power(1.0-np.power(c,2),0.5)
        slow = np.where(s<0.001)[0]
        for i in range(slow.shape[0]):
             s[i] = 0.001
        s = 1.0/s
        
        dtheta = np.arccos(c)-math.radians(self.angle_init)
        angle_energy = self.angle_energy* np.power(dtheta,2)
        self.angle_dis = dtheta
        return np.sum(angle_energy) 
    #@jit
    def CalPress(self):     
        pxx = ((np.sum(np.power(self.system[:,3:6],2)[:,0]*self.mass)*0.239005736*10000+(np.sum(np.dot(self.system[:,0],self.CalculateForces()[:,0])))))/self.volume*68568.415
        pyy = ((np.sum(np.power(self.system[:,3:6],2)[:,1]*self.mass)*0.239005736*10000+(np.sum(np.dot(self.system[:,1],self.CalculateForces()[:,1])))))/self.volume*68568.415
        pzz = ((np.sum(np.power(self.system[:,3:6],2)[:,2]*self.mass)*0.239005736*10000+(np.sum(np.dot(self.system[:,2],self.CalculateForces()[:,2])))))/self.volume*68568.415

        press = (pxx+pyy+pzz)/3
        
        return press

      
    #@jit
    def run(self):
        print('%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s' % ('Timestep','KE','Temp','Press','evdw','ebond','eangle','Time'))
        times = 0
        start = time.time()
        for i in range(0,self.nrun+1): 
            force = self.CalculateForces()/ 48.88821291 /48.88821291 
            self.IncrementalPos(dt*self.system[:,3:6]+(0.5*(dt*dt)*force)/self.mass)
            force_next = self.CalculateForces()/ 48.88821291 /48.88821291
            self.IncrementalVel((0.5*dt*force+0.5*dt*force_next)/self.mass)
            
            end = time.time()
            times = end - start
            #if (i+1) % ndump == 0:
            #    self.SetTemp()

            vdw_e, bond_e, angle_e = self.CalculateEnergy() 
            print('%10d\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f' % (i,self.KineticEnergy(),self.GetTemp(),self.CalPress(),vdw_e, bond_e, angle_e,times))
            
    

    

  