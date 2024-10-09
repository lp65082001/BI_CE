""" Create by Amborse hui from M^5 lab """
'''
to do list
1. all numpy
2. calculate force not use for loop
'''
from scipy.spatial import distance_matrix
import numpy as np
import math
import time

# Molecule dynamics setting (setting from LAMMPS)#
gamma = 3
k = 0.0019872067
cutoff = 20
dt = 4 # might be control
nrun = 1

class MD_Universe: 
    def __init__(self,initial_setting,initial_parameter,mass,temperature=None,pressure=None):
        print("Status: Parameter loading")
        # setting parameter #
        self.mass = mass
        self.mass_rs = math.sqrt(mass)
        self.a = np.array([initial_setting[3][0],0,0])
        self.b = np.array([0,initial_setting[3][1],0])
        self.c = np.array([0,0,initial_setting[3][2]])
        self.a_len = np.linalg.norm(np.array([initial_setting[3][0],0,0]))
        self.b_len = np.linalg.norm(np.array([0,initial_setting[3][1],0]))
        self.c_len = np.linalg.norm(np.array([0,0,initial_setting[3][2]]))
        self.volume = np.prod(initial_setting[3])
        self.bond_init = initial_parameter[0]
        self.bond_energy = initial_parameter[1]
        self.angle_init = initial_parameter[2]
        self.angle_energy = initial_parameter[3]
        self.sigma = initial_parameter[4]
        self.epsilon = initial_parameter[5]
        self.temp = temperature
        self.press = pressure
        self.bond_table = initial_setting[1]
        self.angle_table = initial_setting[2]
        self.bond_dis = []
        self.angle_dis = []
        self.system = np.hstack((initial_setting[0],np.zeros([initial_setting[0].shape[0],3])))

    # ignore 1-2, 1-3 neighbor (need improve and check)#
    def non_bonded_neighbor_list(self):
        atom_list = np.ones((self.system.shape[0],self.system.shape[0]))
        distance_table = distance_matrix(self.system[:,0:3],self.system[:,0:3])
        # consider to PBC #
        distance_table_pbc = self.calculate_pbc_subtract(distance_table) 
        # ignore 1-2, 1-3 #
        atom_list[np.ix_(self.bond_table[:,1],self.bond_table[:,2])] = 0
        atom_list[np.ix_(self.angle_table[:,1],self.angle_table[:,3])] = 0
        close_dist_table = np.where(distance_table_pbc>cutoff)
        atom_list[np.ix_(close_dist_table[0],close_dist_table[1])] = 0
        self.lj_cal = np.where(np.triu(atom_list,1)==1)
        print(self.lj_cal)
        #self.distance_table_pbc = distance_table_pbc

    # calculate pair-distance and subtract a, b, c (need improve and check)#
    def calculate_pbc_subtract(self,distance_table):
        a_ = np.round(np.dot(self.system[:,0],self.a)/self.a_len**2)*self.a
        b_ = np.round(np.dot(self.system[:,1],self.b)/self.b_len**2)*self.b
        c_ = np.round(np.dot(self.system[:,2],self.c)/self.c_len**2)*self.c

        return distance_table - np.dot(a_,a_.T) - np.dot(b_,b_.T) - np.dot(c_,c_.T)

    # set initial velocity #
    def InitVelDis(self):
        self.system[:,3] = np.random.randn(self.system.shape[0])
        self.system[:,4] = np.random.randn(self.system.shape[0])
        self.system[:,5] = np.random.randn(self.system.shape[0])
        
        self.system[:,3] -= np.sum(self.system[:,3])/self.system.shape[0]/self.mass_rs
        self.system[:,4] -= np.sum(self.system[:,4])/self.system.shape[0]/self.mass_rs
        self.system[:,5] -= np.sum(self.system[:,5])/self.system.shape[0]/self.mass_rs
        
        scale = np.power(self.temp/self.GetTemp(),0.5)
        self.system[:,3:6] = np.multiply(self.system[:,3:6],scale)

    # NVT Ensemble (temp_rescale) #
    def SetTemp(self):
        t_s = self.GetTemp() - self.temp
        if (abs(t_s) > 0.02):
            t_t = self.GetTemp() - 0.5*t_s
            scale = np.power(t_t/self.GetTemp(),0.5)
            self.system[:,3:6] = self.system[:,3:6]*scale

    # get temperature #
    def GetTemp(self):
        return (self.mass*np.sum(np.power(self.system[:,3:6],2))*0.239005736*10000)/(gamma*(self.system.shape[0]-1)*k)

    # update position #
    def IncrementalPos(self,increments):
        self.system[:,0:3] = self.system[:,0:3] + increments

    # updata velocity #
    def IncrementalVel(self,increments):
        self.system[:,3:6] = self.system[:,3:6] + increments

    # calculate KE #
    def KineticEnergy(self):
        return 0.5*self.mass*np.sum(np.power(self.system[:,3:6],2))*0.239005736*10000

    # calcaule potential force #
    def CalculateForces(self):
        forces = np.zeros((self.system.shape[0],3))
        '''
        ## nonbond term (origin)##
        for i in range(self.lj_cal[0].shape[0]):
            rij = self.system[self.lj_cal[1][i],0:3] - self.system[self.lj_cal[0][i],0:3]
            rij = rij - round(np.dot(rij,self.a)/self.a_len**2)*self.a - round(np.dot(rij,self.b)/self.b_len**2)*self.b - round(np.dot(rij,self.c)/self.c_len**2)*self.c
            sr6 = pow(self.sigma**2/np.dot(rij,rij),3)
            sr12 = sr6**2
            rij_len = np.power(np.dot(rij,rij),0.5)
            if rij_len < cutoff:
                fij = 4*self.epsilon/rij_len*(-12*sr12+6*sr6)
                forces[self.lj_cal[0][i],:] += fij*rij/rij_len
                forces[self.lj_cal[1][i],:] -= fij*rij/rij_len
        '''
        ## nonbond term (matrix)##
        rij = self.system[self.lj_cal[1],0:3] - self.system[self.lj_cal[0],0:3]
        rij = rij - np.round(np.dot(rij,self.a)/self.a_len**2)*self.a - \
        - np.round(np.dot(rij,self.b)/self.b_len**2)*self.b \
        - np.round(np.dot(rij,self.c)/self.c_len**2)*self.c
        sr6 = np.power(self.sigma**2/np.linalg.norm(rij,axis=1),3)
        sr12 = np.power(sr6,2)
        rij_len = np.linalg.norm(rij,axis=1)
        fij = 4*self.epsilon/rij_len*(-12*sr12+6*sr6)
        forces[self.lj_cal[0],:] += fij*rij/rij_len
        forces[self.lj_cal[1],:] -= fij*rij/rij_len
        '''
        ## bond term (origin)##
        for i in range(0,self.bond_table.shape[0]):
            pair = self.bond_table[i]
            fbond = -2*self.bond_energy*((np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))-self.bond_init)
            forces[pair[1]-1,:] += ((self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])/np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))*fbond
            forces[pair[2]-1,:] -= ((self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])/np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))*fbond
        '''
        ## bond term (matrix)##
        bond_rij = self.system[self.bond_table[1],0:3] - self.system[self.bond_table[2],0:3]
        bond_rij = bond_rij - np.round(np.dot(rij,self.a)/self.a_len**2)*self.a - \
        - np.round(np.dot(rij,self.b)/self.b_len**2)*self.b \
        - np.round(np.dot(rij,self.c)/self.c_len**2)*self.c
        bond_rij_len = np.linalg.norm(bond_rij,axis=1)
        fbond = -2*self.bond_energy*(bond_rij_len-self.bond_init)
        # modify
        forces[self.bond_table[1]-1,:] += ((self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])/np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))*fbond
        forces[self.bond_table[2]-1,:] -= ((self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])/np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))*fbond



        ## angle term ##

        for i in range(0,self.angle_table.shape[0]):
            pair = self.angle_table[i]
            
            # 1st bond length
            delx1 = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[0]
            dely1 = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[1]
            delz1 = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[2]
            r1 = np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])

            # 2rd bond length
            delx2 = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[0]
            dely2 = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[1]
            delz2 = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[2]
            r2 = np.linalg.norm(self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])
            
            # angle

            c = delx1*delx2 + dely1*dely2 + delz1*delz2
            c /= r1*r2

            if (c> 1.0):
                c = 1.0
            elif (c< -1.0):
                c = -1.0
            s = pow(1.0-c*c,0.5)
            if (s< 0.001):
                s = 0.001
            s = 1.0/s

            dtheta = math.acos(c)-math.radians(self.angle_init)
            tk = self.angle_energy * dtheta
            a = -2.0 * tk * s
            a11 = a*c / r1**2
            a12 = -a / (r1*r2)
            a22 = a*c / r2**2

            f1x = a11*delx1 + a12*delx2
            f1y = a11*dely1 + a12*dely2
            f1z = a11*delz1 + a12*delz2
            f3x = a22*delx2 + a12*delx1
            f3y = a22*dely2 + a12*dely1
            f3z = a22*delz2 + a12*delz1

            forces[pair[1]-1,0] += f1x 
            forces[pair[1]-1,1] += f1y
            forces[pair[1]-1,2] += f1z

            forces[pair[2]-1,0] -= f1x + f3x 
            forces[pair[2]-1,1] -= f1y + f3y
            forces[pair[2]-1,2] -= f1z + f3z

            forces[pair[3]-1,0] += f3x 
            forces[pair[3]-1,1] += f3y
            forces[pair[3]-1,2] += f3z

        return forces

    # calcaule non-bond energy #
    def CalculateEnergy(self):
        energy = 0
        for i in range(self.lj_cal[0].shape[0]):
            rij = self.system[self.lj_cal[0][i],0:3] - self.system[self.lj_cal[1][i],0:3]
            rij = rij - round(np.dot(rij,self.a)/self.a_len**2)*self.a - round(np.dot(rij,self.b)/self.b_len**2)*self.b - round(np.dot(rij,self.c)/self.c_len**2)*self.c
                
            sr6 = pow(self.sigma**2/np.dot(rij,rij),3)
            sr12 = sr6**2
            rij_len = np.power(np.dot(rij,rij),2)
            if rij_len < cutoff:
                energy += 4*self.epsilon*(sr12-sr6)
                #print(energy)
        return energy

    # calcaule bond-term energy #
    def CalculateBondEnergy(self):
        bond_energy = 0
        for i in range(0,self.bond_table.shape[0]):
            pair = self.bond_table[i]
            bond_energy += self.bond_energy*((np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))-self.bond_init)**2
            self.bond_dis.append((np.linalg.norm(self.system[pair[1]-1]-self.system[pair[2]-1])))
        return bond_energy
 
    # calcaule angle-term energy #
    def CalculateAngleEnergy(self):
        angle_energy = 0
        for i in range(0,self.angle_table.shape[0]):
            pair = self.angle_table[i]
            
            # 1st bond length
            delx1 = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[0]
            dely1 = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[1]
            delz1 = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[2]
            r1 = np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])

            # 2rd bond length
            delx2 = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[0]
            dely2 = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[1]
            delz2 = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[2]
            r2 = np.linalg.norm(self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])
            
            # angle

            c = delx1*delx2 + dely1*dely2 + delz1*delz2
            c /= r1*r2

            if (c> 1.0):
                c = 1.0
            elif (c< -1.0):
                c = -1.0
            s = pow(1.0-c*c,0.5)
            if (s< 0.001):
                s = 0.001
            s = 1.0/s
            
            dtheta = math.acos(c)-math.radians(self.angle_init)
            angle_energy += self.angle_energy * dtheta * dtheta
            self.angle_dis.append(math.degrees(math.acos(c)))
        return angle_energy

    # calculate pressure # 
    def CalPress(self):     
        pxx = ((np.sum(np.power(self.system[:,3:6],2)[:,0]*self.mass)*0.239005736*10000+(np.sum(np.dot(self.system[:,0],self.CalculateForces()[:,0])))))/self.volume*68568.415
        pyy = ((np.sum(np.power(self.system[:,3:6],2)[:,1]*self.mass)*0.239005736*10000+(np.sum(np.dot(self.system[:,1],self.CalculateForces()[:,1])))))/self.volume*68568.415
        pzz = ((np.sum(np.power(self.system[:,3:6],2)[:,2]*self.mass)*0.239005736*10000+(np.sum(np.dot(self.system[:,2],self.CalculateForces()[:,2])))))/self.volume*68568.415

        press = (pxx+pyy+pzz)/3
        
        return press

    # running process #
    def run(self):
        print("Status: Running (MD)")
        print('%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s' % ('Timestep','KE','Temp','Press','evdw','ebond','eangle'))
        for i in range(nrun+1):  
            self.non_bonded_neighbor_list()
            print('%10d\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f' % (i,self.KineticEnergy(),self.GetTemp(),self.CalPress(),self.CalculateEnergy(),self.CalculateBondEnergy(),self.CalculateAngleEnergy()))
            force = self.CalculateForces()/ 48.88821291 /48.88821291
            self.IncrementalPos(dt*self.system[:,3:6]+(0.5*(dt*dt)*force)/self.mass)
            self.non_bonded_neighbor_list()
            force_next = self.CalculateForces()/ 48.88821291 /48.88821291
            self.IncrementalVel((0.5*dt*force+0.5*dt*force_next)/self.mass)
  