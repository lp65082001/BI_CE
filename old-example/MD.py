import numpy as np
import math
#from numba import jit
import warnings
warnings.filterwarnings("ignore")

class NVT_ensemble:

    def __init__(self,position,bond_table,angle_table,torsion_table,volume,potential,mass,temp,press):
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
        self.torsion_init = potential[4]
        self.torsion_energy = potential[5]
        self.sigma = potential[6]
        self.epsilon = potential[7]
        self.temp = temp
        self.press = press
        self.bond_table = bond_table
        self.angle_table = angle_table
        self.torsion_table = torsion_table
        self.bond_dis = []
        self.angle_dis = []
        self.torsion_dis = []
        self.system = np.hstack((position,np.zeros([position.shape[0],3])))


    def neighbor_list(self):
        atom_list = np.ones((self.system.shape[0],self.system.shape[0]))
        for i in range(0,self.torsion_table.shape[0]):
            pair = self.torsion_table[i]
            atom_list[pair[1]-1:pair[4],pair[1]-1:pair[4]] = 0
        self.lj_cal = np.where(np.triu(atom_list,1)==1)
        #print(atom_list)


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

    # calcaule potential
    #@jit
    def CalculateForces(self):
        forces = np.zeros((self.system.shape[0],3))

        ## nonbond term ##
        
        for i in range(self.lj_cal[0].shape[0]):
            rij = self.system[self.lj_cal[1][i],0:3] - self.system[self.lj_cal[0][i],0:3]
            rij = rij - round(np.dot(rij,self.a)/self.a_len**2)*self.a - round(np.dot(rij,self.b)/self.b_len**2)*self.b - round(np.dot(rij,self.c)/self.c_len**2)*self.c
            sr6 = pow(self.sigma**2/np.dot(rij,rij),3)
            sr12 = sr6**2
            rij_len = math.sqrt(np.dot(rij,rij))
            if rij_len < cutoff:
                fij = 4*self.epsilon/rij_len*(-12*sr12+6*sr6)
                forces[self.lj_cal[0][i],:] += fij*rij/rij_len
                forces[self.lj_cal[1][i],:] -= fij*rij/rij_len

        ## bond term ##
        for i in range(0,self.bond_table.shape[0]):
            pair = self.bond_table[i]
            fbond = -2*self.bond_energy*((np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))-self.bond_init)
            forces[pair[1]-1,:] += ((self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])/np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))*fbond
            forces[pair[2]-1,:] -= ((self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])/np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))*fbond

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

        ## dihedral term ##
        for i in range(0,self.torsion_table.shape[0]):
            pair = self.torsion_table[i]

            # 1st bond length
            vb1x = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[0]
            vb1y = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[1]
            vb1z = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[2]
            rb1 = 1/np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])
            # 2rd bond length
            vb2x = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[0]
            vb2y = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[1]
            vb2z = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[2]
            #rb2 = np.linalg.norm(self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])

            vb2xm = -vb2x
            vb2ym = -vb2y
            vb2zm = -vb2z         

            # 3nd bond length
            vb3x = (self.system[pair[4]-1][0:3]-self.system[pair[3]-1][0:3])[0]
            vb3y = (self.system[pair[4]-1][0:3]-self.system[pair[3]-1][0:3])[1]
            vb3z = (self.system[pair[4]-1][0:3]-self.system[pair[3]-1][0:3])[2]
            rb3 = 1/np.linalg.norm(self.system[pair[4]-1][0:3]-self.system[pair[3]-1][0:3])

            # c0 calculate 
            c0 = (vb1x*vb3x + vb1y*vb3y + vb1z*vb3z) * rb1*rb3
            sb1 = 1.0 / (vb1x*vb1x + vb1y*vb1y + vb1z*vb1z)
            sb2 = 1.0 / (vb2x*vb2x + vb2y*vb2y + vb2z*vb2z)
            sb3 = 1.0 / (vb3x*vb3x + vb3y*vb3y + vb3z*vb3z)
            
            # 1st and 2rd angle
            b1mag2 = vb1x*vb1x + vb1y*vb1y + vb1z*vb1z
            b1mag = pow(b1mag2,0.5)
            b2mag2 = vb2x*vb2x + vb2y*vb2y + vb2z*vb2z
            b2mag = pow(b2mag2,0.5)
            b3mag2 = vb3x*vb3x + vb3y*vb3y + vb3z*vb3z
            b3mag = pow(b3mag2,0.5)

            ctmp = vb1x*vb2x + vb1y*vb2y + vb1z*vb2z
            r12c1 = 1.0 / (b1mag*b2mag)
            c1mag = ctmp * r12c1

            ctmp = vb2xm*vb3x + vb2ym*vb3y + vb2zm*vb3z
            r12c2 = 1.0 / (b2mag*b3mag)
            c2mag = ctmp * r12c2

            # cos or sin
            sin2 = max(1.0 - c1mag*c1mag,0.0)
            sc1 = pow(sin2,0.5)
            if (sc1 < 0.001):
                 sc1 = 0.001
            sc1 = 1.0/sc1

            sin2 = max(1.0 - c2mag*c2mag,0.0)
            sc2 = pow(sin2,0.5)
            if (sc2 < 0.001):
                 sc2 = 0.001
            sc2 = 1.0/sc2

            s1 = sc1 * sc1
            s2 = sc2 * sc2
            s12 = sc1 * sc2
            c = (c0 + c1mag*c2mag) * s12

            cx = vb1y*vb2z - vb1z*vb2y
            cy = vb1z*vb2x - vb1x*vb2z
            cz = vb1x*vb2y - vb1y*vb2x
            cmag = pow(cx*cx + cy*cy + cz*cz,0.5)
            dx = (cx*vb3x + cy*vb3y + cz*vb3z)/cmag/b3mag

            if (c > 1.0):
                c = 1.0
            if (c < -1.0):
                c = -1.0

            phi = math.acos(c)
            if (dx < 0.0):
                 phi *= -1.0
            si = math.sin(phi)

            dphi = phi-math.radians(self.torsion_init)
            p = self.torsion_energy*dphi
            if (abs(si) < 0.00001): 
                pd = - 2.0 * self.torsion_energy
            else:
                pd = - 2.0 * p / si
            
            p = p * dphi
            #print(p)
            a = pd
            c = c * a
            s12 = s12 * a
            a11 = c*sb1*s1
            a22 = -sb2 * (2.0*c0*s12 - c*(s1+s2))
            a33 = c*sb3*s2
            a12 = -r12c1 * (c1mag*c*s1 + c2mag*s12)
            a13 = -rb1*rb3*s12
            a23 = r12c2 * (c2mag*c*s2 + c1mag*s12)

            sx2  = a12*vb1x + a22*vb2x + a23*vb3x
            sy2  = a12*vb1y + a22*vb2y + a23*vb3y
            sz2  = a12*vb1z + a22*vb2z + a23*vb3z

            fd1x = a11*vb1x + a12*vb2x + a13*vb3x
            fd1y = a11*vb1y + a12*vb2y + a13*vb3y
            fd1z = a11*vb1z + a12*vb2z + a13*vb3z

            fd2x = -sx2 - fd1x
            fd2y = -sy2 - fd1y
            fd2z = -sz2 - fd1z

            fd4x = a13*vb1x + a23*vb2x + a33*vb3x
            fd4y = a13*vb1y + a23*vb2y + a33*vb3y
            fd4z = a13*vb1z + a23*vb2z + a33*vb3z

            fd3x = sx2 - fd4x
            fd3y = sy2 - fd4y
            fd3z = sz2 - fd4z

            forces[pair[1]-1,0] += fd1x 
            forces[pair[1]-1,1] += fd1y
            forces[pair[1]-1,2] += fd1z

            forces[pair[2]-1,0] += fd2x 
            forces[pair[2]-1,1] += fd2y
            forces[pair[2]-1,2] += fd2z

            forces[pair[3]-1,0] += fd3x 
            forces[pair[3]-1,1] += fd3y
            forces[pair[3]-1,2] += fd3z

            forces[pair[4]-1,0] += fd4x 
            forces[pair[4]-1,1] += fd4y
            forces[pair[4]-1,2] += fd4z
        #print(forces[:,2])

        return forces

    #@jit
    def CalculateEnergy(self):
        energy = 0
        for i in range(self.lj_cal[0].shape[0]):
            rij = self.system[self.lj_cal[0][i],0:3] - self.system[self.lj_cal[1][i],0:3]
            rij = rij - round(np.dot(rij,self.a)/self.a_len**2)*self.a - round(np.dot(rij,self.b)/self.b_len**2)*self.b - round(np.dot(rij,self.c)/self.c_len**2)*self.c
                
            sr6 = pow(self.sigma**2/np.dot(rij,rij),3)
            sr12 = sr6**2
            rij_len = math.sqrt(np.dot(rij,rij))
            if rij_len < cutoff:
                energy += 4*self.epsilon*(sr12-sr6)
                #print(energy)
        return energy
    #@jit
    def CalculateBondEnergy(self):
        bond_energy = 0
        for i in range(0,self.bond_table.shape[0]):
            pair = self.bond_table[i]
            bond_energy += self.bond_energy*((np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))-self.bond_init)**2
            self.bond_dis.append((np.linalg.norm(self.system[pair[1]-1]-self.system[pair[2]-1])))
        return bond_energy
    
    #@jit
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
    
    #@jit
    def CalculatetorsionEnergy(self):
        torsion_energy = 0
        for i in range(0,self.torsion_table.shape[0]):
            pair = self.torsion_table[i]

            # 1st bond length
            vb1x = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[0]
            vb1y = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[1]
            vb1z = (self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])[2]
            rb1 = 1/np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3])
            # 2rd bond length
            vb2x = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[0]
            vb2y = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[1]
            vb2z = (self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])[2]
            #rb2 = np.linalg.norm(self.system[pair[3]-1][0:3]-self.system[pair[2]-1][0:3])

            vb2xm = -vb2x
            vb2ym = -vb2y
            vb2zm = -vb2z         

            # 3nd bond length
            vb3x = (self.system[pair[4]-1][0:3]-self.system[pair[3]-1][0:3])[0]
            vb3y = (self.system[pair[4]-1][0:3]-self.system[pair[3]-1][0:3])[1]
            vb3z = (self.system[pair[4]-1][0:3]-self.system[pair[3]-1][0:3])[2]
            rb3 = 1/np.linalg.norm(self.system[pair[4]-1][0:3]-self.system[pair[3]-1][0:3])

            # c0 calculate 
            c0 = (vb1x*vb3x + vb1y*vb3y + vb1z*vb3z) * rb1*rb3
            sb1 = 1.0 / (vb1x*vb1x + vb1y*vb1y + vb1z*vb1z)
            sb2 = 1.0 / (vb2x*vb2x + vb2y*vb2y + vb2z*vb2z)
            sb3 = 1.0 / (vb3x*vb3x + vb3y*vb3y + vb3z*vb3z)
            
            # 1st and 2rd angle
            b1mag2 = vb1x*vb1x + vb1y*vb1y + vb1z*vb1z
            b1mag = pow(b1mag2,0.5)
            b2mag2 = vb2x*vb2x + vb2y*vb2y + vb2z*vb2z
            b2mag = pow(b2mag2,0.5)
            b3mag2 = vb3x*vb3x + vb3y*vb3y + vb3z*vb3z
            b3mag = pow(b3mag2,0.5)

            ctmp = vb1x*vb2x + vb1y*vb2y + vb1z*vb2z
            r12c1 = 1.0 / (b1mag*b2mag)
            c1mag = ctmp * r12c1

            ctmp = vb2xm*vb3x + vb2ym*vb3y + vb2zm*vb3z
            r12c2 = 1.0 / (b2mag*b3mag)
            c2mag = ctmp * r12c2

            # cos or sin
            sin2 = max(1.0 - c1mag*c1mag,0.0)
            sc1 = pow(sin2,0.5)
            if (sc1 < 0.001):
                 sc1 = 0.001
            sc1 = 1.0/sc1

            sin2 = max(1.0 - c2mag*c2mag,0.0)
            sc2 = pow(sin2,0.5)
            if (sc2 < 0.001):
                 sc2 = 0.001
            sc2 = 1.0/sc2

            s1 = sc1 * sc1
            s2 = sc2 * sc2
            s12 = sc1 * sc2
            c = (c0 + c1mag*c2mag) * s12

            cx = vb1y*vb2z - vb1z*vb2y
            cy = vb1z*vb2x - vb1x*vb2z
            cz = vb1x*vb2y - vb1y*vb2x
            cmag = pow(cx*cx + cy*cy + cz*cz,0.5)
            dx = (cx*vb3x + cy*vb3y + cz*vb3z)/cmag/b3mag

            if (c > 1.0):
                c = 1.0
            if (c < -1.0):
                c = -1.0

            phi = math.acos(c)
            if (dx < 0.0):
                 phi *= -1.0
            si = math.sin(phi)

            dphi = phi-math.radians(self.torsion_init)
            p = self.torsion_energy*dphi
            if (abs(si) < 0.00001): 
                pd = - 2.0 * self.torsion_energy
            else:
                pd = - 2.0 * p / si
            
            torsion_energy += p * dphi
            self.torsion_dis.append(math.degrees(phi))

        return torsion_energy

    #@jit
    def CalPress(self):     
        pxx = ((np.sum(np.power(self.system[:,3:6],2)[:,0]*self.mass)*0.239005736*10000+(np.sum(np.dot(self.system[:,0],self.CalculateForces()[:,0])))))/self.volume*68568.415
        pyy = ((np.sum(np.power(self.system[:,3:6],2)[:,1]*self.mass)*0.239005736*10000+(np.sum(np.dot(self.system[:,1],self.CalculateForces()[:,1])))))/self.volume*68568.415
        pzz = ((np.sum(np.power(self.system[:,3:6],2)[:,2]*self.mass)*0.239005736*10000+(np.sum(np.dot(self.system[:,2],self.CalculateForces()[:,2])))))/self.volume*68568.415

        press = (pxx+pyy+pzz)/3
        
        return press

      
    #@jit
    def run(self):
        self.neighbor_list()
        print('%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s' % ('Timestep','KE','Temp','Press','evdw','ebond','eangle','edihedral'))
        for i in range(nrun+1):  
            print('%10d\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f' % (i,self.KineticEnergy(),self.GetTemp(),self.CalPress(),self.CalculateEnergy(),self.CalculateBondEnergy(),self.CalculateAngleEnergy(),self.CalculatetorsionEnergy()))
            force = self.CalculateForces()/ 48.88821291 /48.88821291
            self.IncrementalPos(dt*self.system[:,3:6]+(0.5*(dt*dt)*force)/self.mass)
            force_next = self.CalculateForces()/ 48.88821291 /48.88821291
            self.IncrementalVel((0.5*dt*force+0.5*dt*force_next)/self.mass)
            
            #if (i+1) % ndump == 0:
            #    self.SetTemp()
            
    
    


if __name__ == '__main__':
    gamma = 3
    k = 0.0019872067
    dt = 4
    cutoff = 20
    nrun = 100
    ndump = 10
    mass = 28
    temp = 310
    press = 1

    position = np.array([[0,1,0],[0,0,1.2],[0,0,2],[0,1,2]])

    volume = np.array([100,100,100])
    potential = np.array([0.5,10,90,10,90,0.5,1.0,0.2])

    bond_table = np.array([[1,1,2],[1,2,3],[1,3,4]])
    angle_table = np.array([[1,1,2,3],[1,2,3,4]])
    torsion_table = np.array([[1,1,2,3,4]])

    MD_test = NVT_ensemble(position,bond_table,angle_table,torsion_table,volume,potential,mass,temp,press)
    #MD_test.InitVelDis()
    
    #MD_test.neighbor_list()
    MD_test.run()


    

  