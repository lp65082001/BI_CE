import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis import transformations
from matplotlib import pyplot as plt
import numpy as np
import gc
import math 
import multiprocessing as mp
import threading
from tqdm import tqdm
from collections import OrderedDict
import time
import warnings
import copy

warnings.filterwarnings("ignore")
plt.style.use('dark_background')


class Iteractive_boltzmann_inversion:
    
    def __init__(self,data):
        self.info = data
        self.bead_mass = data.select_atoms('all').total_mass()/len(set(data.select_atoms('all').segids))/len(set(data.select_atoms('all').resids))
        self.res = len(set(data.select_atoms('all').resids))
        self.sig = len(set(data.select_atoms('all').segids))
        self.frame = data.trajectory.n_frames
        self.box = data.dimensions
        self.bins_num = 101
        self.max_edge = 10 
        self.path = "./test"
        print("===Initialized successfully===")       

    def gaussian(self,x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def generate_position(self):
        all_list = np.zeros((self.frame,self.res*self.sig,3))
        print("center of mass (position)") 
        with tqdm(total=self.frame) as pbar:
            for i in range(0,self.frame):
                self.info.trajectory[i]
                atom_list = np.array([])
                for j in range(1,self.sig+1):
                    for k in range(1,self.res+1):
                        centerofmass_cg = self.info.select_atoms('segid P'+str(j)+' and resid ' + str(k))
                        atom_list = np.append(atom_list,centerofmass_cg.center_of_mass())
                atom_list = atom_list.reshape((-1,3)) 
                all_list[i][:][:] = atom_list
                pbar.update(1)
        all_list = np.array(all_list)
        self.list_position = all_list
        self.position =  np.mean(all_list,axis=0)

    def pairCorrelationFunction_3D(self):
        structure = mda.Universe.empty(self.sig*self.res,trajectory=True)
        structure.atoms.positions = self.position
        transform = transformations.boxdimensions.set_dimensions(self.box)
        structure.trajectory.add_transformations(transform)

        all_rdf = np.zeros([100,1])
        he = 0
        ed = self.res-1
        print("Radius density function")
        with tqdm(total=self.sig) as pbar:
            for i in range(0,self.sig):
                c1 = structure.select_atoms('index '+str(he+self.sig*i)+' to '+str(ed+self.sig*i))
                cn = structure.select_atoms('all and not index '+str(he+self.sig*i)+' to '+str(ed+self.sig*i))
                ss_rdf = InterRDF(c1,cn,nbins=100,range=(0.0, 10.0))
                ss_rdf.run()

                all_rdf += np.array(ss_rdf.rdf).reshape((100,1))
                pbar.update(1)
            
        self.rdf_m = (all_rdf/self.sig).reshape(-1,1)

    def bond_list(self,x):
        y = []
        for i in range(0,x.shape[0]-1):
            y.append(np.linalg.norm(x[i+1]-x[i]))
        return y   

    def angle_list(self,x):
        y = []
        for i in range(0,x.shape[0]-2):
            point_1 = x[i+0]
            point_2 = x[i+1]
            point_3 = x[i+2]
            a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
            b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
            c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))

            y.append(math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c))))

        return y  
    
    def cal_distribution(self):
        self.generate_position()
        self.pairCorrelationFunction_3D()
        bond_all = []
        angle_all = []
        torsion_all = []
        print("Bond and Angle")
        with tqdm(total=self.frame) as pbar:
            for k in range(0,self.frame):
                for j in range(0,self.sig):
                    bond_all.append(self.bond_list(self.list_position[k][j*self.res:(j+1)*self.res,:])) 
                    angle_all.append(self.angle_list(self.list_position[k][j*self.res:(j+1)*self.res,:])) 
                pbar.update(1)    
        b_a = np.array(bond_all).reshape((-1,1))
        a_a = np.array(angle_all).reshape((-1,1))
        t_a = np.array(torsion_all).reshape((-1,1))
        b_d = self.gaussian(b_a,np.mean(b_a),np.std(b_a))
        a_d = self.gaussian(a_a,np.mean(a_a),np.std(a_a))


        self.b_a1 = b_a
        self.a_a1 = a_a
        self.b_d1 = b_d
        self.a_d1 = a_d


        Kb = 0.008314
        T = 300

        ### bond, angle ###
        bond_energy = -Kb*T*np.log(b_d)/4.2
        bond_energy = bond_energy + abs(min(bond_energy)) 
        angle_energy = -Kb*T*np.log(a_d)/4.2 
        angle_energy = angle_energy - abs(min(angle_energy))
        #torsion_energy = -Kb*T*np.log(t_d)/4.2 
        #torsion_energy = torsion_energy - abs(min(torsion_energy))
        

        fit_a = -1

        bond_init = b_a[np.where(bond_energy == min(bond_energy))[0][0]][0]
        angle_init = a_a[np.where(angle_energy == min(angle_energy))[0][0]][0]

        bond_init2 = b_a[np.where(bond_energy == np.sort(bond_energy,axis=0)[fit_a])[0][0]][0]
        angle_init2 = a_a[np.where(angle_energy == np.sort(angle_energy,axis=0)[fit_a])[0][0]][0]
        
        bond_k = (np.sort(bond_energy,axis=0)[fit_a])/(bond_init2-bond_init)**2
        angle_k = (np.sort(angle_energy,axis=0)[fit_a])/(angle_init2-angle_init)**2
       
        x_b = np.linspace(0,3,100)
        x_a = np.linspace(0,180,500)

        y_b = bond_k*(x_b-bond_init)**2
        y_a = angle_k*(x_a-angle_init)**2



        ### pair potential ###
        rdf_energy = -Kb*T*np.log(self.rdf_m)/4.2
        sigma = np.linspace(0,10,100)[np.where(rdf_energy == min(rdf_energy))[0][0]]/(2**(1/6))
        epsilon = abs(min(rdf_energy))
        min_eps = np.linspace(0,10,100)[np.where(rdf_energy == min(rdf_energy))[0][0]]
        x_r = np.linspace(3.5,10,100)
        Uvdw = 4*epsilon*((sigma/x_r)**12-(sigma/x_r)**6)

        self.bondi = bond_init
        self.bondk = bond_k[0]
        self.anglei = angle_init
        self.anglek = angle_k[0]*180**2/math.pi**2
        self.epsilon_ = epsilon[0]
        self.sigma_ = sigma
        print("done")

    def get_ref_fa(self):
        return [self.b_a1,self.b_d1,self.a_a1,self.a_d1], self.rdf_m

    def get_init_parameter(self):
        return [self.bondi, self.bondk,self.anglei,self.anglek,self.epsilon_,self.sigma_], [self.bead_mass,self.sig,self.res,self.box] 

    def get_position(self):
        return self.position

    
