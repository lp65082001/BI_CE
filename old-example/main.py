import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.transformations.boxdimensions import set_dimensions
from matplotlib import pyplot as plt
import numpy as np
import gc
import math 
import multiprocessing as mp
from os import getpid
import threading
from numba import jit
from numba.experimental import jitclass
from collections import OrderedDict
from numba import int32, float32, int64,float64   # import the types
from scipy.spatial import distance_matrix
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
        self.box = psf.dimensions
        self.bins_num = 101
        self.max_edge = 10 
        self.path = "./test"
        print("===Initialized successfully===\n")       

    def generate_position(self,job):
        all_list = np.zeros((self.frame,self.res*self.sig,3))   
        bei, eni = self.time_div(job)

        for i in range(bei,eni):
            self.info.trajectory[i]
            atom_list = np.array([])
            for j in range(1,self.sig+1):
                for k in range(1,self.res+1):
                    centerofmass_cg = self.info.select_atoms('segid P'+str(j)+' and resid ' + str(k))
                    atom_list = np.append(atom_list,centerofmass_cg.center_of_mass())
            atom_list = atom_list.reshape((-1,3)) 
            all_list[i][:][:] = atom_list
        #print("process "+str(job)+" done!") 
        return all_list

    def build_universe(self):
        self.cg_info = mda.Universe(self.path+".xyz")
        transform = mda.transformations.boxdimensions.set_dimensions(self.box)
        self.cg_info.trajectory.add_transformations(transform)


    def pairCorrelationFunction_3D(self,job):
        bei, eni = self.sig_div(job)
        all_rdf = np.zeros([100,1])
        he = 0
        ed = self.res-1
        for i in range(bei, eni):
            c1 = self.cg_info.select_atoms('index '+str(he)+' to '+str(ed))
            cn = self.cg_info.select_atoms('all and not index '+str(he)+' to '+str(ed))
            ss_rdf = InterRDF(c1,cn,nbins=100,range=(0.0, 10.0))
            ss_rdf.run()

            all_rdf += np.array(ss_rdf.rdf).reshape((100,1))
            
            he += self.res
            ed += self.res
        return all_rdf


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


    def torsion_list(self,x):
        y = []
        for i in range(0,x.shape[0]-3):
            point_1 = x[i+0]
            point_2 = x[i+1]
            point_3 = x[i+2]
            point_4 = x[i+3]
            
            v12 = point_2 - point_1
            v23 = point_2 - point_3
            v34 = point_3 - point_4

            vec1 = np.cross(v12,v23)
            vec2 = np.cross(v23,v34)
            cosphi = (np.dot(vec1,vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

            y.append(math.degrees(math.acos(cosphi)))
        
        return y



    def gaussian(self,x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def multi_process(self,task):
        num_process = 4
        start = time.time()
        pool = mp.Pool(processes=num_process)
        result_list = pool.map(task,[0,1,2,3])
        end = time.time()
        print("Cost: "+str(end - start)) 
        return result_list

    def time_div(self,num):
        neighbor = int(self.frame/4)
        be = 0
        if (num==0):
            en = be+neighbor
            #en = 1
        elif (num==1):
            be = be+neighbor
            en = be+neighbor
        elif (num == 2):
            be = be+neighbor+neighbor
            en = be+neighbor
        elif (num == 3):
            be = be+neighbor+neighbor+neighbor
            en = self.frame
        return be, en


    def sig_div(self,num):
        neighbor = int(self.sig/4)
        be = 0
        if (num==0):
            en = be+neighbor
            #en = 1
        elif (num==1):
            be = be+neighbor
            en = be+neighbor
        elif (num == 2):
            be = be+neighbor+neighbor
            en = be+neighbor
        elif (num == 3):
            be = be+neighbor+neighbor+neighbor
            en = self.sig
        return be, en
    
    
    def mapping(self):
        print("=====mpi-process-position=====\n")
        self.list_position = np.sum(self.multi_process(self.generate_position),axis=0)
        self.FA_mapping_xyz()
        self.build_universe()
        print("=========== Done! ============\n")
        
        


    def cal_distribution(self,figg = -1):
        bond_all = []
        angle_all = []
        torsion_all = []
        for k in range(0,self.frame):
            for j in range(0,self.sig):
                bond_all.append(self.bond_list(self.list_position[k][j*60:(j+1)*60,:])) 
                angle_all.append(self.angle_list(self.list_position[k][j*60:(j+1)*60,:])) 
                torsion_all.append(self.torsion_list(self.list_position[k][j*60:(j+1)*60,:])) 
        b_a = np.array(bond_all).reshape((-1,1))
        a_a = np.array(angle_all).reshape((-1,1))
        t_a = np.array(torsion_all).reshape((-1,1))
        b_d = self.gaussian(b_a,np.mean(b_a),np.std(b_a))
        a_d = self.gaussian(a_a,np.mean(a_a),np.std(a_a))
        t_d = self.gaussian(t_a,np.mean(t_a),np.std(t_a))

        self.b_a1 = b_a
        self.a_a1 = a_a
        self.t_a1 = t_a
        self.b_d1 = b_d
        self.a_d1 = a_d
        self.t_d1 = t_d


        print("========mpi-process-rdf=======\n")
        rdf = np.sum(np.array(self.multi_process(self.pairCorrelationFunction_3D)).reshape((4,100)),axis=0)
        self.rdf_m = rdf/self.sig

        print("=========== Done! ============\n")

        Kb = 0.008314
        T = 300

        ### bond, angle ###
        bond_energy = -Kb*T*np.log(b_d)/4.2
        bond_energy = bond_energy + abs(min(bond_energy)) 
        angle_energy = -Kb*T*np.log(a_d)/4.2 
        angle_energy = angle_energy - abs(min(angle_energy))
        torsion_energy = -Kb*T*np.log(t_d)/4.2 
        torsion_energy = torsion_energy - abs(min(torsion_energy))
        

        fit_a = -1

        bond_init = b_a[np.where(bond_energy == min(bond_energy))[0][0]][0]
        angle_init = a_a[np.where(angle_energy == min(angle_energy))[0][0]][0]
        torsion_init = t_a[np.where(torsion_energy == min(torsion_energy))[0][0]][0]

        bond_init2 = b_a[np.where(bond_energy == np.sort(bond_energy,axis=0)[fit_a])[0][0]][0]
        angle_init2 = a_a[np.where(angle_energy == np.sort(angle_energy,axis=0)[fit_a])[0][0]][0]
        torsion_init2 = t_a[np.where(torsion_energy == np.sort(torsion_energy,axis=0)[fit_a])[0][0]][0]

        bond_k = (np.sort(bond_energy,axis=0)[fit_a])/(bond_init2-bond_init)**2
        angle_k = (np.sort(angle_energy,axis=0)[fit_a])/(angle_init2-angle_init)**2
        torsion_k = (np.sort(torsion_energy,axis=0)[fit_a])/(torsion_init2-torsion_init)**2

        x_b = np.linspace(0,3,100)
        x_a = np.linspace(0,180,500)

        y_b = bond_k*(x_b-bond_init)**2
        y_a = angle_k*(x_a-angle_init)**2
        y_t = torsion_k*(x_a-torsion_init)**2
        

        ### pair potential ###
        rdf_energy = -Kb*T*np.log(self.rdf_m)/4.2
        sigma = np.linspace(0,10,100)[np.where(rdf_energy == min(rdf_energy))]/(2**(1/6))
        epsilon = abs(min(rdf_energy))
        min_eps = np.linspace(0,10,100)[np.where(rdf_energy == min(rdf_energy))]
        x_r = np.linspace(3.5,10,100)
        Uvdw = 4*epsilon*((sigma/x_r)**12-(sigma/x_r)**6)

        self.bondi = bond_init
        self.bondk = bond_k[0]
        self.anglei = angle_init
        self.anglek = angle_k
        self.epsilon_ = epsilon
        self.sigma_ = sigma[0]
        self.torsionk = torsion_k
        self.torsioni = torsion_init

        ### plot figure ###
        if (figg != -1):

            ### print value ###

            print("bond_init: "+str(bond_init))
            print("bond_k: "+str(bond_k[0]))

            print("angle_init: "+str(angle_init))
            print("angle_k: "+str(angle_k[0]*(180/3.14)**2))

            print("torsion_init: "+str(torsion_init))
            print("torsion_k: "+str(torsion_k[0]*(180/3.14)**2))

            print("epsilon: "+str(epsilon))
            print("sigma: "+str(sigma[0]))

            fig, axes = plt.subplots(figsize = (12, 12), nrows = 4, ncols = 2)  

            axes[0][0].scatter(b_a,b_d,marker = ".",label = "bond distribution")
            axes[0][0].set_xlabel("bond length")
            axes[0][0].set_ylabel("distributions")
            axes[0][0].legend()

            axes[0][1].scatter(b_a,bond_energy,marker = ".",label="potential distribution")
            axes[0][1].plot(x_b,y_b,"r-",label="L0: {:.2f} \nKbond: {:.2f}".format(bond_init,bond_k[0]))

            axes[0][1].set_xlabel("bond length")
            axes[0][1].set_ylabel("potential energy")
            axes[0][1].legend()

            axes[1][0].scatter(a_a,a_d,marker = ".",label = "angle distribution")
            axes[1][0].set_xlabel("angle")
            axes[1][0].set_ylabel("distributions")
            axes[1][0].legend()

            axes[1][1].scatter(a_a,angle_energy,marker = ".",label="potential distribution")
            axes[1][1].plot(x_a,y_a,"r-",label="θ0: {:.2f} \nKangle: {:.2f}".format(angle_init,angle_k[0]*(180/3.14)**2))

            axes[1][1].set_xlabel("angle")
            axes[1][1].set_ylabel("potential energy")
            axes[1][1].legend()

            axes[2][0].plot(np.linspace(0,10,100),self.rdf_m,label = "RDF")
            axes[2][0].set_ylabel("g(r)")
            axes[2][0].set_xlabel("distance")
            axes[2][0].legend(loc = "upper left")

            axes[2][1].plot(np.linspace(0,10,100),rdf_energy,label="potential distribution")
            axes[2][1].plot(x_r,Uvdw,"r-",label="sigma: {:.2f} \nepsilon: {:.2f}".format(sigma[0],epsilon))

            axes[2][1].set_xlabel("distance")
            axes[2][1].set_ylabel("potential (kcal)")
            axes[2][1].set_ylim((-0.5,1.4))
            axes[2][1].legend()

            
            axes[3][0].scatter(t_a,t_d,marker = ".",label = "torsion angle distribution")
            axes[3][0].set_xlabel("torison angle")
            axes[3][0].set_ylabel("distributions")
            axes[3][0].legend()
            
            axes[3][1].scatter(t_a,torsion_energy,marker = ".",label="potential distribution")
            axes[3][1].plot(x_a,y_t,"r-",label="θ0: {:.2f} \nKtorsion: {:.2f}".format(torsion_init,torsion_k[0]*(180/3.14)**2))

            axes[3][1].set_xlabel("torsion angle")
            axes[3][1].set_ylabel("potential energy")
            axes[3][1].legend()

            fig.savefig('./initial.png')
    
    def xyz2data(self):
        #header#
        f = open(self.path+".data", 'w')
        f.write("create by Amborse hui from M^5 lab\n")
        f.write("\n")
        f.write(str(self.res*self.sig)+" atoms\n")
        f.write("1 atom types\n")
        f.write(str(self.sig*(self.res-1))+" bonds\n")
        f.write("1 bond types\n")
        f.write(str(self.sig*(self.res-2))+" angles\n")
        f.write("1 angle types\n")
        f.write(str(self.sig*(self.res-3))+" dihedrals\n")
        f.write("1 dihedral types\n")
        f.write("0 impropers\n")
        f.write("0 improper types\n")
        f.write("\n")
        f.write("0.0 "+str(self.box[0])+" xlo xhi\n")
        f.write("0.0 "+str(self.box[1])+" ylo yhi\n") 
        f.write("0.0 "+str(self.box[2])+" zlo zhi\n")  
        f.write("\n")
        f.write("Masses\n")
        f.write("\n")
        f.write("1 "+str(self.bead_mass)+" \n")
        f.write("\n")
        f.write("Pair Coeffs # lj/cut\n")
        f.write("\n")
        f.write("1 "+str(self.epsilon_)+" "+str(self.sigma_)+"\n")
        f.write("\n")
        f.write("Bond Coeffs # harmonic\n")
        f.write("\n")
        f.write("1 "+str(self.bondk)+" "+str(self.bondi)+"\n")
        f.write("\n")
        f.write("Angle Coeffs # harmonic\n")
        f.write("\n")
        f.write("1 "+str(self.anglek[0])+" "+str(self.anglei)+"\n")
        f.write("\n")
        f.write("Dihedral Coeffs # quadratic\n")
        f.write("\n")
        f.write("1 "+str(self.torsionk[0])+" "+str(self.torsioni)+"\n")
        f.write("\n")

        # Atom list
        f.write("Atoms #molecule \n")
        f.write("\n")
        for i in range(0,self.list_position.shape[1]):
            f.write(str(i+1)+" 1 1 "+str(self.get_pos()[-1][i][0])+" "+str(self.get_pos()[-1][i][1])+" "+str(self.get_pos()[-1][i][2])+" 0 0 0\n")
        f.write("\n")

        # Bond list
        f.write("Bonds\n")
        f.write("\n")
        n = 1
        for j in range(0,self.sig):
            for i in range(0,self.res-1):
                f.write(str(n)+" 1 "+str(i+1+j*(self.res))+" "+str(i+2+j*(self.res))+"\n")
                n += 1
        f.write("\n")

        # angle list
        f.write("Angles\n")
        f.write("\n")
        n = 1
        for j in range(0,self.sig):
            for i in range(0,self.res-2):
                f.write(str(n)+" 1 "+str(i+1+j*(self.res))+" "+str(i+2+j*(self.res))+" "+str(i+3+j*(self.res))+"\n")
                n += 1
        f.write("\n")

        # dihedral list
        f.write("Dihedrals\n")
        f.write("\n")
        n = 1
        for j in range(0,self.sig):
            for i in range(0,self.res-3):
                f.write(str(n)+" 1 "+str(i+1+j*(self.res))+" "+str(i+2+j*(self.res))+" "+str(i+3+j*(self.res))+" "+str(i+4+j*(self.res))+"\n")
                n += 1
        f.write("\n")
        f.close()
            
    
    def get_pos(self):
        return self.list_position

    def get_rdf(self):
        return  self.rdf_m

    def get_dis(self):
        return np.array([self.b_a1,self.a_a1,self.t_a1,self.b_d1,self.a_d1,self.t_d1])
    
    def FA_mapping_xyz(self):
        name = self.path+".xyz"
        f = open(name, 'w')
        for k in range(0,self.frame):
            f.write(str(self.res*self.sig)+"\n")
            f.write(str(k)+"\n")
            for z in self.list_position[k][:][:]:
                f.write("1 "+str(z[0])+" "+str(z[1])+" "+str(z[2])+"\n")   
        f.close()

    def plot_rdf(self):
        plt.plot(np.linspace(0,10,100),self.list_rdf)
        plt.show()

    def get_parameter(self):
        position = []
        bond_l= []
        angle_l = []
        dihedral_l = []

        # Atom list

        for i in range(0,self.list_position.shape[1]):
            position.append([(self.get_pos()[-1][i][0]),(self.get_pos()[-1][i][1]),(self.get_pos()[-1][i][2])])


        # Bond list

        n = 1
        for j in range(0,self.sig):
            for i in range(0,self.res-1):
                bond_l.append([n,(i+1+j*(self.res)),(i+2+j*(self.res))])
                n += 1


        # angle list

        for j in range(0,self.sig):
            for i in range(0,self.res-2):
                angle_l.append([n,(i+1+j*(self.res)),(i+2+j*(self.res)),(i+3+j*(self.res))])
                n += 1


        # dihedral list

        n = 1
        for j in range(0,self.sig):
            for i in range(0,self.res-3):
                dihedral_l.append([n,(i+1+j*(self.res)),(i+2+j*(self.res)),(i+3+j*(self.res)),(i+4+j*(self.res))])
                n += 1

        volume = np.array([self.box[0],self.box[1],self.box[2]]).astype("float64")
        pot = np.array([self.bondi,self.bondk,self.anglei,self.anglek,self.torsioni,self.torsionk,self.sigma_,self.epsilon_]).astype("float64")
        
        position = np.array(position,dtype = "float64").reshape((-1,3))
        position = np.hstack((position,np.zeros([position.shape[0],3],dtype = "float64"))).astype("float64")
        bond_l = np.array(bond_l,dtype = "int64").reshape((-1,3))
        angle_l = np.array(angle_l,dtype = "int64").reshape((-1,4))
        dihedral_l = np.array(dihedral_l,dtype = "int64").reshape((-1,5))
        
        atom_list = np.ones((position.shape[0],position.shape[0]))
        for i in range(0,dihedral_l.shape[0]):
            pair = dihedral_l[i]
            atom_list[pair[1]-1:pair[4],pair[1]-1:pair[4]] = 0
        #print(np.where(np.triu(atom_list,1)==1)[0].shape)
        a11 = np.where(np.triu(atom_list,1)==1)[0].astype("int64")
        a12 = np.where(np.triu(atom_list,1)==1)[1].astype("int64")
        nb = np.array([a11,a12]).astype("int64")

        return [position,bond_l,angle_l,dihedral_l,volume,pot,nb]

class compare_iter:

    def __init__(self,data1,data2):
        self.d1_rdf = data1.get_rdf()
        self.d1_dis = data1.get_dis()
        self.d2_rdf = data2.get_rdf()
        self.d2_dis = data2.get_dis()  
        print("========== compare ===========\n") 
    def compare_plot(self):
        fig, axes = plt.subplots(figsize = (12, 12), nrows = 2, ncols = 2)  

        axes[0][0].scatter(self.d1_dis[0],self.d1_dis[3],marker = ".",label = "bond distribution (initial)")
        axes[0][0].scatter(self.d2_dis[0],self.d2_dis[3],marker = ".",label = "bond distribution (latest)")
        axes[0][0].set_xlabel("bond length")
        axes[0][0].set_ylabel("distributions")
        axes[0][0].legend()

        axes[0][1].scatter(self.d1_dis[1],self.d1_dis[4],marker = ".",label = "angle distribution (intitial)") 
        axes[0][1].scatter(self.d2_dis[1],self.d2_dis[4],marker = ".",label = "angle distribution (latest)")
        axes[0][1].set_xlabel("angle")
        axes[0][1].set_ylabel("distributions")
        axes[0][1].legend()

        axes[1][0].plot(np.linspace(0,10,100),self.d1_rdf,label = "RDF (initial)")
        axes[1][0].plot(np.linspace(0,10,100),self.d2_rdf,label = "RDF (latest)")
        axes[1][0].set_ylabel("g(r)")
        axes[1][0].set_xlabel("distance")
        axes[1][0].legend(loc = "upper left")
            
        axes[1][1].scatter(self.d1_dis[2],self.d1_dis[5],marker = ".",label = "torsion angle distribution (initial)")
        axes[1][1].scatter(self.d2_dis[2],self.d2_dis[5],marker = ".",label = "torsion angle distribution (latest)")
        axes[1][1].set_xlabel("torison angle")
        axes[1][1].set_ylabel("distributions")
        axes[1][1].legend()
            

        fig.savefig('./compare.png')
        print("=========== Done! ============\n")

class xyz_dis:

    def __init__(self,data,res_num,sig_num):
        self.info = data
        self.res = res_num
        self.sig = sig_num
        self.frame = data.trajectory.n_frames
        self.bins_num = 101
        self.max_edge = 10 
        print("===Initialized successfully===\n")

    @jit
    def pairCorrelationFunction_3D(self,job):
        bei, eni = self.sig_div(job)
        all_rdf = np.zeros([100,1])
        he = 0
        ed = self.res-1
        for i in range(bei, eni):
            c1 = self.info.select_atoms('index '+str(he)+' to '+str(ed))
            cn = self.info.select_atoms('all and not index '+str(he)+' to '+str(ed))
            ss_rdf = InterRDF(c1,cn,nbins=100,range=(0.0, 10.0))
            ss_rdf.run()

            all_rdf += np.array(ss_rdf.rdf).reshape((100,1))
            
            he += self.res
            ed += self.res
        return all_rdf

    @jit
    def bond_list(self,x):
        y = []
        for i in range(0,x.shape[0]-1):
            y.append(np.linalg.norm(x[i+1]-x[i]))
        return y   
    @jit
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

    @jit
    def torsion_list(self,x):
        y = []
        for i in range(0,x.shape[0]-3):
            point_1 = x[i+0]
            point_2 = x[i+1]
            point_3 = x[i+2]
            point_4 = x[i+3]
            
            v12 = point_2 - point_1
            v23 = point_2 - point_3
            v34 = point_3 - point_4

            vec1 = np.cross(v12,v23)
            vec2 = np.cross(v23,v34)
            cosphi = (np.dot(vec1,vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

            y.append(math.degrees(math.acos(cosphi)))
        
        return y


    @jit
    def gaussian(self,x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def multi_process(self,task):
        num_process = 4
        start = time.time()
        pool = mp.Pool(processes=num_process)
        result_list = pool.map(task,[0,1,2,3])
        end = time.time()
        print("Cost: "+str(end - start)) 
        return result_list
    @jit
    def time_div(self,num):
        neighbor = int(self.frame/4)
        be = 0
        if (num==0):
            en = be+neighbor
            #en = 1
        elif (num==1):
            be = be+neighbor
            en = be+neighbor
        elif (num == 2):
            be = be+neighbor+neighbor
            en = be+neighbor
        elif (num == 3):
            be = be+neighbor+neighbor+neighbor
            en = self.frame
        return be, en
    @jit
    def sig_div(self,num):
        neighbor = int(self.sig/4)
        be = 0
        if (num==0):
            en = be+neighbor
            #en = 1
        elif (num==1):
            be = be+neighbor
            en = be+neighbor
        elif (num == 2):
            be = be+neighbor+neighbor
            en = be+neighbor
        elif (num == 3):
            be = be+neighbor+neighbor+neighbor
            en = self.sig
        return be, en


    def IBM_tr_cg(self):

        bond_all = []
        angle_all = []
        torsion_all = []
        for k in range(0,self.frame):
            self.info.trajectory[k]
            num = 0
            for j in range(1,self.sig+1):
                atom_list = np.array([])
                for i in range(0,self.res):
                    cg = self.info.select_atoms("index {}".format(num+i))
                    #print(cg[0])
                    #print(cg.positions())
                    atom_list = np.append(atom_list,cg.positions)
                    #print(centerofmass_cg.center_of_mass())   
                atom_list = atom_list.reshape((-1,3)) 
                #print(atom_list)
                bond_all.append(self.bond_list(atom_list)) 
                angle_all.append(self.angle_list(atom_list)) 
                torsion_all.append(self.torsion_list(atom_list))
                num += self.res
                 
        gc.collect()
        bond_all = np.array(bond_all).reshape((-1,1))
        angle_all = np.array(angle_all).reshape((-1,1))
        torsion_all = np.array(torsion_all).reshape((-1,1))
        bond_new = self.gaussian(bond_all,np.mean(bond_all),np.std(bond_all))
        angle_new = self.gaussian(angle_all,np.mean(angle_all),np.std(angle_all))
        torsion_new = self.gaussian(torsion_all,np.mean(torsion_all),np.std(torsion_all))  

        self.b_a1 = bond_all
        self.a_a1 = angle_all
        self.t_a1 = torsion_all
        self.b_d1 = bond_new
        self.a_d1 = angle_new
        self.t_d1 = torsion_new

        print("========mpi-process-rdf=======\n")
        rdf = np.sum(np.array(self.multi_process(self.pairCorrelationFunction_3D)).reshape((4,100)),axis=0)
        self.rdf_m = rdf/self.sig

        print("=========== Done! ============\n")

    def get_rdf(self):
        return  self.rdf_m


#### Molecule dynamics ###

gamma = 3
k = 0.0019872067
dt = 4
cutoff = 20
nrun = 1000
ndump = 10


@jitclass( 
    OrderedDict([ 
        ('position', float64[:,:]), 
        ('bond_table', float64[:,:]),
        ('angle_table',float64[:,:]),
        ('torsion_table',float64[:,:]),
        ('volume',float64[:]),
        ('potential',float64[:]),
        ('mass',float64),
        ('temp',float64),
        ('press',float64),
        ('a',float64[:]),
        ('b',float64[:]),
        ('c',float64[:]),
        ('a_len',float64),
        ('b_len',float64),
        ('c_len',float64),
        ('volume2',float64),
        ('bond_init',float64),
        ('bond_energy',float64),
        ('angle_init',float64),
        ('angle_energy',float64),
        ('torsion_init',float64),
        ('torsion_energy',float64),
        ('sigma',float64),
        ('epsilon',float64),
        ('temp',float64),
        ('press',float64),
        ('bond_table',int64[:,:]),
        ('angle_table',int64[:,:]),
        ('torsion_table',int64[:,:]),
        ('system',float64[:,:]),
        ('lj_cal',int64[:,:]),
    ]) 
)

class Ensemble:

    def __init__(self,position,bond_table,angle_table,torsion_table,volume,potential,mass,temp,press,nb):
        self.mass = mass
        self.a = np.array([volume[0],0,0]).astype("float64")
        self.b = np.array([0,volume[1],0]).astype("float64")
        self.c = np.array([0,0,volume[2]]).astype("float64")
        self.a_len = np.float64(math.sqrt(np.dot(np.array([volume[0],0,0]),np.array([volume[0],0,0]))))
        self.b_len = np.float64(math.sqrt(np.dot(np.array([0,volume[1],0]),np.array([0,volume[1],0]))))
        self.c_len = np.float64(math.sqrt(np.dot(np.array([0,0,volume[2]]),np.array([0,0,volume[2]]))))
        self.volume2 = np.float64(volume[0]*volume[1]*volume[2])
        self.bond_init = np.float64(potential[0])
        self.bond_energy = np.float64(potential[1])
        self.angle_init = np.float64(potential[2])
        self.angle_energy = np.float64(potential[3])
        self.torsion_init = np.float64(potential[4])
        self.torsion_energy = np.float64(potential[5])
        self.sigma = np.float64(potential[6])
        self.epsilon = np.float64(potential[7])
        self.temp = np.float64(temp)
        self.press = np.float64(press)
        self.bond_table = bond_table.astype("int64")
        self.angle_table = angle_table.astype("int64")
        self.torsion_table = torsion_table.astype("int64")
        #self.bond_dis = np.zeros(0,dtype = "float64")
        #self.angle_dis = np.zeros(0,dtype = "float64")
        #self.torsion_dis = np.zeros(0,dtype = "float64")
        self.system = position
        self.lj_cal = nb
        #self.lj_cal = np.zeros((0,0),dtype = "int32")

    def InitVelDis(self):
        self.system[:,3] = np.random.randn(self.system.shape[0])
        self.system[:,4] = np.random.randn(self.system.shape[0])
        self.system[:,5] = np.random.randn(self.system.shape[0])
        
        self.system[:,3] -= np.sum(self.system[:,3])/self.system.shape[0]/math.sqrt(self.mass)
        self.system[:,4] -= np.sum(self.system[:,4])/self.system.shape[0]/math.sqrt(self.mass)
        self.system[:,5] -= np.sum(self.system[:,5])/self.system.shape[0]/math.sqrt(self.mass)
        
        scale = math.sqrt(self.temp/self.GetTemp())
        self.system[:,3:6] = np.multiply(self.system[:,3:6],scale)
    #@property
    def SetTemp(self):
        t_s = self.GetTemp() - self.temp
        if (abs(t_s) > 0.02):
            t_t = self.GetTemp() - 0.5*t_s
            scale = math.sqrt(t_t/self.GetTemp())
            self.system[:,3:6] = self.system[:,3:6]*scale
    #@property
    def GetTemp(self):
        return (self.mass*np.sum(np.power(self.system[:,3:6],2))*0.239005736*10000)/(gamma*(self.system.shape[0]-1)*k)
    #@property
    def IncrementalPos(self,increments):
        self.system[:,0:3] = self.system[:,0:3] + increments
    #@property
    def IncrementalVel(self,increments):
        self.system[:,3:6] = self.system[:,3:6] + increments
    #@property
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

    #@property
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
    #@property
    def CalculateBondEnergy(self):
        bond_energy = 0
        bond_dis = []
        for i in range(0,self.bond_table.shape[0]):
            pair = self.bond_table[i]
            bond_energy += self.bond_energy*((np.linalg.norm(self.system[pair[1]-1][0:3]-self.system[pair[2]-1][0:3]))-self.bond_init)**2
            #self.bond_dis.append((np.linalg.norm(self.system[pair[1]-1]-self.system[pair[2]-1])))
            bond_dis.append(np.linalg.norm(self.system[pair[1]-1]-self.system[pair[2]-1]))

        return bond_energy
    
    #@property
    def CalculateAngleEnergy(self):
        angle_energy = 0
        angle_dis = []
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
            #self.angle_dis.append(math.degrees(math.acos(c)))
            angle_dis.append(math.degrees(math.acos(c)))
        return angle_energy
    
    #@property
    def CalculatetorsionEnergy(self):
        torsion_energy = 0
        torsion_dis = []
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
            #self.torsion_dis.append(math.degrees(phi))
            torsion_dis.append(math.degrees(phi))
        return torsion_energy

    #@property
    def CalPress(self):  
        pxx = ((np.sum(np.power(self.system[:,3:6],2)[:,0]*self.mass)*0.239005736*10000+(np.dot(self.system[:,0],self.CalculateForces()[:,0]))))/self.volume2*68568.415
        pyy = ((np.sum(np.power(self.system[:,3:6],2)[:,1]*self.mass)*0.239005736*10000+(np.dot(self.system[:,1],self.CalculateForces()[:,1]))))/self.volume2*68568.415
        pzz = ((np.sum(np.power(self.system[:,3:6],2)[:,2]*self.mass)*0.239005736*10000+(np.dot(self.system[:,2],self.CalculateForces()[:,2]))))/self.volume2*68568.415

        pxy = (np.dot(self.system[:,3],self.system[:,4]*self.mass*0.239005736*10000+(np.dot(self.system[:,0],self.CalculateForces()[:,1]))))/self.volume2*68568.415
        pxz = (np.dot(self.system[:,3],self.system[:,5]*self.mass*0.239005736*10000+(np.dot(self.system[:,0],self.CalculateForces()[:,2]))))/self.volume2*68568.415
        pyz = (np.dot(self.system[:,4],self.system[:,5]*self.mass*0.239005736*10000+(np.dot(self.system[:,1],self.CalculateForces()[:,2]))))/self.volume2*68568.415

        press = (pxx+pyy+pzz+pxy+pxz+pyz)/6
        
        return press

      
    #@jit(nopython = True)
    def run(self):
        #self.printff('Timestep','KE','Temp','Press','evdw','ebond','eangle','edihedral')
        total_dump = []
        for i in range(nrun+1):
            if i % 10 == 0:
                print(f"step: {i}")  
            total_dump.append([i,self.KineticEnergy(),self.GetTemp(),self.CalPress(),self.CalculateEnergy(),self.CalculateBondEnergy(),self.CalculateAngleEnergy(),self.CalculatetorsionEnergy()])
            force = self.CalculateForces()/ 48.88821291 /48.88821291
            self.IncrementalPos(dt*self.system[:,3:6]+(0.5*(dt*dt)*force)/self.mass)
            force_next = self.CalculateForces()/ 48.88821291 /48.88821291
            self.IncrementalVel((0.5*dt*force+0.5*dt*force_next)/self.mass)
        

        return total_dump


        
class print_output:

    def __init__(self,r):
        self.out = r
    def dump(self):
        print('%8s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t' % ('Timestep','KE','Temp','Press','evdw','ebond','eangle','edihedral'))
        for i in range(0,len(self.out)):
            print('%8d\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f\t' % (self.out[i][0],self.out[i][1],self.out[i][2],self.out[i][3],self.out[i][4],self.out[i][5],self.out[i][6],self.out[i][7]))




if __name__ == '__main__':
    psf = mda.Universe("../pe_l.psf","../1nptts1.dcd")
    data_all = Iteractive_boltzmann_inversion(psf)
    data_all.mapping()
    data_all.cal_distribution(-1)
    #data_all.xyz2data()

    #psf2 = mda.Universe("./test.data","./PE_bead_npt29.dcd")
    #data_all2 = xyz_dis(psf2,60,16)
    #data_all2.IBM_tr_cg()

    #compare_iter(data_all,data_all2).compare_plot()
    initial_parameter = data_all.get_parameter()
    initial_state = copy.deepcopy(initial_parameter[0])
    
    print("MD run")
    start = time.time()
    initial_state = copy.deepcopy(initial_parameter[0])
    sim = Ensemble(initial_state,initial_parameter[1],initial_parameter[2],initial_parameter[3],initial_parameter[4],initial_parameter[5],28,310,1,initial_parameter[6] )
    output = sim.run()
    print_output(output).dump()
    end = time.time()
    print("Cost: "+str(end - start)) 
    print("Done")

   
    
    gc.collect()


