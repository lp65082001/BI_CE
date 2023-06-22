import MDAnalysis as mda
#from MDAnalysis.tests.datafiles import PSF, DCD
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis import transformations
#from MDAnalysis.transformations.boxdimensions import set_dimensions
from matplotlib import pyplot as plt
import numpy as np
import gc
import math 
import multiprocessing as mp
#from os import getpid
import threading
#from numba import jit
#from numba.experimental import jitclass
from collections import OrderedDict
#from numba import int32, float32, int64,float64   # import the types
#from scipy.spatial import distance_matrix
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
        transform = transformations.boxdimensions.set_dimensions(self.box)
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
                #torsion_all.append(self.torsion_list(self.list_position[k][j*60:(j+1)*60,:])) 
        b_a = np.array(bond_all).reshape((-1,1))
        a_a = np.array(angle_all).reshape((-1,1))
        t_a = np.array(torsion_all).reshape((-1,1))
        b_d = self.gaussian(b_a,np.mean(b_a),np.std(b_a))
        a_d = self.gaussian(a_a,np.mean(a_a),np.std(a_a))
        #t_d = self.gaussian(t_a,np.mean(t_a),np.std(t_a))

        self.b_a1 = b_a
        self.a_a1 = a_a
        self.t_a1 = t_a
        self.b_d1 = b_d
        self.a_d1 = a_d
        #self.t_d1 = t_d


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
        #torsion_energy = -Kb*T*np.log(t_d)/4.2 
        #torsion_energy = torsion_energy - abs(min(torsion_energy))
        

        fit_a = -1

        bond_init = b_a[np.where(bond_energy == min(bond_energy))[0][0]][0]
        angle_init = a_a[np.where(angle_energy == min(angle_energy))[0][0]][0]
        #torsion_init = t_a[np.where(torsion_energy == min(torsion_energy))[0][0]][0]

        bond_init2 = b_a[np.where(bond_energy == np.sort(bond_energy,axis=0)[fit_a])[0][0]][0]
        angle_init2 = a_a[np.where(angle_energy == np.sort(angle_energy,axis=0)[fit_a])[0][0]][0]
        #torsion_init2 = t_a[np.where(torsion_energy == np.sort(torsion_energy,axis=0)[fit_a])[0][0]][0]

        bond_k = (np.sort(bond_energy,axis=0)[fit_a])/(bond_init2-bond_init)**2
        angle_k = (np.sort(angle_energy,axis=0)[fit_a])/(angle_init2-angle_init)**2
        #torsion_k = (np.sort(torsion_energy,axis=0)[fit_a])/(torsion_init2-torsion_init)**2

        x_b = np.linspace(0,3,100)
        x_a = np.linspace(0,180,500)

        y_b = bond_k*(x_b-bond_init)**2
        y_a = angle_k*(x_a-angle_init)**2
        #y_t = torsion_k*(x_a-torsion_init)**2
        

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
        self.anglek = angle_k[0]
        self.epsilon_ = epsilon
        self.sigma_ = sigma[0]

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
        n = 1
        for j in range(0,self.sig):
            for i in range(0,self.res-2):
                angle_l.append([n,(i+1+j*(self.res)),(i+2+j*(self.res)),(i+3+j*(self.res))])
                n += 1


        volume = np.array([self.box[0],self.box[1],self.box[2]]).astype("float64")
        pot = np.array([self.bondi,self.bondk,self.anglei,self.anglek,self.sigma_,self.epsilon_]).astype("float64")
        
        position = np.array(position,dtype = "float64").reshape((-1,3))
        position = np.hstack((position,np.zeros([position.shape[0],3],dtype = "float64"))).astype("float64")
        bond_l = np.array(bond_l,dtype = "int64").reshape((-1,3))
        angle_l = np.array(angle_l,dtype = "int64").reshape((-1,4))
        #dihedral_l = np.array(dihedral_l,dtype = "int64").reshape((-1,5))
        return [position,bond_l,angle_l,volume,pot]

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

    #@jit
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

    #@jit
    def bond_list(self,x):
        y = []
        for i in range(0,x.shape[0]-1):
            y.append(np.linalg.norm(x[i+1]-x[i]))
        return y   
    #@jit
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

    #@jit
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


    #@jit
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
    #@jit
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
    #@jit
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

'''

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
'''


