""" Create by Amborse hui from M^5 lab """

import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis import transformations
import numpy as np
import math 
import multiprocessing as mp
import warnings

class FA2CG:
    
    def __init__(self,structure,trajectory):
        data = mda.Universe(structure,trajectory)
        self.info = data
        self.bead_mass = data.select_atoms('all').total_mass()/len(set(data.select_atoms('all').segids))/len(set(data.select_atoms('all').resids))
        self.res = len(set(data.select_atoms('all').resids))
        self.sig = len(set(data.select_atoms('all').segids))
        self.frame = data.trajectory.n_frames
        self.box = data.dimensions
        self.bins_num = 101
        self.max_edge = 10 
        self.path = "./CG"
        print("Status: Initialized successfully")       

    # get all position information #
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

    # build CG model env #
    def build_universe(self):
        self.cg_info = mda.Universe(self.path+".xyz")
        transform = transformations.boxdimensions.set_dimensions(self.box)
        self.cg_info.trajectory.add_transformations(transform)

    # calculate RDF #
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

    # build bond list #
    def bond_list(self,x):
        y = []
        for i in range(0,x.shape[0]-1):
            y.append(np.linalg.norm(x[i+1]-x[i]))
        return y   

    # build angle list #
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

    # calculate gaussian #
    def gaussian(self,x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    # Parallelization #    
    def multi_process(self,task):
        num_process = 4
        pool = mp.Pool(processes=num_process)
        result_list = pool.map(task,[0,1,2,3])
        return result_list

    # split by time #
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

    # split by sig #
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
    
    # main process #
    def mapping(self):
        print("Status: mpi-process-position")
        self.list_position = np.sum(self.multi_process(self.generate_position),axis=0)
        self.FA2xyz()
        self.build_universe()

    # calculate distribution from FA #    
    def cal_distribution(self):
        bond_all = []
        angle_all = []
        for k in range(0,self.frame):
            for j in range(0,self.sig):
                bond_all.append(self.bond_list(self.list_position[k][j*60:(j+1)*60,:])) 
                angle_all.append(self.angle_list(self.list_position[k][j*60:(j+1)*60,:])) 
        b_a = np.array(bond_all).reshape((-1,1))
        a_a = np.array(angle_all).reshape((-1,1))
        b_d = self.gaussian(b_a,np.mean(b_a),np.std(b_a))
        a_d = self.gaussian(a_a,np.mean(a_a),np.std(a_a))


        self.b_a1 = b_a
        self.a_a1 = a_a
        self.b_d1 = b_d
        self.a_d1 = a_d


        print("Status: mpi-process-rdf")
        rdf = np.sum(np.array(self.multi_process(self.pairCorrelationFunction_3D)).reshape((4,100)),axis=0)
        self.rdf_m = rdf/self.sig

        Kb = 0.008314
        T = 300

        ### bond, angle ###
        bond_energy = -Kb*T*np.log(b_d)/4.2
        bond_energy = bond_energy + abs(min(bond_energy)) 
        angle_energy = -Kb*T*np.log(a_d)/4.2 
        angle_energy = angle_energy - abs(min(angle_energy))  

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
        
        print("Status: Mapping Done!")
    
    # output datafile (CG) #
    def xyz2data(self):
        #header#
        f = open(self.path+".data", 'w')
        f.write("Create by Amborse hui from M^5 lab\n")
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
               
    # output CG model from FA model #
    def FA2xyz(self):
        name = self.path+".xyz"
        f = open(name, 'w')
        for k in range(0,self.frame):
            f.write(str(self.res*self.sig)+"\n")
            f.write(str(k)+"\n")
            for z in self.list_position[k][:][:]:
                f.write("1 "+str(z[0])+" "+str(z[1])+" "+str(z[2])+"\n")   
        f.close()

    # get parameter #
    def get_parameter(self):
        position = []
        bond_l= []
        angle_l = []

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
        # box size #
        volume = np.array([self.box[0],self.box[1],self.box[2]])
        
        # potential list #
        potential = np.array([self.bondi,self.bondk,self.anglei,self.anglek[0],self.sigma_,self.epsilon_])

        return [np.array(position).reshape((-1,3)),np.array(bond_l).reshape((-1,3)),np.array(angle_l).reshape((-1,4)),volume],potential

    # get all time position #
    def get_pos(self):
        return self.list_position

    # get rdf #
    def get_rdf(self):
        return  self.rdf_m
    
    # get distribution # 
    def get_dis(self):
    	return np.array([np.mean(self.b_a1),np.std(self.b_a1),np.mean(self.a_a1),np.std(self.a_a1)]) 