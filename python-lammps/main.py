import multiprocessing as mp 
import time
from lammps import lammps
import numpy as np
import os


""" multi-process lammps simulation module"""
class multi_lammps:

    def __init__(self,par):    
        self.par = par
        self.step = 10
    
    """ simulation process """
    def sample_simulation(self,path):
        press = []  
        
        lmp = lammps()
        bet, ent  = self.time_div(path)
        print(path)
        for j in range(bet,ent):

            lmp.file("./cg"+str(path)+".in")
            lmp.command("pair_coeff * * "+str(self.par[j][0])+" "+str(self.par[j][1]))
            lmp.command("bond_coeff 1 "+str(self.par[j][2])+" "+str(self.par[j][3]))
            lmp.command("angle_coeff 1 "+str(self.par[j][4])+" "+str(self.par[j][5]))

            for i in range(self.step):
                #lmp.command("fix 2 all nvt temp 300.0 300.0 200.0")
                lmp.command("run 2500 pre no post no")
                #lmp.command("""fix 2 all nvt temp 300.0 300.0 200.0
                #        run 2500 pre no post no""")
                #lmp.commands_string(END_BLOCK)
                press.append(lmp.get_thermo("press"))
            
            os.remove("./"+str(path)+".data")
        lmp.close()
        
        return np.mean(np.array(press)[5:9]),path

    """ minimize process """
    def minimize(self,path):
        min = lammps()
        bet, ent  = self.time_div(path)
        print(path)
        for j in range(bet,ent):       
            min.file("./cg"+str(path)+"_min.in") 
            min.command("pair_coeff * * "+str(self.par[j][0])+" "+str(self.par[j][1]))
            min.command("bond_coeff 1 "+str(self.par[j][2])+" "+str(self.par[j][3]))
            min.command("angle_coeff 1 "+str(self.par[j][4])+" "+str(self.par[j][5]))
            min.command("minimize 0.0 1.0e-8 1000 100000")
            #min.command("run 1000")
            #min.commands_string("""minimize 0.0 1.0e-8 1000 100000 
            #                run 1000""")
            min.command("write_data "+str(path)+".data")
            min.force_timeout()
            min.close()

    """ multi-process div """
    def time_div(self,num):
            neighbor = int(self.par.shape[0]/4)
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
                en = self.par.shape[0]
            return be, en

    """ load file (bond,angle) """
    def load_file(self,path):
        info = []
        n = 0
        with open(path) as f:
            for line in f.readlines():
                if line[0:13]=="ITEM: ENTRIES":
                    n = 1
                    continue
                if line[0:14]=="ITEM: TIMESTEP":
                    n = 0
                if n==1:
                    s = line.split(' ')
                    info.append(float(s[1]))
        return np.array(info)

    """ load file (rdf) """
    def load_file_rdf(self,path):
        info1 = []
        info2 = []
        n = 0
        with open(path) as f:
            for line in f.readlines():
                if line[0:5] == "# Row":
                    n = 1
                    continue
                if n==1:
                    s = line.split(' ')
                    if len(s)==2:
                        continue
                    info1.append(float(s[1]))
                    info2.append(float(s[2]))
        return np.array(info1), np.array(info2).reshape((-1,100))

    """ calculate distribution """
    def caldis(self,path):
        bond_all = self.load_file("./bond"+str(path)+".dump")
        angle_all = self.load_file("./angle"+str(path)+".dump")
        rdf_d, rdf_v = self.load_file_rdf("./rdf"+str(path)+".dump")
        bond_d = np.mean(bond_all)
        bond_std = np.std(bond_all)
        angle_d = np.mean(angle_all)
        angle_std = np.std(angle_all)
        rdf = np.mean(rdf_v,axis=0)

        os.remove("./bond"+str(path)+".dump")
        os.remove("./angle"+str(path)+".dump")
        os.remove("./rdf"+str(path)+".dump")

        print(path)
        return bond_d, bond_std, angle_d, angle_std,rdf                                                                                                                                                                                                                                                                                                        
    
    """ multi-process """
    def multi_process(self,task):
        num_process = 4
        start = time.time()
        pool = mp.Pool(processes=num_process)
        result_list = pool.map(task,[0,1,2,3])
        end = time.time()
        print("Cost: "+str(end - start))
        return result_list

    """ simuation process """
    def do_simulation(self,path):  
        press,job = self.sample_simulation(path)
        time.sleep(1)
        return self.caldis(path),press,job
    
    """ minimize process """
    def min_simulation(self,path):
        self.minimize(path)

    """ run process """
    def run(self):
        print("================================= min =================================")
        self.multi_process(self.min_simulation) 
        time.sleep(5)
        print("================================= done =================================")
        return(self.multi_process(self.do_simulation))

""" cross entropy optimizer workflow"""
class cross_entropy:

    def __init__(self,sigma,press,in_dis,rdf,sample=12,par = 6,output_name = "./dump_modify.txt",output_name2 = "./dump_modify2.txt"):
        self.par = par
        self.sigma = sigma
        self.sample = sample
        self.bond_dis = in_dis
        self.rdf_dis = rdf
        self.press = press
        self.dump = output_name
        self.dump2 = output_name2
        #self.threshold = 1
        if os.path.isfile(self.dump):
            os.remove(self.dump)
        if os.path.isfile(self.dump2):
            os.remove(self.dump2)

    """ sample angle and angle stiffness """
    def create_sample_a(self,x):
        y = np.zeros((self.sample,self.par))
        y[:,0] = np.random.normal(x[0],self.sigma,self.sample)
        y[:,1] = np.random.normal(x[1],self.sigma,self.sample)
        #y[:,2] = np.random.normal(x[2],self.sigma,self.sample)
        #y[:,3] = np.random.normal(x[3],self.sigma,self.sample)
        y[:,2] = x[2]
        y[:,3] = x[3]
        y[:,4] = np.random.normal(x[4],self.sigma*2,self.sample)
        y[:,5] = np.random.normal(x[5],self.sigma*100,self.sample)

        for i in range(0,y[:,5].shape[0]):
            if y[i,5] > 180:
                y[i,5] = 180

        return y

    """ sample epsilon and sigma """
    def create_sample_se(self,x):
        y = np.zeros((self.sample,self.par))
        y[:,0] = np.random.normal(x[0],self.sigma*2,self.sample)
        y[:,1] = np.random.normal(x[1],self.sigma*2,self.sample)
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
        y[:,0] = np.random.normal(x11,self.sigma,self.sample)
        y[:,1] = np.random.normal(x12,self.sigma,self.sample)
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
    
    """ none """
    def clear_sample(self,x):
        c_list = []
        n = 0
        for i in range(3):
            c_list.append(x[i][0][2]+n)
            c_list.append(x[i][1][2]+n)
            c_list.append(x[i][2][2]+n)
            c_list.append(x[i][3][2]+n)
            n += 4
        return c_list

    """ loss calculation"""
    def compare_sample(self,x):
        rdf_list = []

        for i in range(3):
            rdf_list.append(self.cal_rdf(self.rdf_dis) - self.cal_rdf(x[i][0][0][4]))
            rdf_list.append(self.cal_rdf(self.rdf_dis) - self.cal_rdf(x[i][1][0][4]))
            rdf_list.append(self.cal_rdf(self.rdf_dis) - self.cal_rdf(x[i][2][0][4]))
            rdf_list.append(self.cal_rdf(self.rdf_dis) - self.cal_rdf(x[i][3][0][4]))

        #print(rdf_list)

        press = []
        press2 = []
        for i in range(3):
            press.append(abs(self.press-x[i][0][1]))
            press.append(abs(self.press-x[i][1][1]))
            press.append(abs(self.press-x[i][2][1]))
            press.append(abs(self.press-x[i][3][1]))
            press2.append(x[i][0][1])
            press2.append(x[i][1][1])
            press2.append(x[i][2][1])
            press2.append(x[i][3][1])

        angle = []
        for i in range(3):
            angle.append(abs(self.bond_dis[2]-x[i][0][0][2]))
            angle.append(abs(self.bond_dis[2]-x[i][1][0][2]))
            angle.append(abs(self.bond_dis[2]-x[i][2][0][2]))
            angle.append(abs(self.bond_dis[2]-x[i][3][0][2]))
        
        angle_dis = []
        for i in range(3):
            angle_dis.append(abs(self.bond_dis[3]-x[i][0][0][3]))
            angle_dis.append(abs(self.bond_dis[3]-x[i][1][0][3]))
            angle_dis.append(abs(self.bond_dis[3]-x[i][2][0][3]))
            angle_dis.append(abs(self.bond_dis[3]-x[i][3][0][3]))
        

        return np.array(angle),np.array(angle_dis),np.absolute(np.array(rdf_list).reshape((-1,2))), np.array(press),np.array(press2)

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
        eps = x[49]
        s1 = np.where(x>=1)[0][0]

        s1p = 0.05+s1*0.1
        sig = 0.1*((1-x[s1])/(x[s1+1]-x[s1]))+s1p
        return np.array([eps, sig])

""" main code """    
if __name__ == '__main__':

    # Parameter setting #
    epech = 250
    cov = 0

    # Parameter #
    initial_dis = [2.5046,0.12784,149.3868,35.2312]
    initial_rdf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00016997529498217342, 0.00208344683188595, 0.008627740977964473, 0.022050350263589327, 0.04937655411865641, 0.09935256659641159, 0.16945465713140392, 0.2722556922206557, 0.3949769912364921, 0.5573100350004938, 0.6938854729090779, 0.8654442778578589, 1.0160726588868556, 1.1233517367356252, 1.2356720561479815, 1.3057548646618078, 1.335590187227102, 1.349195932553016, 1.3402494973522179, 1.3039376532811335, 1.269512084915391, 1.2441620112520324, 1.2146578969217805, 1.1656486160495276, 1.1501526478574395, 1.109495949113851, 1.0627631461686116, 1.0341757757618903, 1.0135787272977217, 0.9683163743807701, 0.9265627043518644, 0.9002105163849191, 0.883071510337252, 0.8738819268256944, 0.8499817628487244, 0.8381528283095702, 0.8263713935629706, 0.7893543037491465, 0.7804896270180164, 0.7637895230594015, 0.7599565024118818, 0.7497972037199938, 0.7447920144524701, 0.7430502888296713, 0.7351787513730245, 0.7404356833356803, 0.7496340982510867, 0.76330144998904, 0.770791909820093, 0.7967562525644208, 0.8081299508773859, 0.8284168578535918, 0.8631183262108293, 0.8888131148680632, 0.9122424671566326, 0.9542912345095615, 0.988980285447886, 1.0123385796118605, 1.047478470346805, 1.069744872705397, 1.089362145057728, 1.1050086630624891, 1.1115858791662083, 1.135743362139896, 1.125168168236439, 1.1372796465384365, 1.1246317445383527])
    #initial_par =   np.array([0.19,4.5,18.17,2.5,0.79,149.39])
    initial_par = np.array([0.298917,4.445173,18.17,2.5,1.265070,178.592109])

    # Initialization #
    optimization = cross_entropy(0.01,1.0,initial_dis,initial_rdf)
    table = optimization.create_sample_a(initial_par)

 
    
    #print(np.where(initial_rdf==np.max(initial_rdf)))
    #print(np.max(initial_rdf))
    #print(initial_rdf.shape)
    #print(initial_rdf[49])

    start = time.time()
    
    # Iter loop #
    for j in range(epech):
        sample_list = []
        n = 0
        """ simulation """
        for i in range(3):            

            sample = multi_lammps(table[n:n+4][:])
            sample_list.append(sample.run())
            
            n +=4
        """ ========== """

        # Loss calculation #
        ad,ast,rdfd,pd,p2 = optimization.compare_sample(sample_list)
        # Dump sample search # 
        optimization.dump_search(table,p2,rdfd,ad,ast)

        # check converage #
        if (cov == 0):
            nxlist = optimization.select_sample_a(ad,ast,rdfd,pd)
            if (optimization.convergence_a(nxlist,ad,1)== -1):
                # Angle convergence #
                cov = 1
                # Next generation and create sample (Epsilon and sigma) #
                nxlist = optimization.select_sample_se(ad,ast,rdfd,pd)
                initial_par = optimization.next_generation(nxlist,table)
                table = optimization.create_sample_se(initial_par)
            else:
                # Next generation and create sample (angle) #
                nxlist = optimization.select_sample_a(ad,ast,rdfd,pd)
                initial_par = optimization.next_generation(nxlist,table)
                table = optimization.create_sample_a(initial_par)
        elif (cov == 1):
            nxlist = optimization.select_sample_se(ad,ast,rdfd,pd)
            se_n, se_refernce = optimization.convergence_se(nxlist,rdfd,0.1,0.1,table)
            if (se_n== -1):
                # Epsilon and Sigma convergence #
                cov = 2
                # Pressure sample #
                nxlist = optimization.select_sample_p(ad,ast,rdfd,pd)
                initial_par = optimization.next_generation(nxlist,table)
                table = optimization.create_sample_p(initial_par,se_refernce) 
            else:
                # Next generation and create sample (Epsilon and sigma) #
                nxlist = optimization.select_sample_se(ad,ast,rdfd,pd)
                initial_par = optimization.next_generation(nxlist,table)
                table = optimization.create_sample_se(initial_par)
        elif (cov == 2):
            # Pressure sample #
            nxlist = optimization.select_sample_p(ad,ast,rdfd,pd)
            initial_par = optimization.next_generation(nxlist,table)
            table = optimization.create_sample_p(initial_par,se_refernce)  

        # dump generation parameter #
        optimization.dump_parameter(j+1,start,cov)
    
    end = time.time()
    print("Cost: "+str(end-start))
    print("done")
