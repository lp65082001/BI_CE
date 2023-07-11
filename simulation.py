from openmm import LangevinIntegrator,NoseHooverIntegrator
from openmm import CustomIntegrator
from openmm.app import Simulation
from openmm.app import PDBReporter
from openmm.app import StateDataReporter
from openmm.app import DCDReporter
from openmm.app import PDBFile, ForceField
from openmm import HarmonicBondForce,HarmonicAngleForce,NonbondedForce
from openmm.unit import picoseconds, kelvin,picosecond
from openmm.openmm import Context, Platform
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis import transformations
import subprocess
import numpy as np
import math
import time

from sys import stdout
import warnings
warnings.filterwarnings("ignore")
  

# Calculate structure #
def bond_angle_rdf(sig,res,pos,vol,b_l,a_l):
    structure = mda.Universe.empty(sig*res,trajectory=True)
    structure.atoms.positions = pos*10
    transform = transformations.boxdimensions.set_dimensions(vol)
    structure.trajectory.add_transformations(transform)
    all_rdf = np.zeros([100,1])
    he = 0
    ed = res-1
    for i in range(0,sig):
        c1 = structure.select_atoms('index '+str(he+sig*i)+' to '+str(ed+sig*i))
        cn = structure.select_atoms('all and not index '+str(he+sig*i)+' to '+str(ed+sig*i))
        ss_rdf = InterRDF(c1,cn,nbins=100,range=(0.0, 10.0))
        ss_rdf.run()

        all_rdf += np.array(ss_rdf.rdf).reshape((100,1))        
    rdf_m = (all_rdf/sig).reshape(-1,1)

    bond_t = []
    for i,j in b_l:
        bond_t.append(np.linalg.norm(pos[j,:]-pos[i,:]))
    bond_t = np.array(bond_t)
    angle_t = []
    for i,j,k in a_l:
        point_1 = pos[i,:]
        point_2 = pos[j,:]
        point_3 = pos[k,:]
        a = np.linalg.norm(point_2-point_3)
        b = np.linalg.norm(point_1-point_3)
        c = np.linalg.norm(point_1-point_2)
        angle_t.append(math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c))))
    angle_t = np.array(angle_t)
    return bond_t, angle_t ,rdf_m

# Calculate pressure #
def cal_press(x,mass,volume):
    velocity = x.getVelocities(asNumpy=True)/100
    position = x.getPositions(asNumpy=True)*10
    force = x.getForces(asNumpy=True)*0.0239
    pxx = ((np.sum(np.power(velocity[:,0],2)*mass)*0.239005736*10000+(np.sum(np.dot(position[:,0],force[:,0])))))/volume*68568.415
    pyy = ((np.sum(np.power(velocity[:,1],2)*mass)*0.239005736*10000+(np.sum(np.dot(position[:,1],force[:,1])))))/volume*68568.415
    pzz = ((np.sum(np.power(velocity[:,2],2)*mass)*0.239005736*10000+(np.sum(np.dot(position[:,2],force[:,2])))))/volume*68568.415

    return (pxx+pyy+pzz)/3

# add bond potential #
def add_bond_table(kb,bl,ss,rs):
    con = HarmonicBondForce()
    bond_l = []
    for j in range(0,ss):
        for i in range(0,rs-1):
            bond_l.append([(i+j*(rs)),(i+1+j*(rs))])
    for ki,kj in bond_l:
        con.addBond(ki,kj,bl,2*kb)
    return con,bond_l

# add angle potential #
def add_angle_table(ka,al,ss,rs):
    con = HarmonicAngleForce()
    n = 1
    angle_l = []
    for j in range(ss):
        for i in range(0,rs-2):
            angle_l.append([(i+j*(rs)),(i+1+j*(rs)),(i+2+j*(rs))])
            n += 1
    for ki,kj,kk in angle_l:
        con.addAngle(ki,kj,kk,al,2*ka)
    return con,angle_l

# add nonbonded potential #
def add_nonbond_table(eps,sig,bl,al,ss,rs):
    con = NonbondedForce()
    con.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    con.setCutoffDistance(1.5)
    for i in range(ss*rs):
        con.addParticle(0,sig,eps)
    for i,j in bl:
        con.addException(i,j,0.0,0.0,0.0)
    for i,j,k in al:
        con.addException(i,k,0.0,0.0,0.0)

    return con

# CG simulation #
def sample_simulation(par,sys,temp = 300, timestep=0.004):
    # detect device #
    try:
        subprocess.check_output('nvidia-smi')
        df_device = 'CUDA'
    except Exception: 
        df_device = 'CPU'
    #print(f"Device: {df_device}")
    platform = Platform.getPlatformByName(df_device)

    # properties list #
    pressure_list = []
    bond_total = []
    angle_total = []
    rdf_total = np.zeros([100,1])

    # parameter setting #
    volume = sys[3][0]*sys[3][1]*sys[3][2]*1000
    sig_sys = sys[1]
    res_sys = sys[2]
    print(f"mass:{sys[0]:.3f}, vol:{volume:.3f},bl:{par[0]*0.1:.3f},bk:{par[1]*4.184*10:.3f},al:{par[2]:.3f},ak:{par[3]*4.184:.3f},ep:{par[4]*4.184:.3f},sig:{par[5]*0.1:.3f}")
    pdb = PDBFile("./cg.pdb")
    forcefield = ForceField("./forcefield.xml")

    # create system #
    system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1.5)
    system.removeForce(0)
    
    # set parameter #
    connect_bond, bond_list = add_bond_table(par[1]*4.184*10,par[0]*0.1,sig_sys,res_sys)
    connect_angle, angle_list = add_angle_table(par[3]*4.184,par[2]*(math.pi/180),sig_sys,res_sys)
    connect_nobond = add_nonbond_table(par[4]*4.184,par[5]*0.1,bond_list,angle_list,sig_sys,res_sys)

    # apply forcefield #
    system.addForce(connect_bond)
    system.addForce(connect_angle)
    system.addForce(connect_nobond)

    #integrator = LangevinIntegrator(500*kelvin, 1/picosecond, 0.004*picoseconds)
    integrator = NoseHooverIntegrator(temp*kelvin, 0.2/picosecond, timestep*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator,platform)
    simulation.context.setPositions(pdb.positions)
    #print(simulation.context.getState(getVelocities=True, getForces=True).getForces(asNumpy=True))
    simulation.minimizeEnergy()
    #simulation.reporters.append(DCDReporter('output.dcd', 1000,append=False))
    #simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
    #        potentialEnergy=True, temperature=True,volume=True))
    for i in range(100):
        # Calculate properties #
        pressure_list.append(cal_press(simulation.context.getState(getPositions=True,getVelocities=True, getForces=True),sys[0],volume))
        b_t, a_t, rdf_t = bond_angle_rdf(sig_sys,res_sys,simulation.context.getState(getPositions=True,getVelocities=True, getForces=True).getPositions(asNumpy=True),sys[3],bond_list,angle_list)
        
        bond_total.append(b_t)
        angle_total.append(a_t)
        rdf_total += rdf_t
        
        simulation.step(50)
    bond_total = np.array(bond_total)
    angle_total = np.array(angle_total)
    #return np.mean(np.array(pressure_list)[-20:100])
    return np.mean(np.array(pressure_list)[-20:100]),[np.mean(bond_total),np.std(bond_total)],[np.mean(angle_total),np.std(angle_total)],(rdf_total/100).reshape(1,100)




