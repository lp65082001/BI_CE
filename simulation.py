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
import subprocess
import numpy as np
import math
import time

from sys import stdout
import warnings
warnings.filterwarnings("ignore")

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
    try:
        subprocess.check_output('nvidia-smi')
        df_device = 'CUDA'
    except Exception: 
        df_device = 'CPU'
    print(f"Device: {df_device}")
    platform = Platform.getPlatformByName(df_device)

    pressure_list = []
    volume = sys[3][0]*sys[3][1]*sys[3][2]*1000
    sig_sys = sys[1]
    res_sys = sys[2]
    print(f"mass:{sys[0]:.3f}, vol:{volume:.3f},bl:{par[0]*0.1:.3f},bk:{par[1]*4.184*10:.3f},al:{par[2]*(math.pi/180):.3f},ak:{par[3]*4.184:.3f},ep:{par[4]*4.184:.3f},sig:{par[5]*0.1:.3f}")
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
    integrator = NoseHooverIntegrator(temp*kelvin, 1/picosecond, timestep*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator,platform)
    simulation.context.setPositions(pdb.positions)
    #print(simulation.context.getState(getVelocities=True, getForces=True).getForces(asNumpy=True))
    simulation.minimizeEnergy()
    #simulation.reporters.append(DCDReporter('output.dcd', 1000,append=False))
    #simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
    #        potentialEnergy=True, temperature=True,volume=True))
    for i in range(100):
        pressure_list.append(cal_press(simulation.context.getState(getPositions=True,getVelocities=True, getForces=True),sys[0],volume))
        simulation.step(500)

    return np.mean(np.array(pressure_list)[-20:100])




