"""
OpenMM-based Molecular Dynamics Engine for BI_CE Coarse-Grained Polymer Model.
Supports multi-process execution on a single GPU (CUDA/OpenCL) or parallel execution on CPU.
"""
import openmm
from openmm import unit
import numpy as np
import math

def get_openmm_platform(gpu_id=0, requested_platform=None):
    """
    Select OpenMM platform and properties.
    If GPU (CUDA/OpenCL) is available, targets the specified gpu_id to allow
    multiple subprocesses to share the same GPU.
    If no GPU is available, falls back to CPU platform (with CpuThreads=1 per worker).
    """
    if requested_platform is not None:
        try:
            plat = openmm.Platform.getPlatformByName(requested_platform)
            props = {'DeviceIndex': str(gpu_id)} if requested_platform in ['CUDA', 'OpenCL'] else {}
            return plat, props
        except Exception:
            pass

    for plat_name in ['CUDA', 'OpenCL']:
        try:
            plat = openmm.Platform.getPlatformByName(plat_name)
            props = {'DeviceIndex': str(gpu_id)}
            # Verify context creation on device
            test_sys = openmm.System()
            test_sys.addParticle(1.0 * unit.amu)
            test_ctx = openmm.Context(test_sys, openmm.VerletIntegrator(1.0 * unit.femtosecond), plat, props)
            del test_ctx
            return plat, props
        except Exception:
            continue

    # Fallback to CPU platform
    try:
        plat = openmm.Platform.getPlatformByName('CPU')
        # In multi-process runs, 1 thread per worker prevents thread oversubscription
        return plat, {'CpuThreads': '1'}
    except Exception:
        plat = openmm.Platform.getPlatformByName('Reference')
        return plat, {}


class MD_Universe:
    """
    OpenMM Molecular Dynamics Simulation Universe for Coarse-Grained Polymer Model.
    """
    def __init__(self, initial_setting, initial_parameter, mass=28, temperature=300, pressure=1,
                 gpu_id=0, platform_name=None, dt=4.0, cutoff=20.0):
        self.mass = float(mass)
        self.temp = float(temperature) if temperature is not None else 300.0
        self.press = float(pressure) if pressure is not None else 1.0
        self.dt = float(dt)
        self.cutoff = float(cutoff)
        self.gpu_id = gpu_id
        self.platform_name = platform_name

        # Parse settings: [positions, bond_table, angle_table, box]
        self.box = np.array(initial_setting[3], dtype=float)
        self.bond_table = np.array(initial_setting[1], dtype=int)
        self.angle_table = np.array(initial_setting[2], dtype=int)
        init_coords = np.array(initial_setting[0], dtype=float)

        self.n_atoms = init_coords.shape[0]
        self.volume_nm3 = np.prod(self.box * 0.1)

        # Parse parameters: [bond_init, bond_energy, angle_init, angle_energy, sigma, epsilon]
        self.bond_init = float(initial_parameter[0])
        self.bond_energy = float(initial_parameter[1])
        self.angle_init = float(initial_parameter[2])
        self.angle_energy = float(initial_parameter[3])
        self.sigma = float(initial_parameter[4])
        self.epsilon = float(initial_parameter[5])

        self.bond_dis = []
        self.angle_dis = []

        # System coordinates and velocities [x, y, z, vx, vy, vz] in Angstrom and Angstrom/ps
        self.system = np.hstack((init_coords, np.zeros((self.n_atoms, 3))))

        # Build OpenMM System
        self._build_openmm_system(init_coords)

    def _build_openmm_system(self, init_coords):
        self.omm_system = openmm.System()

        # Set periodic box vectors (converted from Angstrom to nm)
        box_vec_a = [self.box[0], 0.0, 0.0] * unit.angstrom
        box_vec_b = [0.0, self.box[1], 0.0] * unit.angstrom
        box_vec_c = [0.0, 0.0, self.box[2]] * unit.angstrom
        self.omm_system.setDefaultPeriodicBoxVectors(box_vec_a, box_vec_b, box_vec_c)

        # Add particles
        for _ in range(self.n_atoms):
            self.omm_system.addParticle(self.mass * unit.amu)

        # 1. Harmonic Bond Force (Force Group 0)
        # Note: OpenMM E = 0.5 * k * (r - r0)^2, while repo E = K_b * (r - r0)^2 -> k_omm = 2 * K_b
        self.bond_force = openmm.HarmonicBondForce()
        self.bond_force.setForceGroup(0)
        k_bond_omm = 2.0 * self.bond_energy * unit.kilocalories_per_mole / (unit.angstrom ** 2)
        r0 = self.bond_init * unit.angstrom
        if self.bond_table.shape[0] > 0:
            for b in self.bond_table:
                self.bond_force.addBond(int(b[1] - 1), int(b[2] - 1), r0, k_bond_omm)
        self.omm_system.addForce(self.bond_force)

        # 2. Harmonic Angle Force (Force Group 1)
        # Note: OpenMM E = 0.5 * k * (theta - theta0)^2 -> k_omm = 2 * K_a
        self.angle_force = openmm.HarmonicAngleForce()
        self.angle_force.setForceGroup(1)
        k_angle_omm = 2.0 * self.angle_energy * unit.kilocalories_per_mole / (unit.radian ** 2)
        theta0 = self.angle_init * unit.degrees
        if self.angle_table.shape[0] > 0:
            for a in self.angle_table:
                self.angle_force.addAngle(int(a[1] - 1), int(a[2] - 1), int(a[3] - 1), theta0, k_angle_omm)
        self.omm_system.addForce(self.angle_force)

        # 3. Nonbonded Lennard-Jones Force (Force Group 2)
        self.nb_force = openmm.NonbondedForce()
        self.nb_force.setForceGroup(2)
        self.nb_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        actual_cutoff = min(self.cutoff, 0.49 * float(np.min(self.box))) * unit.angstrom
        self.nb_force.setCutoffDistance(actual_cutoff)

        sigma_val = self.sigma * unit.angstrom
        eps_val = self.epsilon * unit.kilocalories_per_mole
        for _ in range(self.n_atoms):
            self.nb_force.addParticle(0.0 * unit.elementary_charge, sigma_val, eps_val)

        # Exclude 1-2 and 1-3 pairs from nonbonded interactions
        if self.bond_table.shape[0] > 0:
            for b in self.bond_table:
                self.nb_force.addException(int(b[1] - 1), int(b[2] - 1),
                                          0.0 * unit.elementary_charge**2, 1.0 * unit.angstrom, 0.0 * unit.kilocalories_per_mole)
        if self.angle_table.shape[0] > 0:
            for a in self.angle_table:
                self.nb_force.addException(int(a[1] - 1), int(a[3] - 1),
                                          0.0 * unit.elementary_charge**2, 1.0 * unit.angstrom, 0.0 * unit.kilocalories_per_mole)
        self.omm_system.addForce(self.nb_force)

        # Integrator: LangevinMiddleIntegrator maintains NVT accurately
        self.integrator = openmm.LangevinMiddleIntegrator(
            self.temp * unit.kelvin,
            1.0 / unit.picosecond,
            self.dt * unit.femtosecond
        )

        # Select platform and initialize context
        self.platform, self.properties = get_openmm_platform(self.gpu_id, self.platform_name)
        self.context = openmm.Context(self.omm_system, self.integrator, self.platform, self.properties)

        # Set initial positions
        self.context.setPositions(init_coords * unit.angstrom)
        self.InitVelDis()

    def InitVelDis(self):
        """Initialize Maxwell-Boltzmann velocities to target temperature."""
        self.context.setVelocitiesToTemperature(self.temp * unit.kelvin)
        self._update_system_state()

    def pbc_displacement(self, dr):
        """Minimum image convention displacement for orthorhombic box."""
        return dr - self.box * np.round(dr / self.box)

    def _update_system_state(self):
        """Sync positions and velocities from OpenMM Context to self.system."""
        state = self.context.getState(getPositions=True, getVelocities=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.angstrom / unit.picosecond)
        self.system[:, 0:3] = pos
        self.system[:, 3:6] = vel

        # Update bond lengths under PBC
        if self.bond_table.shape[0] > 0:
            idx1 = self.bond_table[:, 1] - 1
            idx2 = self.bond_table[:, 2] - 1
            dr = self.pbc_displacement(pos[idx2] - pos[idx1])
            self.bond_dis = np.linalg.norm(dr, axis=1).tolist()

        # Update angles under PBC
        if self.angle_table.shape[0] > 0:
            idx1 = self.angle_table[:, 1] - 1
            idx2 = self.angle_table[:, 2] - 1
            idx3 = self.angle_table[:, 3] - 1
            r1 = self.pbc_displacement(pos[idx1] - pos[idx2])
            r2 = self.pbc_displacement(pos[idx3] - pos[idx2])
            norm1 = np.linalg.norm(r1, axis=1)
            norm2 = np.linalg.norm(r2, axis=1)
            valid = (norm1 > 1e-6) & (norm2 > 1e-6)
            cos_a = np.sum(r1[valid] * r2[valid], axis=1) / (norm1[valid] * norm2[valid])
            cos_a = np.clip(cos_a, -1.0, 1.0)
            self.angle_dis = np.degrees(np.arccos(cos_a)).tolist()

    def GetTemp(self):
        state = self.context.getState(getEnergy=True)
        ke_kJ_mol = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        k_B = 0.008314462618 # kJ/(mol*K)
        dof = max(1, 3 * self.n_atoms - 3)
        return float((2.0 * ke_kJ_mol) / (dof * k_B))

    def KineticEnergy(self):
        state = self.context.getState(getEnergy=True)
        return float(state.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole))

    def CalculateEnergy(self):
        """Non-bonded van der Waals energy in kcal/mol."""
        state = self.context.getState(getEnergy=True, groups={2})
        return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

    def CalculateBondEnergy(self):
        """Bond energy in kcal/mol."""
        state = self.context.getState(getEnergy=True, groups={0})
        return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

    def CalculateAngleEnergy(self):
        """Angle energy in kcal/mol."""
        state = self.context.getState(getEnergy=True, groups={1})
        return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

    def CalPress(self):
        """Virial pressure calculation in atm."""
        state = self.context.getState(getPositions=True, getForces=True, getEnergy=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        forces_kJ = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        ke_kJ_mol = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        vol_nm3 = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)

        virial = np.sum(pos_nm * forces_kJ)
        # 1 (kJ/mol)/nm^3 = 16.60539067 bar = 16.38824641 atm
        p_bar = ((2.0 * ke_kJ_mol + virial) / (3.0 * vol_nm3)) * 16.60539067
        p_atm = p_bar / 1.01325
        return float(p_atm)

    def run(self, nsteps=100, print_interval=None):
        if print_interval is None:
            print_interval = max(1, nsteps // 5)
        print("Status: Running (OpenMM on %s, Device: %s)" % (self.platform.getName(), str(self.gpu_id)))
        print('%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s' % ('Timestep','KE','Temp','Press','evdw','ebond','eangle'))

        current_step = 0
        while current_step < nsteps:
            steps_to_take = min(print_interval, nsteps - current_step)
            self.integrator.step(steps_to_take)
            current_step += steps_to_take
            self._update_system_state()
            print('%10d\t%10f\t%10f\t%10f\t%10f\t%10f\t%10f' % (
                current_step,
                self.KineticEnergy(),
                self.GetTemp(),
                self.CalPress(),
                self.CalculateEnergy(),
                self.CalculateBondEnergy(),
                self.CalculateAngleEnergy()
            ))