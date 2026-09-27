"""
Simulation interface for CEM parameter sampling using OpenMM.
Connects OpenMM MD_Universe simulation with parameter evaluation.
"""
import numpy as np
from MD import MD_Universe

def compute_rdf(coords, box, nbins=100, r_max=10.0):
    """Calculate radial distribution function g(r) with periodic boundaries."""
    n_atoms = coords.shape[0]
    if n_atoms < 2:
        return np.zeros(nbins)

    dr = coords[:, None, :] - coords[None, :, :]
    dr = dr - box * np.round(dr / box)
    dist = np.linalg.norm(dr, axis=-1)

    i_upper, j_upper = np.triu_indices(n_atoms, k=1)
    pair_dists = dist[i_upper, j_upper]

    hist, bin_edges = np.histogram(pair_dists, bins=nbins, range=(0.0, r_max))
    volume = np.prod(box)
    rho = n_atoms / volume
    r_lower = bin_edges[:-1]
    r_upper = bin_edges[1:]
    shell_vol = (4.0 / 3.0) * np.pi * (r_upper**3 - r_lower**3)
    denom = 0.5 * n_atoms * shell_vol * rho
    denom[denom == 0] = 1e-12
    rdf = hist / denom
    return rdf

def sample_simulation(input_par, env, temp=300, timestep=4.0, mass=28, nsteps=50, gpu_id=0, platform_name=None):
    """
    Run an OpenMM MD simulation for a parameter set and return sampled properties.

    Parameters:
    -----------
    input_par : array-like [bond_init, bond_energy, angle_init, angle_energy, sigma, epsilon]
    env : initial_setting [coords, bond_table, angle_table, box]
    temp : float, target temperature (K)
    timestep : float, MD timestep in femtoseconds
    mass : float, bead mass in amu
    nsteps : int, MD simulation steps
    gpu_id : int, device index to run on (enables multiple subprocesses on same GPU)
    platform_name : str or None, 'CUDA', 'OpenCL', 'CPU', or None (auto)

    Returns:
    --------
    press : float
    bond : np.ndarray [mean_bond, std_bond]
    angle : np.ndarray [mean_angle, std_angle]
    rdf : np.ndarray (nbins,)
    """
    sim = MD_Universe(env, input_par, mass=mass, temperature=temp, gpu_id=gpu_id, platform_name=platform_name, dt=timestep)
    # Run simulation quietly for sampling
    sim.run(nsteps=nsteps, print_interval=nsteps)

    press = sim.CalPress()

    if len(sim.bond_dis) > 0:
        bond = np.array([np.mean(sim.bond_dis), np.std(sim.bond_dis)])
    else:
        bond = np.array([float(input_par[0]), 0.0])

    if len(sim.angle_dis) > 0:
        angle = np.array([np.mean(sim.angle_dis), np.std(sim.angle_dis)])
    else:
        angle = np.array([float(input_par[2]), 0.0])

    box = np.array(env[3], dtype=float)
    rdf = compute_rdf(sim.system[:, 0:3], box, nbins=100)

    return press, bond, angle, rdf
