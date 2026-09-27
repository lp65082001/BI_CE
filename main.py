#!/usr/bin/env python3
"""
BI_CE: Boltzmann Inversion & Cross-Entropy Method Coarse-Graining CLI
Author: Ambrose hui (M^5 lab), refactored with OpenMM & CLI interface.
"""
import argparse
import sys
import os
import warnings
from mapping import FA2CG
from MD import MD_Universe
from CEM import CEM

warnings.filterwarnings("ignore")

def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Coarse-Graining workflow using Boltzmann Inversion and Cross-Entropy Optimization with OpenMM."
    )
    # Required inputs
    parser.add_argument("-s", "--structure", required=True, type=str,
                        help="Path to full-atomic structure/topology file (.psf, .pdb, .gro, .tpr, etc.)")
    parser.add_argument("-t", "--trajectory", required=True, type=str,
                        help="Path to full-atomic trajectory file (.dcd, .xtc, .trr, .nc, etc.)")

    # Outputs
    parser.add_argument("-o", "--output", default="./CG", type=str,
                        help="Output path prefix for generated CG files (default: ./CG)")
    parser.add_argument("--export-data", action="store_true",
                        help="Export CG LAMMPS data file (.data)")

    # Physical parameters
    parser.add_argument("--temp", "--temperature", dest="temperature", type=float, default=300.0,
                        help="Simulation temperature in Kelvin (default: 300.0)")
    parser.add_argument("--press", "--pressure", dest="pressure", type=float, default=1.0,
                        help="Target pressure in atm (default: 1.0)")
    parser.add_argument("--mass", type=float, default=None,
                        help="Bead mass in amu (default: auto-calculated from topology)")

    # OpenMM simulation options
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of MD simulation steps (default: 100)")
    parser.add_argument("--dt", type=float, default=4.0,
                        help="MD timestep in femtoseconds (default: 4.0)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="Target GPU device index (default: 0)")
    parser.add_argument("--platform", type=str, default=None,
                        choices=["CUDA", "OpenCL", "CPU", "Reference"],
                        help="OpenMM platform (default: auto-detect CUDA/OpenCL -> CPU)")

    # CEM optimization options
    parser.add_argument("--cem", action="store_true",
                        help="Run Cross-Entropy Method (CEM) optimization after Boltzmann Inversion")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of parameter samples for CEM (default: 20)")
    parser.add_argument("--processes", type=int, default=4,
                        help="Number of parallel worker processes for CEM (default: 4)")
    parser.add_argument("--cem-md-steps", type=int, default=50,
                        help="Number of MD simulation steps per CEM sample (default: 50)")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.structure):
        print(f"Error: Structure file not found: {args.structure}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.trajectory):
        print(f"Error: Trajectory file not found: {args.trajectory}", file=sys.stderr)
        sys.exit(1)

    print("==================================================")
    print("BI_CE: Coarse-Graining Simulation & Optimization")
    print("==================================================")
    print(f"Structure   : {args.structure}")
    print(f"Trajectory  : {args.trajectory}")
    print(f"Output      : {args.output}")
    print(f"Temperature : {args.temperature} K")
    print(f"Pressure    : {args.pressure} atm")
    print(f"GPU ID      : {args.gpu_id}")
    print(f"Platform    : {args.platform or 'Auto'}")
    print("--------------------------------------------------")

    # Step 1: Mapping & Boltzmann Inversion
    data_all = FA2CG(args.structure, args.trajectory, output_prefix=args.output)
    data_all.mapping()
    data_all.cal_distribution()

    if args.export_data:
        data_all.xyz2data()
        print(f"CG LAMMPS data exported to {args.output}.data")

    initial_setting, initial_potential = data_all.get_parameter()
    mass = args.mass if args.mass is not None else float(data_all.bead_mass)
    print(f"Bead Mass   : {mass:.4f} amu")

    # Step 2: OpenMM MD Simulation
    print("\n[Step 2] Running Verification MD Simulation in OpenMM...")
    sim = MD_Universe(
        initial_setting,
        initial_potential,
        mass=mass,
        temperature=args.temperature,
        pressure=args.pressure,
        gpu_id=args.gpu_id,
        platform_name=args.platform,
        dt=args.dt
    )
    sim.run(nsteps=args.steps)

    # Step 3 (Optional): CEM Optimization
    if args.cem:
        print("\n[Step 3] Running Cross-Entropy Method (CEM) Optimization...")
        conformation = data_all.get_dis()
        optimizer = CEM(
            initial_potential,
            initial_setting,
            conformation,
            pressure=args.pressure,
            temperature=args.temperature,
            n_s=args.samples,
            gpu_id=args.gpu_id,
            num_process=args.processes,
            platform_name=args.platform,
            md_steps=args.cem_md_steps
        )
        props, loss = optimizer.CEM_optimizer_run()
        print("CEM optimization epoch completed.")

    print("\nProcess finished successfully!")

if __name__ == '__main__':
    main()