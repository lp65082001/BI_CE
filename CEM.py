import multiprocessing as mp
from simulation import sample_simulation 
import time
import numpy as np
import os

def _worker_task(args):
    """Picklable top-level worker for multiprocessing."""
    job, input_par, env, temp, dt, nsteps, gpu_id, platform_name = args
    press, bond, angle, rdf = sample_simulation(
        input_par, env, temp=temp, timestep=dt, nsteps=nsteps,
        gpu_id=gpu_id, platform_name=platform_name
    )
    return job, press, bond, angle, rdf

class CEM:
    def __init__(self, par, sys, conformation, pressure=1, temperature=300, n_s=20,
                 gpu_id=0, num_process=4, platform_name=None, md_steps=50):
        self.init_par = np.array(par, dtype=float)
        self.env = sys
        self.con = conformation
        self.num_sample = n_s
        self.dis_sigma = 0.01
        self.pressure_ref = pressure
        self.temperature = temperature
        self.gpu_id = gpu_id
        self.num_process = num_process
        self.platform_name = platform_name
        self.md_steps = md_steps

        self.dump = "dump_parameter.txt"
        self.dump2 = "dump_search.txt"
        self.sample_table = np.zeros((self.num_sample, len(self.init_par)))
        self.bl = self.init_par[0]
        self.be = self.init_par[1]
        self.al = self.init_par[2]
        self.ae = self.init_par[3]
        self.sig = self.init_par[4]
        self.eps = self.init_par[5]
        self.pp = pressure

    """ multi-process """
    def multi_process(self, task, sample_l):
        num_process = min(self.num_process, len(sample_l))
        with mp.Pool(processes=num_process) as pool:
            result_list = pool.map(task, sample_l)
        return result_list

    """ run simulation """
    def run(self, nloop=None):
        if nloop is None:
            nloop = int(np.ceil(self.num_sample / self.num_process))

        pressure_table = np.zeros((self.num_sample, 1))
        bond_table = np.zeros((self.num_sample, 2))
        angle_table = np.zeros((self.num_sample, 2))
        rdf_table = np.zeros((self.num_sample, 100))

        start = time.time()
        for i in range(nloop):
            batch_indices = list(range(i * self.num_process, min((i + 1) * self.num_process, self.num_sample)))
            if not batch_indices:
                break
            batch_tasks = [
                (
                    idx,
                    self.sample_table[idx, :],
                    self.env,
                    self.temperature,
                    4.0,
                    self.md_steps,
                    self.gpu_id,
                    self.platform_name
                )
                for idx in batch_indices
            ]
            pr = self.multi_process(_worker_task, batch_tasks)
            for res in pr:
                pressure_table[res[0], :] = res[1]
                bond_table[res[0], :] = res[2]
                angle_table[res[0], :] = res[3]
                rdf_table[res[0], :] = res[4]

        end = time.time()
        print("Cost: " + str(end - start))
        return [pressure_table, bond_table, angle_table, rdf_table]

    """ sample angle and angle stiffness """
    def create_sample_a(self, x):
        y = np.zeros((self.num_sample, len(self.init_par)))
        y[:, 0] = x[0]
        y[:, 1] = x[1]
        y[:, 2] = np.random.normal(x[2], self.dis_sigma * 100, self.num_sample)
        y[:, 3] = np.random.normal(x[3], self.dis_sigma * 2, self.num_sample)
        y[:, 4] = x[4]
        y[:, 5] = x[5]

        for i in range(0, y[:, 2].shape[0]):
            if y[i, 2] > 180:
                y[i, 2] = 180
            elif y[i, 2] < 0:
                y[i, 2] = 0

        self.sample_table = y
        return y

    """ sample epsilon and sigma """
    def create_sample_se(self, x):
        y = np.zeros((self.num_sample, len(self.init_par)))
        y[:, 0] = x[0]
        y[:, 1] = x[1]
        y[:, 2] = x[2]
        y[:, 3] = x[3]
        y[:, 4] = np.random.normal(x[4], self.dis_sigma * 2, self.num_sample)
        y[:, 5] = np.random.normal(x[5], self.dis_sigma * 2, self.num_sample)
        self.sample_table = y
        return y

    """ sample pressure """
    def create_sample_p(self, x, xx):
        y = np.zeros((self.num_sample, len(self.init_par)))
        thresholdx1 = 0.01
        thresholdx2 = 0.5

        if (x[4] > xx[4] + thresholdx1):
            x11 = xx[4] + thresholdx1
        elif (x[4] < xx[4] - thresholdx1):
            x11 = xx[4] - thresholdx1
        else:
            x11 = x[4]

        if (x[5] > xx[5] + thresholdx2):
            x12 = xx[5] + thresholdx2
        elif (x[5] < xx[5] - thresholdx2):
            x12 = xx[5] - thresholdx2
        else:
            x12 = x[5]

        y[:, 0] = x[0]
        y[:, 1] = x[1]
        y[:, 2] = x[2]
        y[:, 3] = x[3]
        y[:, 4] = np.random.normal(x11, self.dis_sigma, self.num_sample)
        y[:, 5] = np.random.normal(x12, self.dis_sigma, self.num_sample)
        self.sample_table = y
        return y

    """ select angle """
    def select_sample_a(self, x1, x2, x3, x4):
        self.pp = np.mean(x4[1:-1]) if len(x4) > 2 else np.mean(x4)
        y1 = x1.argsort()
        return np.array(y1)[0:min(6, len(y1))]

    """ select epsilon and sigma"""    
    def select_sample_se(self, x1, x2, x3, x4):
        self.pp = np.mean(x4[1:-1]) if len(x4) > 2 else np.mean(x4)
        y31 = x3[:, 0].argsort()
        y32 = x3[:, 1].argsort()
        top_k = min(8, len(y31))
        z1 = np.intersect1d(y31[0:top_k], y32[0:top_k])
        if len(z1) == 0:
            z1 = y31[0:top_k]
        return np.array(z1)

    """ select pressure """
    def select_sample_p(self, x1, x2, x3, x4):
        self.pp = np.mean(x4[1:-1]) if len(x4) > 2 else np.mean(x4)
        y31 = x3[:, 0].argsort()
        y32 = x3[:, 1].argsort()
        y4 = x4.argsort()

        top_k = min(6, len(y31))
        z1 = np.intersect1d(y31[0:top_k], y4[0:top_k])
        z2 = np.intersect1d(y32[0:top_k], y4[0:top_k])
        zz = np.concatenate((z1, z2))
        if zz.shape[0] == 0:
            zz = y4[0:top_k]
        return zz

    """ generation next sample average"""
    def next_generation(self, x1, x2):
        newlist = x2[x1, :]
        new_gen = np.mean(newlist, axis=0)
        self.bl = new_gen[0]
        self.be = new_gen[1]
        self.al = new_gen[2]
        self.ae = new_gen[3]
        self.sig = new_gen[4]
        self.eps = new_gen[5]
        return new_gen

    """ loss calculation"""
    def loss_calculation(self, x):
        loss_table = np.zeros((self.num_sample, 5))
        press_t = x[0]
        angle_t = x[2]

        loss_table[:, 0] = (np.absolute(press_t - self.pressure_ref)).reshape(-1)

        ref_mean = float(self.con[2]) if hasattr(self.con, '__len__') and len(self.con) > 2 else float(self.init_par[2])
        ref_std = float(self.con[3]) if hasattr(self.con, '__len__') and len(self.con) > 3 else 1.0

        loss_table[:, [1, 2]] = np.array([np.absolute(angle_t[:, 0] - ref_mean), np.absolute(angle_t[:, 1] - ref_std)]).T

        return loss_table

    """ dump sample average """
    def dump_parameter(self, x, t, stage):
        with open(self.dump, "a+") as f:
            f.write("epoch: {0:3d}, bond_length: {1:5f}, bond_energy: {2:5f}, angle: {3:5f}, angle_energy: {4:5f}, sigma: {5:5f}, epsilon: {6:5f}, press: {7:5f}, stage: {8:5f}, times: {9:5f}\n".format(
                x, self.bl, self.be, self.al, self.ae, self.sig, self.eps, self.pp, stage, time.time() - t
            ))

    """ dump sample , loss ,pressure """
    def dump_search(self, x, press, error, error2, error3):
        with open(self.dump2, "a+") as f:
            for i in range(0, x.shape[0]):
                f.write("{0:5f}\t{1:5f}\t{2:5f}\t{3:5f}\t{4:5f}\t{5:5f}\t{6:5f}\t{7:5f}\t{8:5f}\t{9:5f}\t{10:5f}\n".format(
                    x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], x[i][5],
                    press[i], error[i, 0], error[i, 1], error2[i], error3[i]
                ))

    def convergence_a(self, x1, x2, threshold):
        newlist = x2[x1]
        return -1 if np.mean(newlist <= threshold) else 1

    def convergence_se(self, x1, x2, threshold1, threshold2, t):
        eps = x2[x1, 4]
        sig = x2[x1, 5]
        eps2 = t[x1, 4]
        sig2 = t[x1, 5]
        if (np.mean(eps) <= threshold1 and np.mean(sig) <= threshold2):
            return -1, [np.mean(eps2), np.mean(sig2)]
        else:
            return 1, [np.mean(eps2), np.mean(sig2)]

    def cal_rdf(self, x):
        eps = np.max(x[0:50])
        hits = np.where(x >= 1)[0]
        if len(hits) > 0:
            s1 = hits[0]
            s1p = 0.05 + s1 * 0.1
            if s1 + 1 < len(x) and (x[s1 + 1] - x[s1]) != 0:
                sig = 0.1 * ((1 - x[s1]) / (x[s1 + 1] - x[s1])) + s1p
            else:
                sig = s1p
        else:
            sig = float(self.init_par[4])
        return np.array([eps, sig])

    def CEM_optimizer_run(self):
        self.create_sample_a(self.init_par)
        properties = self.run()
        loss = self.loss_calculation(properties)
        print("Loss table shape:", loss.shape)
        return properties, loss