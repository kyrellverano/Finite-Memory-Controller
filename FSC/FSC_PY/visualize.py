# Description:
# Visualization of the optimization of the FSC model for the plume tracking task
# ----------------------------------------------------------------------------
# Author: kyrellverano
# LastUpdate: 2019-09-16
#
# Reference:
#
# ----------------------------------------------------------------------------
# Contributors: LuisAlfredoNu <luis.alfredo.nu@gmail.com>
# LastUpdate: 2023-12-12
# ----------------------------------------------------------------------------
#
# Usage: python optimize.py -h
#
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


import sys
import json
import os

import numpy as np
import itertools as it
from scipy.special import softmax as softmax

import time as time

# Path to the comm folder
file_path = os.path.realpath(__file__)
file_path = os.path.dirname(os.path.dirname(file_path))
file_path += '/FSC_requisites/'
sys.path.append(file_path)
import utils as utils
import fsc_visualize_tools as fsc_visual

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000)

# Fix the seed for reproducibility
# np.random.seed(33+33)

import argparse
parser = argparse.ArgumentParser(description='Visualization of Finite State Controller for plume tracking')
parser.add_argument('--dir', type=str,
                    required=True,
                    help='Directory with optimization results')

args = parser.parse_args()
directory = args.dir
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # ----------------------------------------------------------------------------
    # parameters for system are loaded from file 
    params = json.load(open(directory+'/input.dat', 'r'))
    print(params)

    fsc = utils.parameters()
    fsc.set_parameters(params)
    # Print parameters
    utils.print_parameters(fsc,'visualize','fsc','cpu')

    # Load the solver
    solver = utils.solver_opt()
    solver.set_lib_solver('scipy')

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # Load experimental data
    if fsc.plume.symmetry == 0:
        data=np.loadtxt('../FSC_requisites/data/exp_plume_threshold{}.dat'.format(fsc.plume.dth))
    if fsc.plume.symmetry == 1:
        data=np.loadtxt('../FSC_requisites/data/exp_plume_symmetric_threshold{}.dat'.format(fsc.plume.dth))
    if fsc.plume.coarse == 1:
        data=np.loadtxt('../FSC_requisites/data/exp_plume_symmetric_coarse_threshold{}.dat'.format(fsc.plume.dth))

    # Load optimization data
    th = np.loadtxt(directory + '/file_theta.out')
    th = th.reshape(fsc.agent.O, fsc.agent.M, fsc.agent.A * fsc.agent.M)
    pi = softmax(th, axis=2)

    Q = np.loadtxt(directory + '/file_Q.out')

    eta = np.loadtxt(directory + '/file_eta.out')
    # ----------------------------------------------------------------------------
    # Initialize the environment
    # Create array where the source is located
    yxs = it.product(np.arange(fsc.env.Ly), np.arange(fsc.env.Lx))
    find_range2 = fsc.env.find_range**2
    yx_founds = it.filterfalse(lambda x: 
                                (x[0]-fsc.env.Ly0)**2 + (x[1]-fsc.env.Lx0)**2 > find_range2, 
                                yxs)
    source_as_zero = np.array([i for i in yx_founds])

    # Create Observation and Reward matrices
    PObs_lim, RR, PObs, RR_np = utils.create_PObs_RR(fsc.agent, fsc.env, fsc.plume, data=data, source_as_zero=source_as_zero)

    # Create the initial positions of the agent
    rho0 = np.zeros(fsc.env.L * fsc.agent.M)
    rho0[:fsc.env.Lx] = (1-PObs_lim[0,:fsc.env.Lx])/np.sum((1-PObs_lim[0,:fsc.env.Lx]))

    utils.get_max_value(fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.agent.M, fsc.env.find_range, rho0, fsc.agent.gamma)
    # ----------------------------------------------------------------------------

    value =  utils.get_value(Q, pi, PObs_lim, fsc.env.L, rho0)
    print('Value of the policy: {}'.format(value))
    qlty_value = np.log(1+value)/np.log(fsc.agent.gamma)
    print('Quality of the policy: {}'.format(qlty_value))
    print("-"*50)

    # Plot the plume 
    print('Plotting plume...')
    fsc_visual.plot_plume(data, fsc, show=False, save=True, save_path=directory)
    # Plot the initial positions of the agent
    print('Plotting initial positions of the agent...')
    fsc_visual.plot_rho(rho0, fsc, show=False, save=True, save_path=directory)
    # Plot the probability of the observations
    print('Plotting probability of the observations...')
    fsc_visual.plot_PObs_lim(PObs_lim, fsc, show=False, save=True, save_path=directory)
    # Plot the reward
    print('Plotting reward...')
    fsc_visual.plot_reward(RR_np, fsc, show=False, save=True, save_path=directory)

    # Plot eta and Q
    print('Plotting eta and Q...')
    fsc_visual.plot_eta_Q(eta, Q, fsc, show=False, save=True, save_path=directory)

    # Plot policy
    print('Plotting policy...')
    obs_status = ['no signal','signal','differs from signal']
    fsc_visual.plot_policy(pi, fsc, obs_status, show=False, save=True, save_path=directory)
    
    print('Computing trajectory...')
    max_trajectory_length = int(qlty_value)

    trj, result, trj_steps = utils.single_traj_obs(pi, fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.env.find_range, max_trajectory_length, PObs_lim, rho0, fsc.agent.act_hdl,progress_bar=True)

    print('Trajectory length: {} | Success: {}'.format(trj_steps, result))

    # Plot the trajectory of the agent
    print('Plotting trajectory...')
    fsc_visual.plot_trajectory(trj, result, PObs_lim, fsc, show=False, save=True, save_path=directory)

    # Validation of the policy
    print('Validation of the policy...')
    time_start = time.time()
    Nep = 1000   #number of trajectories to reproduce
    maxT = int(qlty_value) * 5

    print('Number episodes: {}, max Time steps: {}'.format(Nep, maxT))

    av_ret, Total_trj = utils.compute_trajectory_future(pi, fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.env.find_range, maxT, PObs_lim, rho0, fsc.agent.act_hdl, Nep=Nep)

    time_end = time.time()
    print('Success Rate: {:.3f}%'.format(100*av_ret / Nep))
    print('Time elapsed: {:.3f} s'.format(time_end - time_start))
