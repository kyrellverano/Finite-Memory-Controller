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
# LastUpdate: 2023-08-25
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
from statistics import median
import matplotlib.pyplot as plt

sys.path.append('../Comm/')
import utils as utils


np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000)

# Fix the seed for reproducibility
# np.random.seed(33+33)


import argparse
parser = argparse.ArgumentParser(description='Visualization of Finite State Controller for plume tracking')
parser.add_argument('--directory', type=str,
                    required=True,
                    help='Input file with parameters')


args = parser.parse_args()
directory = args.directory


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def plot_policy(pi,M,O,act_hdl,obs_status,directory,round_size=3):

    # Create labels
    actions = ["m{:d}-{}".format(i+1,act_hdl.action_name(j)) for i in range(M) for j in range(act_hdl.A)]
    memory = ["Mem{:d}".format(i+1) for i in range(M)]

    # List of styles for each observation
    styles = ['spring','summer','autumn','winter','cool','hot','bone','copper','pink','gray','flag','prism','ocean']

    cmap_style = ['viridis', 'plasma', 'cividis']
    cmap_style = ['Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r']
    
    # Loop over observations
    for o in range(O):
        pn = pi[o].reshape(M,M*act_hdl.A)
        
        # Create the plot
        fig, ax = plt.subplots()
        im = ax.imshow(pn,cmap = cmap_style[o])
        # Rotate the tick labels and set their alignment.

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(actions)), labels=actions)
        ax.set_yticks(np.arange(len(memory)), labels=memory)

        pnn=np.round(pn,round_size)
        for i in range(len(memory)):
            for j in range(len(actions)):
                ax.text(j, i, pnn[i, j], ha="center", va="center", color="black")

        plt.title('policy p(m*,a|m,y={}) when {}'.format(o,obs_status[o]))
        plt.ylabel('initial memory(m)',fontsize=10)
        plt.xlabel('memory update and action(m*,a)',fontsize=10)
        fig.set_size_inches(10, 6)

        plt.savefig(directory+'/policy_o{}.png'.format(o), bbox_inches = 'tight', pad_inches = 1)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def plot_plume(fsc, data):
    if fsc.plume.experimental :
        cplume = utils.create_plume_from_exp(fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.plume.beta, data)
        title = f'Experimental plume\nLx:{fsc.env.Lx}, Ly:{fsc.env.Ly}'
        info = f'beta:{fsc.plume.beta}\ndth:{fsc.plume.dth}'
    else:
        cplume = utils.create_cplume(fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.plume.D, fsc.plume.V, fsc.plume.tau, fsc.plume.beta, alpha=1.0 / fsc.plume.adjust_factor)
        title = f'Model plume\nLx:{fsc.env.Lx}, Ly:{fsc.env.Ly}'
        info = f'beta:{fsc.plume.beta}\nD:{fsc.plume.D}\nV:{fsc.plume.V}\ntau:{fsc.plume.tau}\nalpha:{fsc.plume.adjust_factor:.2f}'

    min_val_2_plot = 0.1
    plt.imshow(cplume.reshape(fsc.env.Ly, fsc.env.Lx), cmap='jet',origin='lower',vmax=min_val_2_plot)
    plt.title(title)
    # put a box with info
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    plt.text(-0.55, 0.95, info, transform=plt.gca().transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.xlabel('x-position')
    plt.ylabel('y-position')
    plt.colorbar()
    # plt.show()  
    plt.savefig(directory+'/plume.png', bbox_inches = 'tight', pad_inches = 0.1)


if __name__ == "__main__":
    
    # ----------------------------------------------------------------------------
    # parameters for system are loaded from file 
    params = json.load(open(directory+'/input.dat', 'r'))
    print(params)

    fsc = utils.parameters()
    fsc.set_parameters(params)
    # Print parameters
    utils.print_parameters(fsc,'visualize','fsc','cpu')



    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # Load experimental data
    if fsc.plume.symmetry == 0:
        data=np.loadtxt('data/exp_plume_threshold{}.dat'.format(fsc.plume.dth))
    if fsc.plume.symmetry == 1:
        data=np.loadtxt('data/exp_plume_symmetric_threshold{}.dat'.format(fsc.plume.dth))
    if fsc.plume.coarse == 1:
        data=np.loadtxt('data/exp_plume_symmetric_coarse_threshold{}.dat'.format(fsc.plume.dth))

    # Load optimization data
    th = np.loadtxt(directory + '/file_theta.out')
    th = th.reshape(fsc.agent.O, fsc.agent.M, fsc.agent.A * fsc.agent.M)
    pi = softmax(th, axis=2)

    Q = np.loadtxt(directory + '/file_Q.out')

    eta = np.loadtxt(directory + '/file_eta.out')
    # ----------------------------------------------------------------------------

    # Compute
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

    # Compute the trajectory of the agent

    # trj -> [[action, x, y, memory, reward, observation], [...], [...], ...]

    max_trajectory_length = 60000
    trj, result, trj_steps = utils.single_traj_obs(pi, fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.env.find_range, max_trajectory_length, PObs_lim, rho0, fsc.agent.act_hdl,progress_bar=True)
    print('Trajectory length: {}, {} | Success: {}'.format(len(trj), trj_steps, result))

    # ----------------------------------------------------------------------------
    utils.get_max_value(fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.env.find_range, rho0, fsc.agent.gamma)

    value =  utils.get_value(Q, pi, PObs_lim, fsc.env.L, rho0)
    print('Value of the policy: {}'.format(value))
    print("-"*50)

    # Plot plume
    print('Plotting plume...')
    # plot_plume(fsc, data)

    # Plot policy
    print('Plotting policy...')
    obs_status = ['no signal','signal','differs from signal']
    # plot_policy(p     i, fsc.agent.M, fsc.agent.O, fsc.agent.act_hdl, obs_status, directory)

    # Plot trajectory
    print('Plotting trajectory...')

    actions = trj[1:,0].astype(int)
    pos_x = trj[1:,1].astype(int)
    pos_y = trj[1:,2].astype(int)
    memory = trj[1:,3].astype(int)
    observation = trj[1:,4]

    cdict = {2: 'orange', 0: 'mediumseagreen', 1: 'crimson', 3:'blue'}
    
    fig, ax = plt.subplots()

    loc_obs = np.where(observation > 0)
    loc_no_obs = np.where(observation == 0)
    ax.scatter(pos_x[loc_obs], pos_y[loc_obs], c = 'black', s = 40, zorder = 2, alpha = 0.5)

    mem0 = memory[0]
    t_mem0 = [0,0]

    for a in range(len(actions)):
        if memory[a] != mem0:
            t_mem0[1] = a
            ax.plot(pos_x[t_mem0[0]:t_mem0[1]], pos_y[t_mem0[0]:t_mem0[1]], c = cdict[mem0], linewidth = 2, zorder = 3)
            mem0 = memory[a]
            t_mem0[0] = a - 1
    
    print('imshow')
    #ax.legend()
    ax.imshow((1-PObs_lim[0,:]-5).reshape(fsc.agent.M,fsc.env.Ly,fsc.env.Lx)[0,:],cmap='Greys',origin='lower')
    # fig.set_size_inches(10, 6)
    crange=plt.Circle((fsc.env.Lx0,fsc.env.Ly0),fsc.env.find_range,fill=False)
    ax.add_artist(crange)
    # ax.set_aspect(1)
    
    plt.title('M: {} A: {} O: {}'.format(fsc.agent.M, fsc.agent.A, fsc.agent.O))
    # plt.xlim((-1,Lx))
    # plt.ylim((-1,Ly))
    plt.xlabel('x-position',fontsize=16)
    plt.ylabel('y-position',fontsize=16)
    plt.savefig(directory+'/trajectory.png', bbox_inches = 'tight', pad_inches = 0.1)
    # plt.show()


    # make a video de la tajectory 
        
















    #++++++++
    #INITIALIZATIONS
    
    eta = np.zeros(L*M)
    PObs_lim, RR, PObs = utils.create_PObs_RR(Lx, Ly, Lx0, Ly0, find_range, cost_move, 
                                        reward_find, M, beta, max_obs, O, A, V, data=data)
    PObs_lim = np.abs(PObs_lim)

    # Initial density is only at y=0, proportional to signal probability
    rho0 = np.zeros(M*Lx*Ly)
    #rho0[:Lx*Ly0//2] = (1-PObs_lim[0,:Lx*Ly0//2])/np.sum((1-PObs_lim[0,:Lx*Ly0//2]))
    rho0[:Lx] = (1-PObs_lim[0,:Lx])/np.sum((1-PObs_lim[0,:Lx]))

    
    if ("ev" in params.keys()):
        rho0[:Lx*Ly0//2] = (1-PObs_lim[0,:Lx*Ly0//2])/np.sum((1-PObs_lim[0,:Lx*Ly0//2]))
    else:
        rho0[:Lx] = (1-PObs_lim[0,:Lx])/np.sum((1-PObs_lim[0,:Lx]))
    
    print('tol_conv:{}, tolQ:{}, toleta:{}'.format(tol_conv,tol_Q,tol_eta)) 
        
    name_folder = 'outputs/Visualize_A{}M{}b{}g{}FR{}LR{}sym{}th{}rep{}LX{}LY{}co{}'.format(A,M,beta,gamma,find_range,lr_th,symmetry,dth,replica,Lx,Ly,coarse)
    os.makedirs(name_folder, exist_ok=True)
    #os.system("cp {} {}".format('./'+sys.argv[1], './'+name_folder))
    
    
    # Policy Initialization with bias
    new = params["new_policy"]
    if ("sub" in params.keys()):
        sub = params["sub"] 

    #new=0 means the saved optimal policy we provided
    #new=1 means the policy you saved as a result
    #sub= 0 : best, 1: second best 
    if new==0:
    #load optimized policy
        name_folder='saved_policies/A{}M{}TH{}sub{}co{}'.format(A,M,dth,sub,coarse)
        th = np.loadtxt(name_folder + '/file_theta.out')
        th = th.reshape(O, M, a_size)
        Q = np.loadtxt(name_folder + '/file_Q.out')
        eta = np.loadtxt(name_folder + '/file_eta.out')
        save_folder='outputs/A{}M{}TH{}sub{}co{}'.format(A,M,dth,sub,coarse)
    elif new==1:
    #load your policy
    #insert after output/ inside the '' the name of the folder 
        name_folder='output/'
        th = np.loadtxt(name_folder + '/file_theta.out')
        th = th.reshape(O, M, a_size)
        Q = np.loadtxt(name_folder + '/file_Q.out')
        eta = np.loadtxt(name_folder + '/file_eta.out')
        pi = softmax(th, axis=2)
        save_folder=name_folder

    os.makedirs(save_folder, exist_ok=True)
    #INITIALIZATIONS
    PObs_lim, RR, PObs = utils.create_PObs_RR(Lx, Ly, Lx0, Ly0, find_range, cost_move, reward_find, M, beta, max_obs, O, A, V, data)
    PObs_lim = np.abs(PObs_lim)

    #distribution relative to the PObs_lim
    rho0 = np.zeros(M*Lx*Ly)
    rho0[:Lx] = (1-PObs_lim[0,:Lx])/np.sum((1-PObs_lim[0,:Lx]))

    # ++++++++++++++++++
    pi = softmax(th, axis=2)
    print('the average value of the policy is:')
    print(utils.get_value(Q, pi, PObs_lim, L, rho0))

    #ILLUSTRATE THE POLICY
    # for y=0
    p0=pi.reshape(O,M,M,A)[0].reshape(M,M*A)
    actions = ["m1-L", "m1-R", "m1-U", "m1-D","m2-L", "m2-R", "m2-U", "m2-D","m3-L", 
            "m3-R", "m3-U", "m3-D","m4-L", "m4-R", "m4-U", "m4-D","m5-L", "m5-R", "m5-U", "m5-D"]
    memory = ["Mem1", "Mem2", "Mem3",
            "Mem4", "Mem5"]

    fig, ax = plt.subplots()
    im = ax.imshow(p0,cmap='summer')
    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(actions[:M*A])), labels=actions[:M*A])
    ax.set_yticks(np.arange(len(memory[:M])), labels=memory[:M])

    p00=np.round(p0,2)
    for i in range(len(memory[:M])):
        for j in range(len(actions[:M*A])):
            if p00[i,j]==0.0:
                text = ax.text(j, i, 0,
                        ha="center", va="center", color="black") 
            else:
                text = ax.text(j, i, p00[i, j],
                        ha="center", va="center", color="black")
                
    plt.title('policy p(m*,a|m,y=0) when no observation')
    plt.ylabel('initial memory(m)',fontsize=10)
    plt.xlabel('memory update and action(m*,a)',fontsize=10)
    fig.set_size_inches(10, 6)
    plt.show()
    plt.savefig(save_folder+'/policy_y0.png')


    #for y=1
    p1=pi.reshape(O,M,M,A)[1].reshape(M,M*A)
    actions = ["m1-L", "m1-R", "m1-U", "m1-D","m2-L", "m2-R", "m2-U", "m2-D","m3-L", 
            "m3-R", "m3-U", "m3-D","m4-L", "m4-R", "m4-U", "m4-D","m5-L", "m5-R", "m5-U", "m5-D"]
    memory = ["Mem1", "Mem2", "Mem3",
            "Mem4", "Mem5"]

    fig, ax = plt.subplots()
    im = ax.imshow(p1,cmap='summer')
    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(actions[:M*A])), labels=actions[:M*A])
    ax.set_yticks(np.arange(len(memory[:M])), labels=memory[:M])

    p11=np.round(p1,2)
    for i in range(len(memory[:M])):
        for j in range(len(actions[:M*A])):
            if p11[i,j]==0.0:
                text = ax.text(j, i, '0',
                        ha="center", va="center", color="black") 
            else:
                text = ax.text(j, i, p11[i, j],
                        ha="center", va="center", color="black")
            
    plt.title('policy p(m*,a|m,y=1) when there is observation')
    plt.ylabel('initial memory(m)',fontsize=10)
    plt.xlabel('memory update and action(m*,a)',fontsize=10)
    fig.set_size_inches(10, 6)
    plt.show()
    plt.savefig(save_folder+'/policy_y1.png')


    ##TRAJECTORY sample
    no_samples=1
    trj, ret, _ = utils.single_traj_obs(softmax(th, axis=2), Lx, Ly, Lx0, Ly0, find_range, gamma, PObs_lim, rho0, A)

    scatter_x = trj[1:,1]
    scatter_y = trj[1:,2]
    group = trj[1:,3]
    cdict = {2: 'orange', 0: 'mediumseagreen', 1: 'crimson', 3:'blue'}

    fig, ax = plt.subplots()
    ix = np.where(trj[1:,4] > 0)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = 'black', s = 80)
    ix = np.where(group == 0.0)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[0], label = 'Memory 0', s = 8)
    ix = np.where(group == 1.0)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[1], label = 'Memory 1', s = 8)
    ix = np.where(group == 2.0)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[2], label = 'Memory 2', s = 8)
    ix = np.where(group == 3.0)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[3], label = 'Memory 3', s = 8)
    #ax.legend()
    ax.imshow((1-PObs_lim[0,:]-5).reshape(M,Ly,Lx)[0,:],cmap='Greys')
    fig.set_size_inches(10, 6)
    crange=plt.Circle((Lx0,Ly0),find_range,fill=False)
    ax.set_aspect(1)
    ax.add_artist(crange)
    plt.title('{} threshold: {} memory states'.format(dth, M))
    plt.xlim((-1,92))
    plt.ylim((-1,133))
    plt.xlabel('x-position',fontsize=16)
    plt.ylabel('y-position',fontsize=16)
    plt.show()
    plt.savefig(save_folder+'/trajectory.png')
    
    if ("no_samples" in params.keys()):
        no_samples = params["no_samples"]
    if no_samples>1:
        for i in range(no_samples):
            trj, ret, _ = utils.single_traj_obs(softmax(th, axis=2), Lx, Ly, Lx0, Ly0, find_range, gamma, PObs_lim, rho0, A)
            scatter_x = trj[1:,1]
            scatter_y = trj[1:,2]
            group = trj[1:,3]
            cdict = {2: 'orange', 0: 'mediumseagreen', 1: 'crimson', 3:'blue'}

            fig, ax = plt.subplots()
            ix = np.where(trj[1:,4] > 0)
            ax.scatter(scatter_x[ix], scatter_y[ix], c = 'black', s = 80)
            ix = np.where(group == 0.0)
            ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[0], label = 'Memory 0', s = 8)
            ix = np.where(group == 1.0)
            ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[1], label = 'Memory 1', s = 8)
            ix = np.where(group == 2.0)
            ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[2], label = 'Memory 2', s = 8)
            ix = np.where(group == 3.0)
            ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[3], label = 'Memory 3', s = 8)
            #ax.legend()
            ax.imshow((1-PObs_lim[0,:]-5).reshape(M,Ly,Lx)[0,:],cmap='Greys')
            fig.set_size_inches(10, 6)
            crange=plt.Circle((Lx0,Ly0),find_range,fill=False)
            ax.set_aspect(1)
            ax.add_artist(crange)
            plt.title('{} threshold: {} memory states'.format(dth, M))
            plt.xlim((-1,92))
            plt.ylim((-1,133))
            plt.xlabel('x-position',fontsize=16)
            plt.ylabel('y-position',fontsize=16)
            plt.savefig(name_folder+'/trajectory_{}.png'.format(i))
            
        