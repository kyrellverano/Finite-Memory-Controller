import numpy as np

from scipy.special import softmax as softmax
import matplotlib.pyplot as plt

import time 
from tqdm import tqdm

# Import the local modules
import utils as utils

# ----------------------------------------------------------------------------
def plot_plume(data, fsc, save=False, show=True, save_path=None, imkwargs={}):
    """ Plot the plume
    """

    # Get the plume
    if fsc.plume.experimental :
        # Create the plume from experimental data
        plume = utils.create_plume_from_exp(fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.plume.beta, data)
        title = f'Plume from exp'
    else:
        # Create the plume from the model
        plume = utils.create_cplume(fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.plume.D, fsc.plume.V, fsc.plume.tau, fsc.plume.beta, fsc.plume.adjust_factor)
        title = f'Plume from model'

    title = title + f'\n(Lx:{fsc.env.Lx}, Ly:{fsc.env.Ly})'

    # Create the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Plot the plume
    im = ax.imshow(plume, cmap='jet',origin='lower', **imkwargs)
    # Add a colorbar
    fig.colorbar(im)

    # Set the title
    ax.set_title(title, fontsize=20)

    # Set the labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Save the figure
    if save:
        if save_path is None:
            assert False, 'save_path is None, please specify a save_path'
        save_path = save_path + f'/plume.png'
        plt.savefig(save_path)

    # Show the figure
    if show:
        plt.show()

    # Close the figure
    plt.close(fig)

# ----------------------------------------------------------------------------
def plot_rho(rho, fsc, save=False, show=True, save_path=None, imkwargs={}):
    """ Plot the initial positions of the agent
    """
    
    Lx = fsc.env.Lx
    Ly = fsc.env.Ly
    M = fsc.agent.M

    rho_map = rho.reshape((Ly,Lx,M))

    # Create the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Plot rho
    im = ax.imshow(rho_map[:,:,0], origin='lower', **imkwargs)
    # Add a colorbar
    fig.colorbar(im)

    # Set the title
    ax.set_title(f'Initial positions of the agent', fontsize=20)

    # Set the labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Save the figure
    if save:
        if save_path is None:
            assert False, 'save_path is None, please specify a save_path'
        save_path = save_path + f'/rho.png'
        plt.savefig(save_path)

    # Show the figure
    if show:
        plt.show()

    # Close the figure
    plt.close(fig)

# ----------------------------------------------------------------------------
def plot_PObs_lim(PObs_lim, fsc, save=False, show=True, save_path=None, imkwargs={}):
    """ Plot the PObs_lim
    """

    Lx = fsc.env.Lx
    Ly = fsc.env.Ly
    M = fsc.agent.M
    O = fsc.agent.O

    PObs_lim_map = PObs_lim.reshape((O,M,Ly,Lx))

    # Create the figure
    fig = plt.figure(figsize=(12, 12//O))
    for o in range(O):
        ax = fig.add_subplot(1,O,o+1)

        # Plot PObs_lim
        im = ax.imshow(PObs_lim_map[o,0,:,:], origin='lower', **imkwargs)
        # Add a colorbar
        fig.colorbar(im)

        # Set the title
        ax.set_title(f'PObs_lim, o:{o}', fontsize=20)

        # Set the labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Save the figure
    if save:
        if save_path is None:
            assert False, 'save_path is None, please specify a save_path'
        save_path = save_path + f'/PObs_lim.png'
        plt.savefig(save_path)

    # Show the figure
    if show:
        plt.show()

    # Close the figure
    plt.close(fig)

# ----------------------------------------------------------------------------
def plot_reward(RR, fsc, save=False, show=True, save_path=None, imkwargs={}):
    """ Plot the reward
    """

    Lx = fsc.env.Lx
    Ly = fsc.env.Ly
    M = fsc.agent.M

    RR_map = RR.reshape((M,Ly,Lx))

    # Create the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Plot the reward
    im = ax.imshow(RR_map[0,:,:], origin='lower', **imkwargs)
    # Add a colorbar
    fig.colorbar(im)

    # Set the title
    ax.set_title(f'Reward', fontsize=20)

    # Set the labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Save the figure
    if save:
        if save_path is None:
            assert False, 'save_path is None, please specify a save_path'
        save_path = save_path + f'/reward.png'
        plt.savefig(save_path)

    # Show the figure
    if show:
        plt.show()

    # Close the figure
    plt.close(fig)

# ----------------------------------------------------------------------------
def plot_eta_Q(eta_init, Q_init, fsc, save=False, show=True, save_path=None):
    """ Plot the values of eta and Q
    """

    Lx = fsc.env.Lx
    Ly = fsc.env.Ly
    M = fsc.agent.M
    A = fsc.agent.A

    eta = eta_init.reshape(M,Ly,Lx)
    Q = Q_init.reshape(M,Ly,Lx,M,A)

    cmap_style = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    for im in range(M):
        fig = plt.figure(figsize=(16,16//(A+1)))

        ax = fig.add_subplot(1,A+1,1)

        # Plot the eta
        img = ax.imshow(eta[im],cmap=cmap_style[im],origin='lower')
        # Add a colorbar
        fig.colorbar(img)

        # Set the title
        ax.set_title(r'$\eta$ | M: {}'.format(im))

        for ia in range(A):
            ax = fig.add_subplot(1,A+1,ia+2)
            # Plot the Q
            img = ax.imshow(Q[0,:,:,im,ia],cmap=cmap_style[im]+'_r',origin='lower')
            # Add a colorbar
            fig.colorbar(img)

            # Set the title
            ax.set_title("Q | M: {}, Action: {}".format(im,ia))

        # Save the figure
        if save:
            if save_path is None:
                assert False, 'save_path is None, please specify a save_path'
            plt.savefig(save_path + f'/eta_Q_m{str(im)}.png')

        # Show the figure
        if show:
            plt.show()

        # Close the figure
        plt.close(fig)

# ----------------------------------------------------------------------------

def plot_policy(pi, fsc, obs_status, round_size=3, save=False, show=True, save_path=None):
    """ Plot the policy
    """

    M = fsc.agent.M
    O = fsc.agent.O
    act_hdl = fsc.agent.act_hdl

    # Create labels
    actions = ["m{:d}-{}".format(i+1,act_hdl.action_name(j)) for i in range(M) for j in range(act_hdl.A)]
    memory = ["Mem{:d}".format(i+1) for i in range(M)]

    # List of styles for each observation
    cmap_style = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    
    # Loop over observations
    for o in range(O):
        pn = pi[o].reshape(M,M*act_hdl.A)
        
        # Create the plot
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(act_hdl.A*3,M*1))
        # Put close the columns of plots
        fig.subplots_adjust(wspace=0.0)
        # Set the title
        fig.suptitle(r'Policy $p(m*,a|m,y={})$ | Observation: {}'.format(o,obs_status[o]),fontsize=16)

        max_val = np.max(pn)
        min_val = np.min(pn)
        # Loop over the plots
        for i, (ax, pni) in enumerate(zip(axs.flat, pn)):
            pni = pni.reshape(M,act_hdl.A)
            img = ax.imshow(pni, cmap=cmap_style[i]+'_r', vmin=min_val, vmax=max_val)

            # Disable the axis ticks and labels 
            if i > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel('initial memory(m)',fontsize=10)
                ax.set_xlabel('memory update and action(m*,a)',fontsize=10)
                ax.set_yticks(np.arange(len(memory)), labels=memory)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", \
                     rotation_mode="anchor")
        
            # # Show all ticks and label them with the respective list entries
            ax.set_xticks(np.arange(act_hdl.A), labels=actions[i*act_hdl.A:(i+1)*act_hdl.A])

            # Turn spines off and create white grid.
            for edge, spine in ax.spines.items():
                spine.set_visible(False)

            # ax.set_xticks(np.arange(act_hdl.A,pn.shape[1],act_hdl.A)-.5, minor=True)
            ax.set_xticks(np.array([act_hdl.A])-.5, minor=True)
            ax.grid(which="minor", color="y", linestyle='-', linewidth=4)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Threshold for text color resp. background color
            threshold = img.norm(pn.max())/2.

            # Loop over data dimensions and create text annotations.
            pnn=np.round(pni,round_size)
            for i in range(len(memory)):
                for j in range(act_hdl.A):
                    text_color = "black" if img.norm(pnn[i, j]) > threshold else "white"
                    ax.text(j, i, pnn[i, j], ha="center", va="center", color=text_color)

        # Save the figure
        if save:
            if save_path is None:
                assert False, 'save_path is None, please specify a save_path'
            # save_path = save_path + f'/policy.png'
            plt.savefig(save_path + f'/policy_o{str(o)}.png')

        # Show the figure
        if show:
            plt.show()

        # Close the figure
        plt.close(fig)

# ----------------------------------------------------------------------------
def plot_trajectory(trj, result, PObs_lim, fsc, show=True, save=False, save_path=''):
    ''' Plot the trajectory of the agent
    '''
    actions = trj[1:,0].astype(int)
    pos_x = trj[1:,1].astype(int)
    pos_y = trj[1:,2].astype(int)
    memory = trj[1:,3].astype(int)
    observation = trj[1:,4]

    trj_steps = len(trj)

    cdict = {2: 'orange', 0: 'mediumseagreen', 1: 'crimson', 3:'blue'}

    # Get the position of the observations
    loc_obs = np.where(observation == 1)
    loc_no_obs = np.where(observation == 0)

    # Create figure    
    fig, ax = plt.subplots()

    # Plot the position of the observations
    ax.scatter(pos_x[loc_obs], pos_y[loc_obs], c = 'black', s = 40, zorder = 2, alpha = 0.5)

    # Initial memory and position
    mem0 = memory[0]
    t_mem0 = [0,0]

    # Plot the trajectory for each memory state
    for a in range(len(actions)):
        if memory[a] != mem0:
            t_mem0[1] = a
            ax.plot(pos_x[t_mem0[0]:t_mem0[1]], pos_y[t_mem0[0]:t_mem0[1]], c = cdict[mem0], linewidth = 2, zorder = 3)
            mem0 = memory[a]
            t_mem0[0] = a - 1
    ax.plot(pos_x[t_mem0[0]:], pos_y[t_mem0[0]:], c = cdict[mem0], linewidth = 2, zorder = 3)

    # ax.plot(pos_x, pos_y, c = 'blue', linewidth = 4, zorder = 2)
    
    # Plot the probability of the observations
    ax.imshow((1-PObs_lim[0,:]-5).reshape(fsc.agent.M,fsc.env.Ly,fsc.env.Lx)[0,:],cmap='Greys',origin='lower')

    # Plot the plume source
    crange=plt.Circle((fsc.env.Lx0,fsc.env.Ly0),fsc.env.find_range,fill=False)
    ax.add_artist(crange)
    
    # Add title and labels  
    ax.set_title(f'Trajectory | steps: {trj_steps}\n Success: {result} | M: {fsc.agent.M} A: {fsc.agent.A} O: {fsc.agent.O}')
    # plt.title('M: {} A: {} O: {}'.format(fsc.agent.M, fsc.agent.A, fsc.agent.O))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    if save:
        if save_path is None:
            assert False, 'save_path is None, please specify a save_path'
        plt.savefig(save_path + '/single_trj.png')
    if show:
        plt.show()
    plt.close()
