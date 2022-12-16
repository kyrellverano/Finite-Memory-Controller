import numpy as np
import sys
import json
import os
import utils
from scipy.special import softmax as softmax
from statistics import median
import matplotlib.pyplot as plt

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000)


if __name__ == "__main__":
    
    # parameters for system are loaded from file 
    print(sys.argv)
    params = json.load(open(sys.argv[1]))

    coarse=params['coarse']   
    p_th=params['thresh']
    M = params["M"]         # size of memory m = {0,1}
    A = params["A"]         # action available in list: {left, right, up, down} 
    
    
    O = params["O"]         # distinct observations for actions [0, 1, 2, .., O-1]
    max_obs = params["max_obs"]  # distinct observations for rewards
    
    symmetry=params['symmetry']
    replica = params["replica"]
    

    Lx = params["Lx"]
    Ly = params["Ly"]
    Lx0 = (Lx/2)-0.5
    Ly0 = params["Ly0"]
    
    if coarse==1 and (Lx % 2) == 0:
        sys.exit('Error in value for Lx. In COARSE set-up, Lx should be an odd number')

    # Tolerance default
    tol_eta = 0.000001
    tol_Q = 0.000001
    lr_th = 0.001
    tol_conv = 0.0000000001

    # Tolerance override
    if ("tol_eta" in params.keys()):
        tol_eta = params["tol_eta"]
    if ("tol_Q" in params.keys()):
        tol_Q = params["tol_Q"]
    if ("lr_th" in params.keys()):
        lr_th = params["lr_th"]
    if ("tol_conv" in params.keys()):
        tol_conv = params["tol_conv"]

    find_range = params["find_range"]
    gamma = params["gamma"]
    V = 100
    if ("V" in params.keys()):
        V = params["V"]

    print('Lx :{}, Ly :{}'.format(Lx, Ly))
    print('Lx0:{}, Ly0:{}'.format(Lx0, Ly0))
    print('M:{}, th:{}, replica:{}'.format(M, p_th,replica))
     
    # combined action space = M * A, lM0, lM1, ... rM0, rM1, ..., sM0, sM1, ...
    a_size = A * M
    L = Lx*Ly

    # cost move
    cost_move = 1-gamma
    if ("cost_move" in params.keys()):
        beta = params["cost_move"]
    reward_find = params["reward_find"]
    
    beta = 5
    if ("beta" in params.keys()):
        beta = params["beta"]

    #Maxtime in optimization
    Ntot = params["Ntot"]
    #Printing time interval
    Nprint = params["Nprint"]


    #data
    if symmetry == 0:
        data=np.loadtxt('data/exp_plume_threshold{}.dat'.format(dth))
    if symmetry == 1:
        data=np.loadtxt('data/exp_plume_symmetric_threshold{}.dat'.format(dth))
    if coarse == 1:
        data=np.loadtxt('data/exp_plume_symmetric_coarse_threshold{}.dat'.format(dth))

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
    elif new==1:
    #load your policy
    #insert after output/ inside the '' the name of the folder 
        name_folder='output/'
        th = np.loadtxt(name_folder + '/file_theta.out')
        th = th.reshape(O, M, a_size)
        Q = np.loadtxt(name_folder + '/file_Q.out')
        eta = np.loadtxt(name_folder + '/file_eta.out')
        pi = softmax(th, axis=2)
    #INITIALIZATIONS
    PObs_lim, RR, PObs = utils.create_PObs_RR(Lx, Ly, Lx0, Ly0, find_range, cost_move, reward_find, M, sigma, max_obs, O, A, V, data)
    PObs_lim = np.abs(PObs_lim)

    #distribution relative to the PObs_lim
    rho0 = np.zeros(M*Lx*Ly)
    rho0[:Lx] = (1-PObs_lim[0,:Lx])/np.sum((1-PObs_lim[0,:Lx]))

    # ++++++++++++++++++
    pi = softmax(th, axis=2)
    print('the average value of the policy is:')
    print(utils.get_value(Q, pi, PObs_lim, L, rho0))

``  #ILLUSTRATE THE POLICY
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
    plt.savefig(name_folder+'/policy_y0.png')


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
    plt.savefig(name_folder+'/policy_y0.png')


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
    plt.savefig(name_folder+'/trajectory.png')
    
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
            
        