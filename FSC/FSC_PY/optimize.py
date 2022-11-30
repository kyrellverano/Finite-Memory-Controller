import numpy as np
import sys
import json
import os
import utils
from scipy.special import softmax as softmax

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000)


if __name__ == "__main__":
    
    # parameters for system are loaded from file 
    print(sys.argv)
    params = json.load(open(sys.argv[1]))
 
    
    #changed parameters
    params2 = json.load(open(sys.argv[2]))
       
    replica = params["replica"]
    

    Lx = params["Lx"]
    Ly = params["Ly"]
    Lx0 = params["Lx0"]
    Ly0 = params["Ly0"]

    print('Lx :{}, Ly :{}'.format(Lx, Ly))
    print('Lx0:{}, Ly0:{}'.format(Lx0, Ly0))

    find_range = params["find_range"]

    M = params["M"]         # size of memory m = {0,1}
    A = params["A"]         # action available in list: {left, right, up, down}
    O = params["O"]         # distinct observations for actions [0, 1, 2, .., O-1]
    max_obs = params["max_obs"]  # distinct observations for rewards
    p_th=params['thresh']
    symmetry=params['symmetry']
    coarse=params['coarse']


    #change memories
    if ("M" in params2.keys()):
        M = params2["M"]
    #change replicas
    if ("replica" in params2.keys()):
        replica = params2["replica"]


    print('M:{}, th:{}, replica:{}'.format(M, p_th,replica))
     
    #data
    #data_file = params["data_file"]
    #data=np.loadtxt(data_file)
    #data
    if symmetry == 0:
        data=np.loadtxt('exp_plume_threshold{}.dat'.format(p_th))
    if symmetry == 1:
        data=np.loadtxt('exp_plume_symmetric_threshold{}.dat'.format(p_th))
    if coarse == 1:
        data=np.loadtxt('exp_plume_symmetric_coarse_threshold{}.dat'.format(p_th))
    
    # combined action space = M * A, lM0, lM1, ... rM0, rM1, ..., sM0, sM1, ...
    a_size = A * M
    L = Lx*Ly

    # gamma
    gamma = params["gamma"]
    if ("gamma" in params2.keys()):
        gamma = params2["gamma"]

    # cost move
    cost_move = 1-gamma
    reward_find = params["reward_find"]
    
    #cost_wall = cost_move * 10

    beta = 5
    if ("beta" in params.keys()):
        beta = params["beta"]
    
    V = 100
    if ("V" in params.keys()):
        V = params["V"]

    eta = np.zeros(L*M)

    PObs_lim, RR, PObs = utils.create_PObs_RR(Lx, Ly, Lx0, Ly0, find_range, cost_move, 
                                        reward_find, M, beta, max_obs, O, A, V, data=data)
    PObs_lim = np.abs(PObs_lim)

    # Initial density is only at y=0, proportional to signal probability
    rho0 = np.zeros(M*Lx*Ly)
    
    if ("ev" in params.keys()):
        rho0[:Lx*Ly0//2] = (1-PObs_lim[0,:Lx*Ly0//2])/np.sum((1-PObs_lim[0,:Lx*Ly0//2]))
    else:
        rho0[:Lx] = (1-PObs_lim[0,:Lx])/np.sum((1-PObs_lim[0,:Lx]))
    
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
        
    #replicas
    
    replica = params2["replica"]

    


    
    print('tol_conv:{}, tolQ:{}, toleta:{}'.format(tol_conv,tol_Q,tol_eta)) 
    
    name_folder = 'ExpPlume_A{}M{}b{}g{}FR{}LR{}sym{}th{}rep{}LX{}LY{}'.format(A,M,beta,gamma,find_range,lr_th,symmetry,p_th,replica,Lx,Ly)
    os.makedirs(name_folder, exist_ok=True)
    os.system("cp {} {}".format('./'+sys.argv[1], './'+name_folder))
    
    f = open(name_folder+"/values.dat", "a")
    
    
    # Policy Initialization with bias
    restart = False
    
    if ("restart" in params.keys()):
        restart = bool(params["restart"])
        #folder_restart = params["folder_restart"]
        folder_restart = name_folder
        
    if (not restart):
        th = (np.random.rand(O, M, a_size)-0.5)*0.5 

        #th[0,:,2::A] += 0.5
        th[1:,:,2::A] += 0.5
        if ("unbiased" in params.keys() ):
            th[1:,:,2::A] -= 0.5

        eta = np.ones(eta.shape)/(L*M)/(1-gamma)
        Q = utils.create_random_Q0(Lx, Ly, Lx0, Ly0, gamma, a_size, M, cost_move, reward_find) 
    else:
        th = np.loadtxt(folder_restart + '/file_theta.out')
        th = th.reshape(O, M, a_size)
        Q = np.loadtxt(folder_restart + '/file_Q.out')
        eta = np.loadtxt(folder_restart + '/file_eta.out')
     
    pi = softmax(th, axis=2)

    value = 0
    oldvalue = value
    
    Ntot = params["Ntot"]
    Nprint = params["Nprint"]
    

   
    for t in range(Ntot):

        pi = softmax(th, axis=2)

        # Iterative solutions of linear system
        eta = utils.iterative_solve_eta(pi, PObs_lim, gamma, rho0, eta, tol_eta, Lx, Ly, Lx0, Ly0, find_range)
        Q = utils.iterative_solve_Q(pi, PObs_lim, gamma, RR, Q, tol_Q, Lx, Ly, Lx0, Ly0, find_range, cost_move)
        
        # Gradient calculation
        grad = utils.find_grad(pi, Q, eta, L, PObs_lim)
        grad -= np.max(grad, axis=2, keepdims=True)

        # Reset value for new iteration afterwards
        
        th += grad * lr_th / np.max(np.abs(grad)) # (t / Ntot + 0.5) #rescaled gradient
        th -= np.max(th, axis=2, keepdims=True)
        th = np.clip(th, -20, 0)
        
        # Print and check convergence
        if (t % Nprint == 0):
            value =  utils.get_value(Q, pi, PObs_lim, L, rho0)

            #print
            #print('\n\n\n', pi)
            f.write('current value: {} @ time:{} \n'.format(value, t))
            f.flush()

            # check convergence
            if (abs((value-oldvalue)/value)<tol_conv):
                f.write('converged at T={}'.format(t))
                break
            oldvalue = value

        #if (t % (Nprint*10) == 0):
            #np.savetxt(name_folder + '/file_theta{}.out'.format(t), th.reshape(-1))
            #np.savetxt(name_folder + '/file_Q{}.out'.format(t), Q.reshape(-1))
            #np.savetxt(name_folder + '/file_eta{}.out'.format(t), eta.reshape(-1))
        if (t % (Nprint*10) == 0):
            np.savetxt(name_folder + '/file_theta.out', th.reshape(-1))
            np.savetxt(name_folder + '/file_Q.out', Q.reshape(-1))
            np.savetxt(name_folder + '/file_eta.out', eta.reshape(-1))

    # final print
    np.savetxt(name_folder + '/file_theta.out', th.reshape(-1))
    np.savetxt(name_folder + '/file_Q.out', Q.reshape(-1))
    np.savetxt(name_folder + '/file_eta.out', eta.reshape(-1))
