# Description:
# Optimization of the FSC model for the plume tracking task
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
# Usage: python optimize.py input.dat
#
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

import sys
import os
import json
import time

import numpy as np
import itertools as it
from scipy.special import softmax as softmax

sys.path.append('../Comm/')
import utils as utils

# ----------------------------------------------------------------------------
# Select the library to solve the linear system 
# lib_solve_linear_system = 'scipy'
# lib_solve_linear_system = 'petsc'
# lib_solve_linear_system = 'cupy'
lib_solve_linear_system = sys.argv[2]

solver = utils.solver_opt()
solver.set_lib_solver(lib_solve_linear_system)

mpi_rank = solver.mpi_rank
mpi_size = solver.mpi_size
mpi_comm = solver.mpi_comm

if sys.argv[4] :
    solver.device = sys.argv[4]

device = solver.device

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000)

# Fix the seed for reproducibility
np.random.seed(33+33)

# @profile
def optimize():
    
    # parameters for system are loaded from file 
    if mpi_rank == 0:
        print(sys.argv)
        # params = json.load(open(sys.argv[1]))
        params = json.load(open('input.dat'))
    else:
        params = None
    
    if mpi_size > 1:
        # broadcast parameters to all processes
        params = mpi_comm.bcast(params, root=0)

    coarse=params['coarse']   
    dth=params['thresh']
    M = params["M"]         # size of memory m = {0,1}

    A = params["A"]         # action size
    act_move = params["actions_move"]
    act_name = params["actions_name"]
    
    O = params["O"]         # distinct observations for actions [0, 1, 2, .., O-1]
    max_obs = params["max_obs"]  # distinct observations for rewards
    
    symmetry=params['symmetry']
    replica = params["replica"]
    

    Lx = params["Lx"]
    Ly = params["Ly"]
    Ly0 = params["Ly0"]

    factor_size = int(sys.argv[3])
    Lx = int(Lx*factor_size)
    Ly = int(Ly*factor_size)
    Ly0 = Ly0*factor_size
    Lx0 = (Lx/2)-0.5
    
    if coarse==1 and (Lx % 2) == 0:
        sys.exit('Error in value for Lx. In COARSE set-up, Lx should be an odd number')

    # Tolerance default
    tol_eta = 0.0000001
    tol_Q = 0.0000001
    lr_th = 0.001
    tol_conv = 0.000001
    minimum_steps = 50

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

    method = sys.argv[1]

    # Load experimental data
    if mpi_rank == 0:
        if symmetry == 0:
            data=np.loadtxt('data/exp_plume_threshold{}.dat'.format(dth))
        if symmetry == 1:
            data=np.loadtxt('data/exp_plume_symmetric_threshold{}.dat'.format(dth))
        if coarse == 1:
            data=np.loadtxt('data/exp_plume_symmetric_coarse_threshold{}.dat'.format(dth))
    else:
        data=None

    if mpi_size > 1:
        # broadcast data to all processes
        data = mpi_comm.bcast(data, root=0)

# ----------------------------------------------------------------------------
    #INITIALIZATIONS
# ----------------------------------------------------------------------------

    if mpi_rank == 0:
        # Create action handler
        act_hdl = utils.AgentActions(A)
        for i,move,name in zip(range(A),act_move,act_name):
            act_hdl.set_action(i,move,name)
        
        # Create array where the source is located
        yxs = it.product(np.arange(Ly), np.arange(Lx))
        yx_founds = it.filterfalse(lambda x: (x[0]-Ly0)**2 + (x[1]-Lx0)**2 > find_range**2, yxs)
        source_as_zero = np.array([i for i in yx_founds])

        # Create the observation and rewards
        eta = np.zeros(L*M)
        PObs_lim, RR, PObs, RR_np = utils.create_PObs_RR(Lx, Ly, Lx0, Ly0, find_range, cost_move, reward_find, M, beta, max_obs, O, A, V, data=data,source_as_zero=source_as_zero,exp_plume=False)
        PObs_lim = np.abs(PObs_lim)

        # Create the initial positions of the agent
        # Initial density is only at y=0, proportional to signal probability
        rho0 = np.zeros(M*Lx*Ly)
        #rho0[:Lx*Ly0//2] = (1-PObs_lim[0,:Lx*Ly0//2])/np.sum((1-PObs_lim[0,:Lx*Ly0//2]))
        rho0[:Lx] = (1-PObs_lim[0,:Lx])/np.sum((1-PObs_lim[0,:Lx]))
        
        if ("ev" in params.keys()):
            rho0[:Lx*Ly0//2] = (1-PObs_lim[0,:Lx*Ly0//2])/np.sum((1-PObs_lim[0,:Lx*Ly0//2]))
        else:
            rho0[:Lx] = (1-PObs_lim[0,:Lx])/np.sum((1-PObs_lim[0,:Lx]))
        
        # Create folder for saving results
        name_folder = 'ExpPlume_A{}M{}b{}g{}FR{}LR{}sym{}th{}rep{}LX{}LY{}co{}_mth-{}_slvr-{}_fac-{}'.format(A,M,beta,gamma,find_range,lr_th,symmetry,dth,replica,Lx,Ly,coarse,method,lib_solve_linear_system,factor_size)
        os.makedirs(name_folder, exist_ok=True)
        #os.system("cp {} {}".format('./'+sys.argv[1], './'+name_folder))
        f = open(name_folder+"/values.dat", "a")
        
        
        # Policy Initialization with bias
        new_policy = params["new_policy"]

        unbias=0
        if ("unbias" in params.keys()):
            unbias = params["unbias"] 

        if (new_policy==0):
            folder_restart = params["folder_restart"]
            th = np.loadtxt(folder_restart + '/file_theta.out')
            Q = np.loadtxt(folder_restart + '/file_Q.out')
            eta = np.loadtxt(folder_restart + '/file_eta.out')

            th = th.reshape(O, M, a_size)
        elif (new_policy==1):
            th = (np.random.rand(O, M, a_size)-0.5)*0.5 
            th[1:,:,2::A] += 0.5
            if (unbias==1):
                th[1:,:,2::A] -= 0.5
            eta = np.ones(eta.shape)/(L*M)/(1-gamma)
            Q = utils.create_random_Q0(Lx, Ly, Lx0, Ly0, gamma, a_size, M, cost_move, reward_find) 
        
        V = np.zeros(L*M)


        # Create the initial policy
        pi = softmax(th, axis=2)

        value = 0
        oldvalue = value

    if mpi_rank != 0:
        act_hdl = None
        PObs_lim = None
        RR = None
        RR_np = None
        source_as_zero = None
        rho0 = None
        th = None
        Q = None
        eta = None
        V = None
        pi = None

        # if mpi_size > 1:
        #     # broadcast data to all processes
        #     th = mpi_comm.bcast(th, root=0)
        #     Q = mpi_comm.bcast(Q, root=0)
        #     eta = mpi_comm.bcast(eta, root=0)
            

    # utils.PETSc.COMM_WORLD.barrier()

# ----------------------------------------------------------------------------
    #Print parameters
# ----------------------------------------------------------------------------

    # Print parameters
    if mpi_rank == 0:
        print("-"*50)
        print("System parameters:")
        print("Lx: ",Lx,"   Ly: ",Ly,"   M: ",M,"   O: ",O,"   Actions: ",act_hdl.A)
        print("gamma: ",gamma,"   lr_th: ",lr_th, "   find_range: ",find_range)
        print('method: {}'.format(method))
        print("-"*50)
        print("Optimization parameters:")
        print("Ntot:",Ntot,"   Nprint:",Nprint,"   tol_conv:",tol_conv, "   minimum_step:",minimum_steps)
        print("-"*50)
    
# ----------------------------------------------------------------------------
    #OPTIMIZATION
# ----------------------------------------------------------------------------

    time_opt = time.time()
    verbose_eta = True
    time_step = np.zeros(Nprint)
    for t in range(Ntot):

        # Time counter
        time_start = time.time()

        if method == 'direct':
            # Direct linear system solution
            eta, T = utils.linear_solve_eta(pi, PObs_lim, gamma, rho0, eta, Lx, Ly, Lx0, Ly0, find_range, act_hdl,source_as_zero,solver=solver,verbose=verbose_eta)
            time_eta = time.time()
            V, Q = utils.linear_solve_Q(T, Lx, Ly, M, A, gamma, find_range, act_hdl, source_as_zero,RR_np,V,solver=solver)
            time_Q = time.time()

        if method == 'iter':
            # Iterative solutions of linear system
            eta = utils.iterative_solve_eta(pi, PObs_lim, gamma, rho0, eta, tol_eta, Lx, Ly, Lx0, Ly0, find_range)
            time_eta = time.time()
            Q = utils.iterative_solve_Q(pi, PObs_lim, gamma, RR, Q, tol_Q, Lx, Ly, Lx0, Ly0, find_range, cost_move)
            time_Q = time.time()

        if mpi_rank == 0:
            # Gradient calculation
            grad = utils.find_grad(pi, Q, eta, L, PObs_lim)
            grad -= np.max(grad, axis=2, keepdims=True)

            # Reset value for new iteration afterwards
            
            th += grad * lr_th / np.max(np.abs(grad)) # (t / Ntot + 0.5) #rescaled gradient
            th -= np.max(th, axis=2, keepdims=True)
            th = np.clip(th, -20, 0)

            time_step[t%Nprint] = time.time() - time_start
        
        # Print and check convergence
        if (t % Nprint == 0) and (mpi_rank == 0):
            value =  utils.get_value(Q, pi, PObs_lim, L, rho0)
            ratio_change = abs((value-oldvalue)/value)
            print('step:{:5d} |  current value: {:.7f} | ratio value : {:.7f}'.format(t, value, ratio_change), end=' | ')
            # Print times
            print('time eta:{:10.3f}'.format(time_eta-time_start), end=' | ')
            print('time Q:{:10.3f}'.format(time_Q-time_eta), end=' || ')
            print('avg time:{:10.3f}'.format(np.mean(time_step)), end=' | ')
            print('std dev:{:10.3f}'.format(np.std(time_step)), end=' | ')
            print('time total:{:10.3f}'.format(np.sum(time_step)),flush=True)
            
            # Print values to file
            f.write('current value: {} @ time:{} \n'.format(value, t))
            f.flush()
            # check convergence
            if (ratio_change < tol_conv) and (minimum_steps < t ):
                f.write('converged at T={}'.format(t))
                break
            oldvalue = value
            verbose_eta = False

        #if (t % (Nprint*10) == 0):
        #    np.savetxt(name_folder + '/file_theta{}.out'.format(t), th.reshape(-1))
        #    np.savetxt(name_folder + '/file_Q{}.out'.format(t), Q.reshape(-1))
        #    np.savetxt(name_folder + '/file_eta{}.out'.format(t), eta.reshape(-1))
        if (t % (Nprint*10) == 0) and (mpi_rank == 0):
            np.savetxt(name_folder + '/file_theta.out', th.reshape(-1))
            np.savetxt(name_folder + '/file_Q.out', Q.reshape(-1))
            np.savetxt(name_folder + '/file_eta.out', eta.reshape(-1))

        if mpi_rank == 0:
            # Policy update
            pi = softmax(th, axis=2)
    # End for loop optimization
    # ----------------------------------------------------------------------------

    # final print
    if mpi_rank == 0:
        np.savetxt(name_folder + '/file_theta.out', th.reshape(-1))
        np.savetxt(name_folder + '/file_Q.out', Q.reshape(-1))
        np.savetxt(name_folder + '/file_eta.out', eta.reshape(-1))
        f.close()

        if t<Ntot:
            print('converged at T= {} with value= {}'.format(t,value), end=' ')
            print('time total:{:10.3f}'.format(time.time()-time_opt))
            print('theta: \n', th)
        else:
            print('did not converged for {} runs and tolerance {}'.format(Ntot,))
            print('current value: {} @ time:{} \n'.format(value, t))
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
## Main function
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    optimize()