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
# Usage: python optimize.py -h
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
from scipy import interpolate

sys.path.append('../Comm/')
import utils as utils


import argparse
parser = argparse.ArgumentParser(description='Finite State Controller for plume tracking')
parser.add_argument('--input_file', type=str,
                    required=True,
                    help='Input file with parameters')

parser.add_argument('--method', type=str, 
                    choices=['direct', 'iter','experimental'],
                    default='direct',
                    help='Method to solve the linear system')
parser.add_argument('--lib_solver', type=str,
                    choices=['scipy', 'petsc', 'cupy', 'numpy_inv', 'numpy', 'torch_inv', 'torch'],
                    default='scipy',
                    help='Library to solve the linear system')
parser.add_argument('--device', type=str,
                    choices=['cpu', 'gpu'],
                    default='cpu',
                    help='Device to use for computation')

parser.add_argument('--output_dir', type=str,
                    default='',
                    help='Input file with parameters')


# args = parser.parse_args()
args, unknown_args = parser.parse_known_args()
input_file = args.input_file
lib_solve_linear_system = args.lib_solver
method = args.method
device = args.device

# ----------------------------------------------------------------------------
# Select the library to solve the linear system 
# lib_solve_linear_system = 'scipy'
# lib_solve_linear_system = 'petsc'
# lib_solve_linear_system = 'cupy'

solver = utils.solver_opt()
solver.set_lib_solver(lib_solve_linear_system)

mpi_rank = solver.mpi_rank
mpi_size = solver.mpi_size
mpi_comm = solver.mpi_comm

solver.device = device

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000)

# Fix the seed for reproducibility
np.random.seed(33+33)

# ----------------------------------------------------------------------------
# Read input file and set parameters
def set_parameters_from_file(input_file):
    """
    Read input file and return a dictionary with the parameters
    """
    if mpi_rank == 0 :
        params = json.load(open(input_file))
        # params = json.load(open('input.dat'))
    else:
        params = None

    if mpi_size > 1:
        # broadcast parameters to all processes
        params = mpi_comm.bcast(params, root=0)

    fsc_param = utils.parameters()
    fsc_param.set_parameters(params)

    # Assert that the parameters are correct
    assert not ((fsc_param.plume.coarse == 1) and (fsc_param.env.Lx % 2) == 0) , 'Error in value for Lx. In COARSE set-up, Lx should be an odd number'

    return fsc_param

# ----------------------------------------------------------------------------
# Optimization function
# @profile
def optimize(fsc):
        
    # ----------------------------------------------------------------------------
    if mpi_rank == 0:

        # Print parameters
        utils.print_parameters(fsc,method,lib_solve_linear_system,device)

        # Create folder for saving results
        name_folder = utils.create_output_folder_name(fsc, method, lib_solve_linear_system)
        if args.output_dir != '':
            name_folder = args.output_dir + '_' + name_folder
        os.makedirs(name_folder, exist_ok=True)
        
        # Create file to save values
        f = open(name_folder+"/values.dat", "a")
    
        # Save parameters to file
        param = {}
        param.update(fsc.agent.__dict__)
        param.update(fsc.env.__dict__)
        param.update(fsc.plume.__dict__)
        param.update(fsc.optim.__dict__)
        param.pop('act_hdl')
        param.pop('factor_dim')
        with open(name_folder+'/input.dat', 'w') as fp:
            json.dump(param, fp)

        print(param)

    # ----------------------------------------------------------------------------
        #INITIALIZATIONS
    # ----------------------------------------------------------------------------
    # Load experimental data
    if mpi_rank == 0:
        if fsc.plume.symmetry == 0:
            data=np.loadtxt('data/exp_plume_threshold{}.dat'.format(fsc.plume.dth))
        if fsc.plume.symmetry == 1:
            data=np.loadtxt('data/exp_plume_symmetric_threshold{}.dat'.format(fsc.plume.dth))
        if fsc.plume.coarse == 1:
            data=np.loadtxt('data/exp_plume_symmetric_coarse_threshold{}.dat'.format(fsc.plume.dth))
    else:
        data=None

    if mpi_size > 1:
        # broadcast data to all processes
        data = mpi_comm.bcast(data, root=0)


    if mpi_rank == 0:

        # Create the theta parameters to optimize
        th = (np.random.rand(fsc.agent.O, fsc.agent.M, fsc.agent.AxM)-0.5)*0.5 
        th[1:,:,2::fsc.agent.A] += 0.5
        if fsc.optim.unbias:
            th[1:,:,2::fsc.agent.A] -= 0.5

        # Create eta array
        eta = np.ones(fsc.env.L * fsc.agent.M)
        eta *= (1-fsc.agent.gamma)/(fsc.env.L * fsc.agent.M)

        # Create Q and V array
        Q = utils.create_random_Q0(fsc.agent, fsc.env) 
        V = np.ones(fsc.env.L * fsc.agent.M)
        
        # Policy Initialization from previous run
        if not fsc.optim.new_policy :
            print('Loading previous policy from:')
            print(fsc.optim.folder_restart)
            # Load previous policy from file
            folder_restart = fsc.optim.folder_restart
            # Loadthe theta parameters to optimize
            th = np.loadtxt(folder_restart + '/file_theta.out')
            th = th.reshape(fsc.agent.O, fsc.agent.M, fsc.agent.A * fsc.agent.M)
            # Load eta and Q
            Q_load = np.loadtxt(folder_restart + '/file_Q.out')
            eta_load = np.loadtxt(folder_restart + '/file_eta.out')

            # Check if the size of the loaded eta and Q are correct
            if eta_load.shape[0] == eta.shape[0]:
                eta = eta_load
            else:
                print('Error in the size of eta, using standard initialization')
            
            if Q_load.shape[0] == Q.shape[0]:
                Q = Q_load
                
                # Extract V from Q
                V = V.reshape(fsc.agent.M, fsc.env.Ly, fsc.env.Lx)
                Q = Q.reshape(fsc.agent.M,fsc.env.Ly, fsc.env.Lx, fsc.agent.M, fsc.agent.A)
                for m in range(fsc.agent.M):
                    V[m,:,:] = (Q[0,:,:,m,0] + fsc.agent.cost_move) / fsc.agent.gamma
                V = V.flatten()
                Q = Q.flatten()

            else:
                print('Error in the size of Q and V, using standard initialization')

        # Create the initial policy
        pi = softmax(th, axis=2)

        # Create array where the source is located
        yxs = it.product(np.arange(fsc.env.Ly), np.arange(fsc.env.Lx))
        find_range2 = fsc.env.find_range**2
        yx_founds = it.filterfalse(lambda x: 
                                   (x[0]-fsc.env.Ly0)**2 + (x[1]-fsc.env.Lx0)**2 > find_range2, 
                                   yxs)
        source_as_zero = np.array([i for i in yx_founds])

        # Create the observation and rewards
        PObs_lim, RR, PObs, RR_np = utils.create_PObs_RR(fsc.agent, fsc.env, fsc.plume, data=data, source_as_zero=source_as_zero)
        PObs_lim = np.abs(PObs_lim)
        
        # Create the initial positions of the agent
        # Initial density is only at y=0, proportional to signal probability
        rho0 = np.zeros(fsc.env.L * fsc.agent.M)
        rho0[:fsc.env.Lx] = (1-PObs_lim[0,:fsc.env.Lx])/np.sum((1-PObs_lim[0,:fsc.env.Lx]))
        # if ("ev" in params.keys()):
        #     rho0[:Lx*Ly0//2] = (1-PObs_lim[0,:fsc.env.Lx*fsc.env.Ly0//2])/np.sum((1-PObs_lim[0,:fsc.env.Lx*fsc.env.Ly0//2]))
        # else:
        #     rho0[:fsc.env.Lx] = (1-PObs_lim[0,:fsc.env.Lx])/np.sum((1-PObs_lim[0,:fsc.env.Lx]))
        
        # Init the initial value
        value = 0
        oldvalue = value
        ratio_change = 1.0
    # ----------------------------------------------------------------------------
        # END INITIALIZATIONS
    # ----------------------------------------------------------------------------
        # Print the max value to achieve
        utils.get_max_value(fsc.env.Lx, fsc.env.Ly, fsc.env.Lx0, fsc.env.Ly0, fsc.env.find_range, rho0, fsc.agent.gamma)


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

# ----------------------------------------------------------------------------
    #OPTIMIZATION
# ----------------------------------------------------------------------------

    time_opt = time.time()
    verbose_eta = True
    time_step = np.zeros(fsc.optim.Nprint)

    if fsc.agent.lr_th == 'auto':
        # Learning rate schedule
        lr_val =  [0.1, 0.01, 0.001 , 0.0001]
        lr_time = [0.0, 0.1 , 0.8   , 1.0]
        
        lr_val = [0.1, 0.01, 0.0001,0.0001]
        lr_time = [0.0, 0.01, 0.8, 1.0]

        lr_val = fsc.agent.lr_val
        lr_time = fsc.agent.lr_time_frac

        lr_val = interpolate.interp1d(lr_time, lr_val, kind='linear',fill_value="extrapolate")
        lr_time = np.linspace(0.0, 1.0, num=fsc.optim.Ntot)
        lr_values = lr_val(lr_time)

        # def lr_descent(start,last,step):
        #     nsteps = step[-1] - step[0]
        #     step = step - step[0]
        #     alpha = (start - last ) / (nsteps * last) 
        #     return 1.0 / (1.0 + alpha * step * 1) * start 

        # lr_values = np.array([])
        # for i in range(1,len(lr_rate)):
        #     lr_values = np.concatenate((lr_values,lr_descent(lr_rate[i-1],lr_rate[i],np.arange(steps[i-1],steps[i]))))
    else:
        lr_values = np.ones(fsc.optim.Ntot)*fsc.agent.lr_th


    init_direct = fsc.optim.init_direct
    save_tol_eta = fsc.optim.tol_eta
    save_tol_Q = fsc.optim.tol_Q

    if init_direct != 0 and method == 'direct':
        fsc.optim.tol_eta = -1
        fsc.optim.tol_Q = -1

    # set steps for choose iter or direct
    method_select = method

    convergence = 0
    for t in range(fsc.optim.Ntot):

        # Time counter
        time_start = time.time()

        if init_direct == t:
            fsc.optim.tol_eta = save_tol_eta
            fsc.optim.tol_Q = save_tol_Q

        if method_select == 'direct':
            # Direct linear system solution
            eta, T = utils.linear_solve_eta(fsc.agent, fsc.env, fsc.optim, eta, rho0, pi, PObs_lim, source_as_zero,solver=solver,verbose=verbose_eta)
            time_eta = time.time()
            V, Q = utils.linear_solve_Q(fsc.agent, fsc.env, fsc.optim, T, V, RR_np, source_as_zero, solver=solver)
            time_Q = time.time()

        elif method_select == 'iter':
            print('method: {}'.format(method_select))
            # Iterative solutions of linear system
            eta = utils.iterative_solve_eta(fsc.agent, fsc.env, fsc.optim, pi, PObs_lim, rho0, eta)

            time_eta = time.time()
            Q = utils.iterative_solve_Q(fsc.agent, fsc.env, fsc.optim, pi, PObs_lim, RR, Q)
            time_Q = time.time()

        # Check convergence 
        if mpi_rank == 0:
            value =  utils.get_value(Q, pi, PObs_lim, fsc.env.L, rho0)
            ratio_change = abs((value-oldvalue)/value)

            oldvalue = value
            verbose_eta = False

            if (ratio_change < fsc.optim.tol_conv) and ( fsc.optim.minimum_iter < t ):
                convergence = 1
        
        if mpi_size > 1:
            convergence = mpi_comm.bcast(convergence, root=0)


        if mpi_rank == 0:
            # Gradient calculation
            grad = utils.find_grad(pi, Q, eta, fsc.env.L, PObs_lim)
            grad -= np.max(grad, axis=2, keepdims=True)

            # Reset value for new iteration afterwards
            lr_th = lr_values[t]
            
            th += grad * lr_th / np.max(np.abs(grad)) # (t / Ntot + 0.5) #rescaled gradient
            th -= np.max(th, axis=2, keepdims=True)
            th = np.clip(th, -20, 0)

        
        # Print and check convergence
        if (t % fsc.optim.Nprint == 0) and (mpi_rank == 0):

            print('lr_th: {:.5f}'.format(lr_th))
            print('tol_eta: {:.5f} | tol_Q: {:.5f}'.format(fsc.optim.tol_eta, fsc.optim.tol_Q))

            time_step[ t % fsc.optim.Nprint] = time.time() - time_start
            print('step:{:5d} |  current value: {:.7f} | ratio value : {:.7f}'.format(t, value, ratio_change), end=' | ')
            # Print times
            print('time eta:{:10.3f}'.format(time_eta-time_start), end=' | ')
            print('time Q:{:10.3f}'.format(time_Q-time_eta), end=' || ')
            print('avg time:{:10.3f}'.format(np.mean(time_step)), end=' | ')
            print('std dev:{:10.3f}'.format(np.std(time_step)), end=' | ')
            print('time total:{:10.3f}'.format(np.sum(time_step)),flush=True)

            print('-'*77)
            
            # Print values to file
            f.write('current value: {} @ time:{} \n'.format(value, t))
            f.flush()


        #if (t % (Nprint*10) == 0):
        #    np.savetxt(name_folder + '/file_theta{}.out'.format(t), th.reshape(-1))
        #    np.savetxt(name_folder + '/file_Q{}.out'.format(t), Q.reshape(-1))
        #    np.savetxt(name_folder + '/file_eta{}.out'.format(t), eta.reshape(-1))
        if (t % (fsc.optim.Nprint * 10 ) == 0) and (mpi_rank == 0):
            np.savetxt(name_folder + '/file_theta.out', th.reshape(-1))
            np.savetxt(name_folder + '/file_Q.out', Q.reshape(-1))
            np.savetxt(name_folder + '/file_eta.out', eta.reshape(-1))

        if convergence == 1:
            break

        if mpi_rank == 0:
            # Policy update
            pi = softmax(th, axis=2)
    # End for loop optimization
    # ----------------------------------------------------------------------------
    if solver.use_petsc:
        solver.free_petsc(solver)
    # final print
    if mpi_rank == 0:
        np.savetxt(name_folder + '/file_theta.out', th.reshape(-1))
        np.savetxt(name_folder + '/file_Q.out', Q.reshape(-1))
        np.savetxt(name_folder + '/file_V.out', V.reshape(-1))
        np.savetxt(name_folder + '/file_eta.out', eta.reshape(-1))
        f.close()

        if t < fsc.optim.Ntot:
            print('converged at T= {} with value= {}'.format(t,value), end=' ')
            print('time total:{:10.3f}'.format(time.time()-time_opt))
            print('theta: \n', th)
        else:
            print('did not converged for {} runs and tolerance {}'.format(fsc.optim.Ntot, fsc.optim.tol_conv))
            print('current value: {} @ time:{} \n'.format(value, t))
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
## Main function
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    fsc_param = set_parameters_from_file(input_file)

    optimize(fsc_param)