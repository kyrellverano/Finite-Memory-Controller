import numpy as np
from scipy.special import softmax as softmax
# from scipy import linalg
import matplotlib.pyplot as plt
import fast_sparce_multiplications_2D as fast_mult

import time 
# Linear solvers
from local_linear_solvers import *

# ----------------------------------------------------------------------------
class solver_opt:

    def __init__(self):
        """ 
        This class contains the solver options.
        """

        # Set the default solver
        self.function_solver_direct = None
        self.function_solver_iter = None

        # Set the algorithm to use
        self.solver_type_eta = 'direct'
        self.solver_type_V = 'direct'

        # Set the MPI options
        self.mpi_rank = 0
        self.mpi_size = 1
        self.mpi_comm = None

        # Set the device to use
        self.use_petsc = False
        self.device = None

        self.Tmatrix_index = None
        self.Tmatrix_p_index = None
        self.Tsm_sm_sp = None

    def set_lib_solver(self,solver):
        """
        This function set the solver to use for the linear system.
        """
        # Set the default solver
        if solver == None:
            solver = 'scipy'

        if solver == 'scipy' :
            load_info = load_scipy()
            self.function_solver_direct = load_info[0]
            self.function_solver_iter = load_info[1]

            self.device = 'cpu'

            if self.mpi_rank == 0:
                print('Using scipy solver')

        elif solver == 'petsc':
            load_info = load_petsc()
            from local_linear_solvers import PETSc 

            self.use_petsc = True
            self.ksp_type = 'preonly'          
            self.pc_type = 'lu'
            # self.solver_type_eta = 'iter'
            # self.solver_type_V = 'iter'

            self.function_solver_direct = load_info[0]
            self.function_solver_iter = load_info[1]
            self.mpi_rank = load_info[2]
            self.mpi_size = load_info[3]
            self.mpi_comm = load_info[4]

            self.device = 'cpu'
            OptDB = PETSc.Options()
            # PETSc.Options().setValue('cuda_device', '2')
            if self.mpi_rank == 0:
                print('Using petsc solver')

            # Keep the allocation
            self.Tsm_sm_sp_petsc = None
            self.var_list = None
            self.A = None
            self.b = None
            self.x = None
            self.ones = None
            self.ksp = None
            self.first_allocation = True
            

        elif solver == 'cupy':
            load_info = load_cupy()

            self.function_solver_direct = load_info[0]
            self.function_solver_iter = load_info[1]

            self.device = 'gpu'

            if self.mpi_rank == 0:
                print('Using cupy solver')
        
        elif solver == 'numpy_inv':
            load_info = load_numpy_inv()

            self.function_solver_direct = load_info[0]
            self.function_solver_iter = load_info[1]

            self.device = 'cpu'

            if self.mpi_rank == 0:
                print('Using numpy inv solver')

        elif solver == 'numpy':
            load_info = load_numpy()

            self.function_solver_direct = load_info[0]
            self.function_solver_iter = load_info[1]

            self.device = 'cpu'

            if self.mpi_rank == 0:
                print('Using numpy solver')

        elif solver == 'torch_inv':
            load_info = load_torch_inv()

            self.function_solver_direct = load_info[0]
            self.function_solver_iter = load_info[1]

            self.device = 'cpu'

            if self.mpi_rank == 0:
                print('Using PyTorch inv solver')

        elif solver == 'torch':
            load_info = load_torch()

            self.function_solver_direct = load_info[0]
            self.function_solver_iter = load_info[1]

            self.device = 'cpu'

            if self.mpi_rank == 0:
                print('Using Pytorch solver')

        else :
            print('Error: solver not recognized', solver)
            assert False, 'Solver not recognized'
            return


# ----------------------------------------------------------------------------

def create_cplume(Lx, Ly, Lx0, Ly0, D, V, tau, aR): 
    """
    Returns a diffusion plume with given parameters.
    """
    spacex = np.arange(1,Lx+1)-(Lx+1)/2.
    spacey = np.arange(Ly)-(Ly0-1)
    xx, yy = np.meshgrid(spacex, spacey)
    rr = np.sqrt(xx**2 + yy**2)
    lam = np.sqrt(D*tau/(1+V*V*tau/D/4))
    cplume = aR/(rr+0.01)*np.exp(-rr/lam-yy*V/2/D)
    return cplume

def create_random_Q0(Lx, Ly, Lx0, Ly0, gamma, a_size, M, cost_move, reward_find):
    """
    Returns an approx Q for a random diffusion.
    """
    spacex = np.arange(1,Lx+1)-(Lx+1)/2.
    spacey = np.arange(Ly)-(Ly0-1)

    xx, yy = np.meshgrid(spacex, spacey)
    rr = xx**2 + yy**2
    random_Q0 = (-1/(1-gamma)*cost_move) / (1 + 2/np.abs(rr)) + (1 - 1/ (1 + 2/np.abs(rr)))*reward_find    
    random_Q0 = np.tile(np.repeat(random_Q0, a_size), M).reshape(-1)
    return random_Q0

def create_plume_from_exp(Lx, Ly, Lx0, Ly0, cmax, data):
    """
    Returns a diffusion plume from a given data.
    """
    Lx0 = int(Lx0)
    Ly0 = int(Ly0)
    
    exp_plume_mat = data.copy()
    # center_x,center_y = np.argmax(exp_plume_mat[:,0])
    center_x,center_y = np.unravel_index(np.argmax(exp_plume_mat, axis=None),exp_plume_mat.shape)

    # symmetrize exp_plume_mat
    min_size = min(center_x, exp_plume_mat.shape[0]-center_x)
    exp_plume_mat = exp_plume_mat[center_x - min_size:center_x + min_size,:].copy()

    # padding into shape
    new_size_x = Lx
    new_size_y = Ly
    
    exp_Lx, exp_Ly = exp_plume_mat.shape 
    print('exp_plume_mat.shape',exp_plume_mat.shape)
    
    assert Ly0+10 <= exp_Ly, 'Size of experimental plume too small' 
    
    new_plume = np.zeros((new_size_x, new_size_y))
    
    if Lx0 < min(exp_Lx-center_x, center_x):
        momo = exp_plume_mat[center_x-(Lx0):center_x+(new_size_x-Lx0),
                     :Ly0+10].copy()
        
    else:
        momo = np.zeros((new_size_x, Ly0+10))
        momo[Lx0-center_x:Lx0-center_x+exp_Lx,:] = exp_plume_mat[:,:Ly0+10].copy()
    
    new_plume[:,-(Ly0+10):] = momo.copy()
    
    #new_plume[new_plume < 0.1] = 0.
    #renorm = np.sum(new_plume[:, -1])/cmax
    #new_plume = np.swapaxes(new_plume/renorm,0,1)
    new_plume = np.swapaxes(new_plume,0,1)
    
    return new_plume[::-1,:]

def average_reward(Tsm_sm_matrix, M, Lx, Ly, cost_move,source_as_zero):
    """
    This function compute the average reward from the Transition matrix T
    The dimension of RR is (M, Ly, Lx)
    """
    # Tsm_sm = Tsm_sm_matrix.reshape(M, Ly, Lx, M, Ly, Lx)
    # RR = -cost_move * Tsm_sm.sum(axis=(0,1,2))

    RR = np.full((M, Ly, Lx),-cost_move)
    RR[:,source_as_zero[:,0],source_as_zero[:,1]] = 0.0

    return RR

def create_PObs_RR(Lx, Ly, Lx0, Ly0, find_range, cost_move, reward_find, M, cmax, max_obs, diff_obs, A, V, data, D=50, tau=2000, plume_stat="Poisson", exp_plume=True,source_as_zero=None):
    spacex = np.arange(1,Lx+1)-(Lx+1)/2.
    spacey = np.arange(Ly)-(Ly0-1)

    # CHECK IF ORDER IS MANTAINED
    #cplume = cmax * np.multiply.outer((np.tanh(-spacey/5)+1)*0.5*np.exp(-np.abs(spacey)**2/450), np.exp(-np.abs(spacex)**2/30))
    xx, yy = np.meshgrid(spacex, spacey)
    
    if exp_plume:
        cplume = create_plume_from_exp(Lx, Ly, Lx0, Ly0, cmax, data)
    else:
        cplume = create_cplume(Lx, Ly, Lx0, Ly0, D, V, tau, cmax)
    # ------------------  
    cplume = cplume[:,:].reshape(-1)  
    PObs = np.zeros((max_obs, Lx * Ly))

    if (plume_stat == "Poisson"):
        fact = 1
        for i in range(max_obs-1):
            PObs[i] = cplume**i * np.exp(-cplume) / fact 
            fact = fact * (i+1)
    elif (max_obs == 2 and plume_stat == "Bernoulli"):
        PObs[0] = 1 - np.clip(cplume, 0, 1)
    else:
        print('Error in the statistics of plume!')
        print('hello')
        
    PObs[-1,:] = 1 - np.sum(PObs[:-1,:], axis=0)
    PObs_2 = np.tile(PObs, (1,M))

    PObs_lim = np.zeros((diff_obs, Lx*Ly*M))
    PObs_lim[:] = PObs_2[:diff_obs]
    PObs_lim[-1] += np.sum(PObs_2[diff_obs:], axis=0)

    # if A == 4:
    #   RR = fast_mult.rewards_four_2d_walls(PObs, M, Lx, Ly, Lx0+1, Ly0+1, find_range, cost_move, reward_find, max_obs)
    # elif A == 5:
    #   RR = fast_mult.rewards_five_2d_walls(PObs, M, Lx, Ly, Lx0+1, Ly0+1, find_range, cost_move, reward_find, max_obs)
    if A == 4:
        RR = fast_mult.rewards_four_2d(M, Lx, Ly, Lx0+1, Ly0+1, find_range, cost_move, reward_find, max_obs)
    elif A == 5:
        RR = fast_mult.rewards_five_2d(M, Lx, Ly, Lx0+1, Ly0+1, find_range, cost_move, reward_find, max_obs)

    RR_np = average_reward(None, M, Lx, Ly, cost_move, source_as_zero)

    return np.abs(PObs_lim), RR, np.abs(PObs), RR_np

def iterative_solve_eta(pi, PObs_lim, gamma, rho0, eta0, tol, Lx, Ly, Lx0, Ly0, find_range):
    max_it = 10000
    eta = eta0.copy()
    new_eta = np.zeros(eta.shape)
    O, M, A = pi.shape
    L = eta.shape[0] // M
    tol2 = 0.
    if (gamma < 1.):
        tol2 = tol/(1-gamma)*tol/(1-gamma)
    else:
        tol2 = tol*tol*1000000
    for i in range(max_it):

        #new_eta = rho0 + gamma * np.matmul(TS_S, eta)
        if A//M == 4:
            new_eta = rho0 + gamma * fast_mult.mult_eta_four_2d(pi, eta, PObs_lim, Lx0+1, Ly0+1, find_range, Lx, Ly, O, M)
        elif A//M == 5:
            new_eta = rho0 + gamma * fast_mult.mult_eta_five_2d(pi, eta, PObs_lim, Lx0+1, Ly0+1, find_range, Lx, Ly, O, M)
        # if (i%1 == 0):
        delta = np.max((new_eta-eta)*(new_eta-eta))
        if (delta < tol2): 
            return new_eta
        #if ((i+1)%1000 == 0):
        #    print('eta: i {}, delta {}'.format(i, delta))
        eta = new_eta.copy()
    print('NOT CONVERGED - eta')
    return new_eta

# @profile
def iterative_solve_Q(pi, PObs_lim, gamma, RR, Q0, tol, Lx, Ly, Lx0, Ly0, find_range, cost_move):
    max_it = 10000
    Q = Q0
    O, M, A = pi.shape
    L = Q.shape[0] // (M*A)
    tol2 = 0.   
    meanQ = np.mean(Q)
    if (gamma < 1.):
        tol2 = tol/(1-gamma)*tol/(1-gamma)
    else:
        tol2 = tol*tol*1000000
    for i in range(max_it):
        if A//M == 4:
            new_Q = RR + gamma * fast_mult.mult_q_four_2d(pi, Q, PObs_lim, Lx0+1, Ly0+1, find_range, Lx, Ly, O, M)
        elif A//M == 5:
            new_Q = RR + gamma * fast_mult.mult_q_five_2d(pi, Q, PObs_lim, Lx0+1, Ly0+1, find_range, Lx, Ly, O, M)
        if (i%1 == 0):
            delta = (new_Q-Q)
            delta = np.max(delta*delta)
            if ( delta < tol2): 
                return new_Q
        #if ((i+1)%1000 == 0):
        #    print('Q: i {}, delta {}, tol^2 {}'.format(i, delta, tol2))
        Q = new_Q.copy()
    print('NOT CONVERGED - Q')
    return new_Q

# Define the action space
class AgentActions():
    def __init__(self,actions_number):
        space_dim = 2
        self.A = actions_number
        self.actions = np.zeros((actions_number,space_dim),dtype=int)
        self.actions_names = {}
    
    def set_action(self,action_index : int, action : list, action_name : str ):
        self.actions[action_index] = np.array(action)
        self.actions_names[action_index] = action_name

    def action_move(self,action_index : int):
        if action_index >= self.A:
            print('Error: action not recognized')
            return np.array([0, 0])
        return self.actions[action_index]
    
    def action_name(self,action_index : int):
        if action_index >= self.A:
            print('Error: action not recognized')
            return 'Error'
        return self.actions_names[action_index]

# @profile
def linear_solve_eta(pi, PObs_lim, gamma, rho0, eta, Lx, Ly, Lx0, Ly0, find_range, act_hdl, source_as_zero, verbose=False, solver = None):
    """
    This function should solve the following:
    --> New_eta = (1 - gamma T)^-1 rho
    """

    timing = True
    if timing:
        timer_step = np.zeros(3)
        timer_step[0] = time.time()


    comm = solver.mpi_comm
    mpi_rank = solver.mpi_rank
    mpi_size = solver.mpi_size

    if mpi_rank == 0:
        # O = Observations, M = Memory, A = Actions
        O, M, A = pi.shape
        # L = Dimension of the environment
        L = Lx * Ly
        new_eta = np.zeros(M*L)
        
        # PY has size ~ 10^5
        PY = PObs_lim.reshape(O, M, Ly, Lx)
        # PY has size ~ 10^2
        PAMU = pi.reshape(O, M, M, A//M)
        # PAMU = softmax( np.zeros((O, M, M, A//M)), 2)

        
        p_a_mu_m_xy = np.einsum( 'omyx, omna -> anmyx', PY, PAMU)
        # T [ s'm'  sm] = sum_a, mu p(s'm' | sm a mu) p(a mu | sm)
        #               = sum_a, mu p(s'm' | sm a mu) sum_y f(y | s) pi(a mu | y m)

        # Tsm_sm_matrix = build_Tsm_sm_sparse_2(M,Lx,Ly,Lx0,Ly0,find_range,p_a_mu_m_xy,act_hdl,source_as_zero)
        Tsm_sm_matrix = build_Tsm_sm_sparse_3(M,Lx,Ly,Lx0,Ly0,find_range,p_a_mu_m_xy,act_hdl,source_as_zero,solver)
        action_size = act_hdl.A

    else:
        Tsm_sm_matrix = None
        action_size = None
        M = None

    # Tsm_sm has size ~ 10^5 x 10^5 or more
    # if mpi_rank == 0:
    #     Tsm_sm_matrix = build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,A//M,p_a_mu_m_xy)
    # else:
    #     Tsm_sm_matrix = None
    
    # Tsm_sm_matrix = comm.bcast(Tsm_sm_matrix, root=0)

    # Tsm_sm_matrix = build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,A//M,p_a_mu_m_xy)


    if timing:
        timer_step[1] = time.time()

    if verbose and mpi_rank == 0:
        print("-"*50)
        print("Solver info:")
        print("pi shape:",pi.shape)
        print("new_eta shape:",new_eta.shape)
        print("PY shape:",PY.shape, "PAMU shape:",PAMU.shape)
        print("p_a_mu_m_xy shape:",p_a_mu_m_xy.shape)
        
        print("               Type of Tsm_sm_matrix:",type(Tsm_sm_matrix))
        print("                 Tsm_sm_matrix shape:",Tsm_sm_matrix.shape)
        print("number of non-zeros in Tsm_sm_matrix:",Tsm_sm_matrix.nnz)
        print("number of     zeros in Tsm_sm_matrix:",M*Lx*Ly*M*Lx*Ly - Tsm_sm_matrix.nnz)
        print("               opacity of the matrix: {:7.4f}".format(Tsm_sm_matrix.nnz/(M*Lx*Ly*M*Lx*Ly) * 100.0), "%")
        print("           Sparse matrix memory size:", Tsm_sm_matrix.data.nbytes/1e6, " MB")
        print("Theoretical dense matrix memory size:", M*Lx*Ly*M*Lx*Ly*8/1e6, " MB")
        print("-"*50)

    if solver.solver_type_eta == 'iter':
        if solver.use_petsc:
            # new_eta = solver.function_solver_iter(Tsm_sm_matrix,eta,gamma,action_size,M,Lx,Ly,rho0,'bcgs','jacobi',device=solver.device)
            new_eta = solver.function_solver_iter(Tsm_sm_matrix,eta,gamma,action_size,M,Lx,Ly,rho0,solver.ksp_type,solver.pc_type,solver,device=solver.device,verbose=verbose)
        else :
            new_eta = solver.function_solver_iter(Tsm_sm_matrix,eta,gamma,M,Lx,Ly,rho0,device=solver.device,verbose=verbose)

    if solver.solver_type_eta == 'direct':
        if solver.use_petsc:
            # new_eta = solver.function_solver_direct(Tsm_sm_matrix,gamma,action_size,M,Lx,Ly,rho0,'bcgs','jacobi',device=solver.device)
            new_eta = solver.function_solver_direct(Tsm_sm_matrix,eta,gamma,action_size,M,Lx,Ly,rho0,solver.ksp_type,solver.pc_type,solver,device=solver.device, verbose=verbose)
        else :
            new_eta = solver.function_solver_direct(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,device=solver.device, verbose=verbose)

        # solver.solver_type_eta = 'iter'
    


    if timing:
        timer_step[2] = time.time()
        total_time = timer_step[-1] - timer_step[0]

        diff_time = np.zeros(timer_step.size)
        diff_time[0] = timer_step[0].copy()

        for i in range(1,timer_step.size):
            diff_time[i] = timer_step[i] - timer_step[i-1]

        timer_step = diff_time / total_time
        print("Rank:",mpi_rank,
            "Set T:",   "{:.2f}".format(timer_step[1]),
            "Solve T:", "{:.2f}".format(timer_step[2]),
            "Total:",   "{:.2f}".format(total_time))


    return new_eta, Tsm_sm_matrix

def get_next_state(state : np.ndarray, action : int, act_hdl : AgentActions, move = None):

    if move is None :
        move = act_hdl.action_move(action)

    n_rolls = [0,0]

    clipped_x = []
    dataClipped_x = None
    clipped_y = []
    dataClipped_y = None

    # Complex move
    if move[0] != 0 and move[1] != 0 :
        state = get_next_state(state, -1, act_hdl, move=[0,move[-1]])
        state = get_next_state(state, -1, act_hdl, move=[move[-2],0])

    # move in x
    elif move[-1] != 0 : 
        col = move[-1]
        if col < 0 :
            data2clip = np.s_[:,0]
            clipped_x = [i for i in range(-col)]
        else :
            data2clip = np.s_[:,-1]
            clipped_x = [-i for i in range(1,col+1)]

        dataClipped_x = state[data2clip].copy()

        n_rolls[-1] = -col

    # move in y 
    elif move[-2] != 0 : 
        row = move[-2]
        if row < 0 :
            data2clip = np.s_[0,:]
            clipped_y = [i for i in range(-row)]
        else :
            data2clip = np.s_[-1,:]
            clipped_y = [-i for i in range(1,row+1)]

        dataClipped_y = state[data2clip].copy()

        n_rolls[-2] = -row

    ## move in z

    state = np.roll(state,n_rolls,axis=(0,1))

    for i in clipped_y :
        state[i,:] = dataClipped_y
    for i in clipped_x :
        state[:,i] = dataClipped_x

    return state

# @profile
def linear_solve_Q(Tsm_sm_matrix, Lx, Ly, M, A, gamma,find_range, act_hdl, source_as_zero, reward, V, verbose=False, solver=None):
    """
    This function computes the Q function from the reward and the transition matrix
    """

    if solver.mpi_rank == 0:
        cost_move = 1.0 - gamma

        reward = reward.reshape(M*Ly*Lx)

        Tsm_sm_matrix = Tsm_sm_matrix.transpose()

        action_size = act_hdl.A
    else:
        action_size = None
        V = None
        Q = None

    if solver.solver_type_V == 'iter':
        if solver.use_petsc:
            V = solver.function_solver_iter(Tsm_sm_matrix,V,gamma,action_size,M,Lx,Ly,reward,solver.ksp_type,solver.pc_type,solver,device=solver.device)
        else :
            V = solver.function_solver_iter(Tsm_sm_matrix,V,gamma,M,Lx,Ly,reward,device=solver.device)

    if solver.solver_type_V == 'direct':
        if solver.use_petsc:
            V = solver.function_solver_direct(Tsm_sm_matrix,V,gamma,action_size,M,Lx,Ly,reward,solver.ksp_type,solver.pc_type,solver,device=solver.device)
        else :
            V = solver.function_solver_direct(Tsm_sm_matrix,gamma,M,Lx,Ly,reward,device=solver.device)

        # solver.solver_type_V = 'iter'

    if solver.mpi_rank == 0:
        # V = gamma * V 
        # V = gamma * V - 1.0

        V = V.reshape(M, Ly, Lx)

        state = np.empty((Ly, Lx))

        Q = np.zeros((Ly, Lx, M, A))
        for im in range(M):
            for a in range(A):
                state[:,:] = V[im,:,:] * gamma
                new_state = get_next_state(state, a, act_hdl)
                new_state = new_state - cost_move
                new_state[source_as_zero[:,0],source_as_zero[:,1]] = 0
                Q[:,:,im, a] = new_state

        Q = np.repeat(Q[np.newaxis,:,:,:,:], M,axis=0)
        Q = Q.flatten()
                
    return V.flatten(), Q

def find_grad(pi, Q, eta, L, PObs_lim):
    # grad J = sum_s sum_a Q(s,a) eta(s) sum_o grad pi(a|s o) f(o|s) 
    O, M, a_size = pi.shape
    Q_reshaped = np.reshape(Q, (L*M, a_size))

    # -------------
    etaobs = np.multiply(eta, PObs_lim)

    Q_eta = np.multiply( etaobs[:,:,np.newaxis] , Q_reshaped[np.newaxis,:,:] )
    Q_reduced = np.sum( np.reshape(Q_eta, (O,M,L,a_size)) , axis=2)
    # gradpi = [ 2 , 4] block-matrix
    Q_reduced -= np.mean(Q_reduced)
    gradpi = np.ones( pi.shape ) * Q_reduced
    #print('Q reduced', Q_reduced)

    return gradpi

def get_value(Q, pi, PObs_lim, L, rho0):
    O, M, a_size = pi.shape
    # We assume rho_starting is uniform on L but only in memory 0
    pi_L_M0 = np.tile(pi[:, 0, :], (1,1,L) ).reshape(O,-1)
    # We want pi_L_M0 to be the policy for M0 and O0/O1 repeated for L states 

    PObs_rep = np.repeat(PObs_lim[:,:L]*rho0[:L], a_size, axis=1)

    true_pi = np.sum(pi_L_M0 * PObs_rep[:, :], axis=0)
    value = np.sum(true_pi*Q[:L*a_size])
    return value

from types import new_class
def one_step(a, x, y, Lx, Ly, act_hdl):
    
    m = a//act_hdl.A
    newx = x
    newy = y

    action = a%act_hdl.A

    clip = lambda x, l, u: l if x < l else u if x > u else x

    newx = clip(newx + act_hdl.action_move(action)[-1], 0, Lx-1)
    newy = clip(newy + act_hdl.action_move(action)[-2], 0, Ly-1)

    # if (a%A == 0):
    #     newx = max(x-1,0)
    # if (a%A == 1):
    #     newx = min(x+1,Lx-1)
    # if (a%A == 2):
    #     newy = min(y+1,Ly-1)
    # if (a%A == 3):
    #     newy = max(y-1,0)
    return m, newx, newy

def single_traj_obs(pi, Lx, Ly, Lx0, Ly0, find_range, gamma, PObs, rho0, act_hdl):
  
    fixed_time = False
    if gamma>1: 
        fixed_time = True
        Tmax = gamma
    
    O, M, a_size = pi.shape
    m = 0
    s = np.random.choice(Lx*Ly, p=rho0[:Lx*Ly])
    x , y = (s%Lx, s//Lx)
    
    done = False
    trj=np.zeros((1,5))
    r = 0
    ret = 0
    r = np.random.choice(PObs.shape[0], p=PObs[:,s])
    o = r
    #if r > O-1: o = O-1
    #print(pi.shape)
    s0 = Lx0-1 + (Ly0-2)*Lx
    t = 0 
    while (not done):
        t += 1
        a = np.random.choice(a_size, p=pi[o,m,:])
        #print('m{}, o{}, a{}, x{}, y{}'.format( m,o, a, x,y))
        #print('s, PObs[s]', s, PObs[:,s])
        m, x, y = one_step(a, x, y, Lx, Ly, act_hdl)
        #print('new x, y', x, y)
        s = x + y*Lx
        r = np.random.choice(PObs.shape[0], p=PObs[:,s])
        o = r
        #if r > O-1: o = O-1
        found = False
        if (x+1-Lx0)**2 + (y+1-Ly0)**2 < find_range**2: found = True
        if found:
            ret += 1
            done = True
            #print('Found!')
        if fixed_time and t == Tmax:
            done = True
        #if np.random.rand()<1-gamma:
            #done = True
        trj = np.append(trj, [[a,x,y,m,r]], axis=0)
    return trj, ret, t
