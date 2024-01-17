import numpy as np
import json

from scipy.special import softmax as softmax
import matplotlib.pyplot as plt

import time 
from tqdm import tqdm
import concurrent.futures

# Linear solvers
# import fast_sparce_multiplications_2D_find_efficient as fast_mult
import fast_sparce_multiplications_2D as fast_mult
import local_linear_solvers
from local_linear_solvers import *

import importlib
importlib.reload(fast_mult)
importlib.reload(local_linear_solvers)
from local_linear_solvers import *


# ----------------------------------------------------------------------------
# Functions to set the solver functions for the linear system
class solver_opt:
    """ Class to handle the solver options. 

    It is possible to choose between different solvers for the linear system. Such as:
    - scipy
    - petsc
    - cupy
    - numpy
    - torch

    Attributes
    ----------
    function_solver_direct : function
        Wrapper to the function to solve the linear system.
    function_solver_iter : function
        Wrapper to the function to solve the linear system.
    solver_type_eta : str
        Type of solver to use for the linear system. options: 'direct' or 'iter'
    solver_type_V : str
        Type of solver to use for the linear system. options: 'direct' or 'iter'
    mpi_rank : int
        Rank of the process.
    mpi_size : int
        Number of processes.
    mpi_comm : MPI_Comm
        MPI communicator.
    use_petsc : bool
        Use PETSc library to solve the linear system.
    device : str
        Device to use. options: 'cpu' or 'gpu'
    Tmatrix_index : list
        Index of the non-zero elements in the T matrix.
    Tmatrix_p_index : list
        Index for the probability matrix.
    Tsm_sm_sp : scipy.sparse.csr_matrix
        Sparse matrix with the transition matrix.
    Tsm_sm_zero : list
        Sparse matrix with the transition matrix.

    Methods
    -------
    set_lib_solver(solver)
        Set the solver to use for the linear system.

    
    """

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

        # Varaibles to build T
        self.Tmatrix_index = None
        self.Tmatrix_p_index = None
        self.Tsm_sm_sp = None
        self.Tsm_sm_zero = None

    def set_lib_solver(self,solver):
        """ Set the solver to use for the linear system.
        """
        # Set the default solver
        if solver == None:
            solver = 'scipy'

        elif solver == 'scipy' :
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
            # self.ksp_type = 'gmres'          
            # self.pc_type = 'ilu'
            # self.solver_type_eta = 'iter'
            # self.solver_type_V = 'iter'

            self.function_solver_direct = load_info[0]
            self.function_solver_iter = load_info[1]
            self.mpi_rank = load_info[2]
            self.mpi_size = load_info[3]
            self.mpi_comm = load_info[4]
            self.free_petsc = load_info[5]

            self.device = 'cpu'
            OptDB = PETSc.Options()
            # PETSc.Options().setValue('cuda_device', '2')
            if self.mpi_rank == 0:
                print('Using petsc solver')

            # variable to keep the allocation of different variables
            self.Tsm_sm_sp_petsc = None
            self.var_list = None
            self.A = None
            self.b = None
            self.x = None
            self.ones = None
            self.ksp = None
            self.transpose = False

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
# Funtion to handle the parameters of all the system
class parameters:
    """ Class to handle the parameters of the system.

    Attributes
    ----------
    agent : parameters_agent
        Parameters of the agent.
    env : parameters_enviroment
        Parameters of the enviroment.
    plume : parameters_plume
        Parameters of the plume.
    optim : parameters_optimization
        Parameters of the optimization.

    Methods
    -------
    set_parameters(json_file)
        Set the parameters of the system from a json file.

    
    """
    def __init__(self):
        self.agent = parameters_agent()
        self.env = parameters_environment()
        self.plume = parameters_plume()
        self.optim = parameters_optimization()

    def set_parameters(self,json_file):
        self.env.set_values(json_file)
        self.agent.set_values(json_file,self.env.Lx)
        self.plume.set_values(json_file,self.env.Lx)
        self.optim.set_values(json_file)

class parameters_agent:
    """ Class to handle the parameters of the agent.

    Attributes
    ----------
    M : int
        Memory of the agent.
    O : int
        Number of observations of the agent.
    max_obs : int
        Maximum number of observations of the agent.
    A : int
        Number of actions of the agent.
    actions_move : list
        List of the actions of the agent.
    actions_name : list
        List of the names of the actions of the agent.
    act_hdl : AgentActions
        Handler of the actions of the agent.
    lr_th : float
        Learning rate of the agent.
    lr_val : list
        List of the learning rates of the agent.
    lr_time_frac : list
        List of the learning rates of the agent.
    cost_move : float
        Cost of moving of the agent.
    reward_find : float
        Reward of finding the source of the agent.
    gamma : float
        Discount factor of the agent.
    AxM : int
        Number of actions times the memory of the agent.


    Methods
    -------
    set_values(json_file,Lx)
        Set the parameters of the agent from a json file.


    """

    def __init__(self):
        # Default values
        self.M : int = 1
        self.O : int = 2
        self.max_obs : int = 10

        self.A : int = 0
        self.actions_move = None
        self.actions_name = None

        self.act_hdl = None

        self.lr_th : float = 0.001
        self.lr_val : list = [0.1,0.01,0.001]
        self.lr_time_frac : list = [0.0,0.1,1.0]

        self.cost_move = None
        self.reward_find : float = 0.0
        self.gamma = 'auto'

        self.AxM : int = 0

    def set_values(self,json_file,Lx):
        self.M = json_file.get('M',self.M)
        self.O = json_file.get('O',self.O)
        self.max_obs = json_file.get('max_obs',self.max_obs)

        self.A = json_file.get('A',self.A)
        self.actions_move = json_file.get('actions_move',self.actions_move)
        self.actions_name = json_file.get('actions_name',self.actions_name)

        self.act_hdl = AgentActions(self.A)

        for i in range(self.A):
            self.act_hdl.set_action(i,json_file['actions_move'][i],json_file['actions_name'][i])
    
        self.reward_find = json_file.get('reward_find',self.reward_find)

        if json_file.get("gamma",self.gamma) == 'auto':
            self.gamma = 1.0 - 1/10**(Lx//35) * 0.005
        else:
            self.gamma = json_file["gamma"]

        self.lr_th = json_file.get('lr_th',self.lr_th)
        if self.lr_th == 'auto':
            self.lr_val = json_file.get('lr_val',self.lr_val)
            self.lr_time_frac = json_file.get('lr_time_frac',self.lr_time_frac)

        self.cost_move = json_file.get('cost_move',1-self.gamma)

        self.AxM = self.A * self.M

class parameters_environment:
    """ Class to handle the parameters of the environment.

    Attributes
    ----------
    Lx : int
        Size of the environment in the x direction.
    Ly : int
        Size of the environment in the y direction.
    Lx0 : float
        Position of the source in the x direction.
    Ly0 : float
        Position of the source in the y direction.
    find_range : float
        Range of the agent to find the source.
    sigma : float
        Standard deviation of the gaussian of the source.
    L : int
        Size of the environment.

    Methods
    -------
    set_values(json_file)
        Set the parameters of the environment from a json file.


    """

    def __init__(self):
        # Default values
        self.Lx  : int = 35
        self.Ly  : int = 60
        self.Lx0 : int = 0
        self.Ly0 : int = 50

        self.find_range : int = 1.1
        self.sigma : int = 4

        self.factor_dim : int = 1

        self.L : int = 0

    def set_values(self,json_file):

        self.factor_dim = json_file.get('factor_dim',self.factor_dim)

        self.Ly  = json_file.get('Ly',self.Ly) * self.factor_dim
        self.Ly  = int(self.Ly)
        self.Lx  = json_file.get('Lx',self.Lx) * self.factor_dim
        self.Lx  = int(self.Lx)

        self.Ly0 = json_file.get('Ly0',self.Ly0) * self.factor_dim
        self.Lx0 = self.Lx / 2.

        self.find_range = json_file.get('find_range',self.find_range)
        self.sigma = json_file.get('sigma',self.sigma)

        self.L = int(self.Lx * self.Ly)

class parameters_plume:
    """ Class to handle the parameters of the plume.

    Attributes
    ----------
    coarse : int
        Coarse of the plume.
    dth : int
        Threshold of the plume.
    symmetry : int
        Symmetry of the plume.
    replica : int
        Replica of the plume.
    D : float
        Diffusion coefficient of the plume.
    V : float
        Wind speed of the plume.
    tau : float
        Time of the plume.
    beta : int
        Beta of the plume.
    plume_stat : str
        Statistics of the plume. "Bernoulli" or "Poisson"
    experimental : bool
        Use experimental data for the plume.
    adjust_factor : float
        Factor to adjust the size of the plume.

    Methods
    -------
    set_values(json_file)
        Set the parameters of the plume from a json file.

        
    """

    def __init__(self):
        # Default values
        self.coarse : int = 0
        self.dth    : int = 5

        self.symmetry : int = 1
        self.replica  : int = 0

        self.D     : float = 50.0
        self.V     : float = 100.0
        self.tau   : float = 2000.0
        self.beta  : int = 5

        self.plume_stat :  str = "Poisson" # "Bernoulli" or "Poisson"
        self.experimental  : bool = False     # Experimental plume
        self.adjust_factor  = "auto"    # Factor for the model plume

    def set_values(self,json_file,Lx):

        self.coarse = json_file.get('coarse',self.coarse)
        self.dth = json_file.get('dth',self.dth)

        self.symmetry = json_file.get('symmetry',self.symmetry)
        self.replica = json_file.get('replica',self.replica)

        self.D = json_file.get('D',self.D)
        self.V = json_file.get('V',self.V)
        self.tau = json_file.get('tau',self.tau)
        self.beta = json_file.get('beta',self.beta)

        self.plume_stat = json_file.get('plume_stat',self.plume_stat)
        self.experimental = json_file.get('experimental',self.experimental)

        if json_file.get("adjust_factor",self.adjust_factor) == 'auto':
            self.adjust_factor = Lx/75
            # self.adjust_factor = 1.0 / self.adjust_factor
        else:
            self.adjust_factor = json_file.get('adjust_factor')
        
class parameters_optimization:
    """ Class to handle the parameters of the optimization.

    Attributes
    ----------
    tol_eta : float
        Tolerance of the eta convergence.
    tol_Q : float
        Tolerance of the Q convergence.
    tol_conv : float
        Tolerance of the convergence of the optimization.
    max_iter_method : int
        Maximum number of iterations using the iterative method Jacobi.
    init_direct : int
        Number of optimization steps using the direct method at the beginning of the optimization.
    Ntot : int
        Total number of steps of the optimization.
    Nprint : int
        Number of steps to print the optimization.
    minimum_iter : int
        Minimum number of steps of the optimization.
    new_policy : bool
        Use a new policy for the optimization.
    unbias : bool
        Unbias the optimization.
    folder_restart : str
        Folder to read the policy for the optimization.

    Methods
    -------
    set_values(json_file)
        Set the parameters of the plume from a json file.

        
    """

    def __init__(self):
        # Default values
        self.tol_eta  : float = 1e-8
        self.tol_Q    : float = 1e-8
        self.tol_conv : float = 1e-8
        
        self.max_iter_method : int = 7
        self.init_direct :int = 1000

        self.Ntot         : int = 1000
        self.Nprint       : int = 100
        self.minimum_iter : int = 50

        self.new_policy : bool = True
        self.unbias     : bool = False
        self.folder_restart : str = None

    def set_values(self,json_file):
        self.tol_eta   = json_file.get('tol_eta',self.tol_eta)
        self.tol_Q     = json_file.get('tol_Q',self.tol_Q)
        self.tol_conv  = json_file.get('tol_conv',self.tol_conv)

        self.max_iter_method = json_file.get('max_iter_method',self.max_iter_method)
        self.init_direct = json_file.get('init_direct',self.init_direct)    

        self.Ntot         = json_file.get('Ntot',self.Ntot)
        self.Nprint       = json_file.get('Nprint',self.Nprint)
        self.minimum_iter = json_file.get('minimum_iter',self.minimum_iter)

        self.new_policy = json_file.get('new_policy',self.new_policy)
        self.unbias     = json_file.get('unbias',self.unbias)
        self.folder_restart = json_file.get('folder_restart',self.folder_restart)

# ----------------------------------------------------------------------------
# Functions to load and print the parameters of the system
def create_output_folder_name(fsc, method, lib_solve_linear_system):
        """ Create the name of the output folder.

        Parameters
        ----------
        fsc : parameters
            Parameters of the system.
        method : str
            Method to use for the optimization.
        lib_solve_linear_system : str
            Library to use for the linear system.

        Returns
        -------
        name_folder : str
            Name of the output folder.

        """
        name_folder = 'FSC_'
        name_folder += f'Agent_'
        name_folder += f'A{fsc.agent.A}'
        name_folder += f'M{fsc.agent.M}'
        name_folder += f'O{fsc.agent.O}'
        name_folder += f'g{fsc.agent.gamma}'

        name_folder += f'_Env_'
        name_folder += f'Lx{fsc.env.Lx}'
        name_folder += f'Ly{fsc.env.Ly}'

        name_folder += f'_Plume_'
        name_folder += f'exp{fsc.plume.experimental}'
        name_folder += f'sym{fsc.plume.symmetry}'
        name_folder += f'dth{fsc.plume.dth}'

        name_folder += f'_Opt_'
        name_folder += f'mth-{method}'
        name_folder += f'slvr-{lib_solve_linear_system}'
        name_folder += f'Ntot{fsc.optim.Ntot}'

        return name_folder

def print_parameters(fsc,method,lib_solve_linear_system,device):
    """ Print the parameters of the system.

    Parameters
    ----------
    fsc : parameters
        Parameters of the system.
    method : str
        Method to use for the optimization.
    lib_solve_linear_system : str
        Library to use for the linear system.
    device : str
        Device to use.

    """
    print("System parameters:")
    print("-"*50)
    # print agent parameters
    print("Agent parameters:")
    print("Memory: ",fsc.agent.M,"   Observation: ",fsc.agent.O,"   Max obs: ",fsc.agent.max_obs)
    print("    Gamma: ",fsc.agent.gamma)
    print("Cost move: ",fsc.agent.cost_move,"   Reward find: ",fsc.agent.reward_find)
    print("      Actions: ",fsc.agent.A)
    print(" Actions name: ",fsc.agent.actions_name)
    print(" Actions move: ",fsc.agent.actions_move)
    print("Learning rate: ",fsc.agent.lr_th)
    if fsc.agent.lr_th == 'auto':
        print("       Learning rate values: ",fsc.agent.lr_val)
        print("Learning rate time fraction: ",fsc.agent.lr_time_frac)
    print("-"*50)
    # print enviroment parameters
    print("Enviroment parameters:")
    print("  Lx: {:<8}   Ly: {:<8}".format(fsc.env.Lx,fsc.env.Ly))
    print(" Lx0: {:<8}  Ly0: {:<8}".format(fsc.env.Lx0,fsc.env.Ly0))
    print("Find range: ",fsc.env.find_range,"   Sigma: ",fsc.env.sigma)
    print("-"*50)
    # print plume parameters
    print("Plume parameters:")
    print("Experimental: ",fsc.plume.experimental)
    print("D: ",fsc.plume.D,"   V: ",fsc.plume.V,"   Tau: ",fsc.plume.tau,"   Beta: ",fsc.plume.beta)
    print("Coarse: ",fsc.plume.coarse,"   Dth: ",fsc.plume.dth)
    print("Symmetry: ",fsc.plume.symmetry,"   Replica: ",fsc.plume.replica)
    print("Plume stat: ",fsc.plume.plume_stat,"   Adjust factor: ",fsc.plume.adjust_factor)
    print("-"*50)
    # print optimization parameters
    print("Optimization parameters:")
    print('method: {}   solver: {}   device: {}'.format(method,lib_solve_linear_system,device))
    print("Ntot:",fsc.optim.Ntot,"   Nprint:",fsc.optim.Nprint)
    print("Tol_conv:",fsc.optim.tol_conv, "   minimum_step:",fsc.optim.minimum_iter)
    print("Init_direct:",fsc.optim.init_direct,"   Max_iter_method:",fsc.optim.max_iter_method)
    print("Tol_eta:",fsc.optim.tol_eta, "   Tol_Q:",fsc.optim.tol_Q)
    print("new_policy:",fsc.optim.new_policy, "   unbias:",fsc.optim.unbias)
    print("Folder restart:",fsc.optim.folder_restart)
    print("-"*50)

def get_max_value(Lx, Ly, Lx0, Ly0, M, find_range, rho0, gamma):
    """ Print the maximum value can be obtained. 

    Parameters
    ----------
    Lx : int
        Size of the environment in the x direction.
    Ly : int    
        Size of the environment in the y direction.
    Lx0 : float
        Position of the source in the x direction.
    Ly0 : float
        Position of the source in the y direction.
    find_range : float
        Range of the agent to find the source.
    rho0 : numpy.ndarray
        Initial distribution of the agent.
    gamma : float
        Discount factor of the agent.

    """
    rho_map = rho0.reshape((Ly,Lx,M))
    rho_map = rho_map[:,:,0]
    max_value = []
    for x in range(Lx):
        for y in range(Ly):

            if rho_map[y,x] < 1e-10:
                continue
            
            short_path = int(np.abs(x-Lx0) + np.abs(y-Ly0) - find_range)
            discount = -np.sum([gamma**i * (1.0 - gamma) for i in range(short_path)])
            max_value.append(discount*rho_map[y,x])

    print("Avg max value: {:.8f}".format(np.sum(max_value)))
    print("-"*50)

def save_parameters(fsc,name_folder):
    param = {}
    param.update(fsc.agent.__dict__)
    param.update(fsc.env.__dict__)
    param.update(fsc.plume.__dict__)
    param.update(fsc.optim.__dict__)
    param.pop('act_hdl')
    param.pop('factor_dim')
    with open(name_folder+'/input.dat', 'w') as fp:
        json.dump(param, fp)


# ----------------------------------------------------------------------------
# Functions to create the initial distribution of the agent
def create_cplume(Lx, Ly, Lx0, Ly0, D, V, tau, aR, alpha=1.0): 
    """ Returns a diffusion plume with given parameters. 
    
    Parameters
    ----------  
    Lx : int
        Size of the environment in the x direction.
    Ly : int
        Size of the environment in the y direction.
    Lx0 : float
        Position of the source in the x direction.
    Ly0 : float
        Position of the source in the y direction.
    D : float
        Diffusion coefficient of the plume.
    V : float
        Wind speed of the plume.
    tau : float
        Time of the plume.
    aR : int
        Beta of the plume.
    alpha : float
        Factor to adjust the size of the plume.

    Returns
    -------
    cplume : numpy.ndarray
        Diffusion plume with given parameters.

    """
    spacex = np.arange(1,Lx+1)-(Lx+1)/2.
    spacey = np.arange(Ly)-(Ly0-1)
    xx, yy = np.meshgrid(spacex, spacey)
    rr = np.sqrt(xx**2 + yy**2)
    lam = np.sqrt(D*tau/(1+V*V*tau/D/4))
    cplume = aR/(rr* alpha+0.01)*np.exp(-rr/lam * alpha -yy*V/2/D *alpha)
    return cplume

def create_random_Q0(agent, env):
    """
    Returns an approx Q for a random diffusion.

    Parameters
    ----------
    agent : parameters_agent
        Parameters of the agent.
    env : parameters_enviroment
        Parameters of the enviroment.

    Returns
    -------
    random_Q0 : numpy.ndarray
        Approx Q for a random diffusion.

    """

    # ------------------------------------------------------------
    # Environment parameters
    Lx = env.Lx   ; Ly = env.Ly
    Lx0 = env.Lx0 ; Ly0 = env.Ly0

    # Agent parameters
    gamma = agent.gamma
    a_size = agent.AxM
    M = agent.M
    cost_move = agent.cost_move
    reward_find = agent.reward_find
    # ------------------------------------------------------------

    spacex = np.arange(1,Lx+1)-(Lx+1)/2.
    spacey = np.arange(Ly)-(Ly0-1)

    xx, yy = np.meshgrid(spacex, spacey)
    rr = xx**2 + yy**2
    random_Q0 = (-1/(1-gamma)*cost_move) 
    random_Q0 /= (1 + 2/np.abs(rr)) 
    random_Q0 += (1 - 1/ (1 + 2/np.abs(rr)))*reward_find    
    random_Q0 = np.tile(np.repeat(random_Q0, a_size), M).reshape(-1)
    return random_Q0

def create_plume_from_exp(Lx, Ly, Lx0, Ly0, cmax, data):
    """ Returns a diffusion plume from a experimental  data.

    Parameters
    ----------
    Lx : int
        Size of the environment in the x direction.
    Ly : int
        Size of the environment in the y direction.
    Lx0 : float
        Position of the source in the x direction.
    Ly0 : float
        Position of the source in the y direction.
    cmax : int
        Beta of the plume.
    data : numpy.ndarray
        Experimental data.

    Returns
    -------
    new_plume : numpy.ndarray
        shape (Lx, Ly)
        Diffusion plume from a experimental data.

    """
    Lx0 = int(Lx0)
    Ly0 = int(Ly0)
    
    exp_plume_mat = data.copy()
    # -----------------------------------------------------
    # To visualize the experimental plume could use the function:
    # plot_plume()
    # from the file fsc_visualize_tools.py
    # -----------------------------------------------------

    # plt.imshow(exp_plume_mat)
    # plt.show()
    # print(f"exp_plume_mat.shape {exp_plume_mat.shape}")
    
    center_x, center_y = np.unravel_index(np.argmax(exp_plume_mat, axis=None), exp_plume_mat.shape)
    
    
    
    # symmetrize exp_plume_mat
    min_size = min(center_x, exp_plume_mat.shape[0]-center_x)
    exp_plume_mat = exp_plume_mat[center_x - min_size:center_x + min_size,:].copy()

    # padding into shape
    new_size_x = Lx
    new_size_y = Ly
    
    print(f"exp_plume_mat.shape {exp_plume_mat.shape}")
    exp_Lx, exp_Ly = exp_plume_mat.shape 
    
    assert Ly0+10 <= exp_Ly, 'Size of experimental plume too small' 
    
    print(f"Lx{Lx} Ly{Ly} expLx{exp_Lx} expLy{exp_Ly}")
    print(f"center_x{center_x} min_size{min_size}")
    
    new_plume = np.zeros((new_size_x, new_size_y))
    
    if Lx0 < min(exp_Lx-center_x, center_x):
        # print('yes')
        momo = exp_plume_mat[center_x-(Lx0):center_x+(new_size_x-Lx0),
                     :Ly0+10].copy()
        
    else:
        # print('no')
        momo = np.zeros((new_size_x, Ly0+10))
        momo[Lx0-center_x:Lx0-center_x+exp_Lx,:] = exp_plume_mat[:,:Ly0+10].copy()
    
    new_plume[:,-(Ly0+10):] = momo.copy()
    
    #new_plume[new_plume < 0.1] = 0.
    #renorm = np.sum(new_plume[:, -1])/cmax
    #new_plume = np.swapaxes(new_plume/renorm,0,1)
    new_plume = np.swapaxes(new_plume,0,1)
    
    return new_plume[::-1,:]

def average_reward(Tsm_sm_matrix, M, Lx, Ly, cost_move,source_as_zero):
    """ This function compute the average reward from the Transition matrix T

    Parameters
    ----------
    Tsm_sm_matrix : numpy.ndarray
        Transition matrix.
    M : int
        Memory of the agent.
    Lx : int
        Size of the environment in the x direction.
    Ly : int
        Size of the environment in the y direction.
    cost_move : float
        Cost of moving of the agent.
    source_as_zero : numpy.ndarray
        Position of the source.

    Returns
    -------
    RR : numpy.ndarray
        shape (M, Ly, Lx)
        Average reward from the Transition matrix T.
    """
    # Tsm_sm = Tsm_sm_matrix.reshape(M, Ly, Lx, M, Ly, Lx)
    # RR = -cost_move * Tsm_sm.sum(axis=(0,1,2))

    RR = np.full((M, Ly, Lx),-cost_move)
    RR[:,source_as_zero[:,0],source_as_zero[:,1]] = 0.0

    return RR

def create_PObs_RR(agent, env, plume, data, source_as_zero=None):
    """
    This function should create the Probability of Observation and Rewards matrices.

    Parameters
    ----------
    agent : parameters_agent
        Parameters of the agent.
    env : parameters_enviroment
        Parameters of the enviroment.
    plume : parameters_plume
        Parameters of the plume.
    data : numpy.ndarray
        Experimental data.
    source_as_zero : numpy.ndarray
        Position of the source.

    Returns
    -------
    PObs_lim : numpy.ndarray
        shape (diff_obs, Lx*Ly*M)
        Probability of Observation matrix.
    RR : numpy.ndarray
        shape (M, Ly, Lx)
        Rewards matrix.
    PObs : numpy.ndarray
        shape (max_obs, Lx*Ly)
        Probability of Observation matrix.
    RR_np : numpy.ndarray
        shape (M*Ly*Lx)
        Rewards matrix.

    """

    # ------------------------------------------------------------
    # Environment parameters
    Lx = env.Lx   ; Ly = env.Ly
    Lx0 = env.Lx0 ; Ly0 = env.Ly0
    find_range = env.find_range

    # Agent parameters
    M = agent.M ; A = agent.A
    diff_obs = agent.O
    cost_move = agent.cost_move
    reward_find = agent.reward_find
    max_obs = agent.max_obs

    # Plume parameters
    cmax = plume.beta
    V = plume.V
    D = plume.D
    tau = plume.tau
    plume_stat = plume.plume_stat
    exp_plume = plume.experimental
    adjust_factor = 1.0 / plume.adjust_factor
    # ------------------------------------------------------------

    spacex = np.arange(1,Lx+1)-(Lx+1)/2.
    spacey = np.arange(Ly)-(Ly0-1)

    # CHECK IF ORDER IS MANTAINED
    #cplume = cmax * np.multiply.outer((np.tanh(-spacey/5)+1)*0.5*np.exp(-np.abs(spacey)**2/450), np.exp(-np.abs(spacex)**2/30))
    xx, yy = np.meshgrid(spacex, spacey)
    
    if exp_plume:
        cplume = create_plume_from_exp(Lx, Ly, Lx0, Ly0, cmax, data)
    else:
        cplume = create_cplume(Lx, Ly, Lx0, Ly0, D, V, tau, cmax, alpha=adjust_factor)
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

    RR_np = average_reward(None, M, Lx, Ly, cost_move, source_as_zero)
    RR_np = RR_np.reshape(M * Ly * Lx)

    if A == 4:
        RR = fast_mult.rewards_four_2d(M, Lx, Ly, Lx0+1, Ly0+1, find_range, cost_move, reward_find, max_obs)
    else :
        RR = RR_np
        
    # elif A == 5:
    #     RR = fast_mult.rewards_five_2d(M, Lx, Ly, Lx0+1, Ly0+1, find_range, cost_move, reward_find, max_obs)


    return np.abs(PObs_lim), RR, np.abs(PObs), RR_np

# ----------------------------------------------------------------------------
# Functions to solve the linear system
def iterative_solve_eta(agent, env, optim, pi, PObs_lim, rho0, eta0):
    """
    This function should solve the following:
    $ \eta = (1 - \gamma T)^{-1} \rho $ 

    Parameters
    ----------
    agent : parameters_agent
        Parameters of the agent.
    env : parameters_enviroment
        Parameters of the enviroment.
    optim : parameters_optimization
        Parameters of the optimization.
    pi : numpy.ndarray
        Policy of the agent.
    PObs_lim : numpy.ndarray
        Probability of Observation matrix.
    rho0 : numpy.ndarray
        Initial distribution of the agent.
    eta0 : numpy.ndarray
        Initial eta of the agent.

    Returns
    -------
    new_eta : numpy.ndarray
        shape (Lx*Ly*M)
        New eta of the agent.

    """
    # ------------------------------------------------------------
    # Environment parameters
    Lx = env.Lx   ; Ly = env.Ly
    Lx0 = env.Lx0 ; Ly0 = env.Ly0
    find_range = env.find_range

    # Agent parameters
    gamma = agent.gamma

    # Optimization parameters
    tol = optim.tol_eta
    # ------------------------------------------------------------

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
            # print('Converged in {} iterations'.format(i))
            return new_eta
        #if ((i+1)%1000 == 0):
        #    print('eta: i {}, delta {}'.format(i, delta))
        eta = new_eta.copy()
    print('NOT CONVERGED - eta')
    return new_eta

# @profile
def iterative_solve_Q(agent, env, optim, pi, PObs_lim, RR, Q0):
    """
    This function should solve the following:
    $ Q = (1 - \gamma T^T)^{-1} \RR $ 

    Parameters
    ----------
    agent : parameters_agent
        Parameters of the agent.
    env : parameters_enviroment
        Parameters of the enviroment.
    optim : parameters_optimization
        Parameters of the optimization.
    pi : numpy.ndarray
        Policy of the agent.
    PObs_lim : numpy.ndarray
        Probability of Observation matrix.  
    RR : numpy.ndarray
        Rewards matrix.
    Q0 : numpy.ndarray
        Initial Q of the agent.

    Returns
    -------
    new_Q : numpy.ndarray
        shape (Lx*Ly*M*A)
        New Q of the agent.

    """
    # ------------------------------------------------------------
    # Environment parameters
    Lx = env.Lx   ; Ly = env.Ly
    Lx0 = env.Lx0 ; Ly0 = env.Ly0
    find_range = env.find_range

    # Agent parameters
    gamma = agent.gamma

    # Optimization parameters
    tol = optim.tol_Q
    # ------------------------------------------------------------
    
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
            #delta = np.sum(delta*delta)
            delta = np.max(delta*delta)
            if ( delta < tol2): 
                # print('Converged in {} iterations'.format(i))
                return new_Q
        #if ((i+1)%1000 == 0):
        #    print('Q: i {}, delta {}, tol^2 {}'.format(i, delta, tol2))
        Q = new_Q.copy()
    print('NOT CONVERGED - Q')
    return new_Q

# Define the action space
class AgentActions():
    """ Class to handle the actions of the agent.

    Attributes
    ----------
    A : int
        Number of actions.
    actions : numpy.ndarray
        Actions of the agent.
    actions_names : dict
        Names of the actions.

    Methods
    -------
    set_action(action_index, action, action_name)
        Set the action of the agent.

    action_move(action_index)
        Return the action of the agent.

    action_name(action_index)
        Return the name of the action of the agent.

    """
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
def linear_solve_eta(agent, env, optim, eta, rho0, pi, PObs_lim, source_as_zero, verbose=False, solver = None):
    """
    This function should solve the following:
    $ \eta = (1 - \gamma T)^{-1} \rho $

    Parameters
    ----------
    agent : parameters_agent
        Parameters of the agent.
    env : parameters_enviroment
        Parameters of the enviroment.
    optim : parameters_optimization
        Parameters of the optimization.
    eta : numpy.ndarray
        eta of the agent.
    rho0 : numpy.ndarray
        Initial distribution of the agent.
    pi : numpy.ndarray
        Policy of the agent.
    PObs_lim : numpy.ndarray
        Probability of Observation matrix.
    source_as_zero : numpy.ndarray
        Position of the source.
    verbose : bool
        Print information of the optimization.
    solver : solver
        Solver object that contains the solver to use.

    Returns
    -------
    new_eta : numpy.ndarray
        shape (Lx*Ly*M)
        New eta of the agent.

    """
    # ------------------------------------------------------------
    # Environment parameters
    Lx = env.Lx   ; Ly = env.Ly
    Lx0 = env.Lx0 ; Ly0 = env.Ly0
    find_range = env.find_range

    # Agent parameters
    act_hdl = agent.act_hdl
    gamma = agent.gamma

    # Optimization parameters
    tol = optim.tol_eta
    max_iter = optim.max_iter_method
    # ------------------------------------------------------------

    timing = False
    if timing:
        timer_step = np.zeros(3)
        timer_step[0] = time.time()

    mpi_rank = solver.mpi_rank
    mpi_size = solver.mpi_size

    if mpi_rank == 0:
        # O = Observations, M = Memory, A = Actions
        O, M, A = pi.shape
        
        # PY has size ~ 10^5
        PY = PObs_lim.reshape(O, M, Ly, Lx)
        # PY has size ~ 10^2
        PAMU = pi.reshape(O, M, M, A//M)
        # PAMU = softmax( np.zeros((O, M, M, A//M)), 2)
        
        p_a_mu_m_xy = np.einsum( 'omyx, omna -> anmyx', PY, PAMU)
        # T [ s'm'  sm] = sum_a, mu p(s'm' | sm a mu) p(a mu | sm)
        #               = sum_a, mu p(s'm' | sm a mu) sum_y f(y | s) pi(a mu | y m)

        # Tsm_sm_matrix = build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,act_hdl.A,p_a_mu_m_xy)
        # Tsm_sm_matrix = build_Tsm_sm_sparse_2(M,Lx,Ly,Lx0,Ly0,find_range,p_a_mu_m_xy,act_hdl,source_as_zero)
        # Tsm_sm_matrix = build_Tsm_sm_sparse_3(M,Lx,Ly,Lx0,Ly0,find_range,p_a_mu_m_xy,act_hdl,source_as_zero,solver)
        Tsm_sm_matrix = build_Tsm_sm_sparse_4(M,Lx,Ly,Lx0,Ly0,find_range,p_a_mu_m_xy,act_hdl,source_as_zero,solver)

        action_size = act_hdl.A

    else:
        Tsm_sm_matrix = None
        action_size = None
        M = None

    if timing:
        timer_step[1] = time.time()

    if verbose and mpi_rank == 0:
        print("T matrix info:")
        # print("pi shape:",pi.shape)
        # print("PY shape:",PY.shape, "PAMU shape:",PAMU.shape)
        # print("p_a_mu_m_xy shape:",p_a_mu_m_xy.shape)
        
        print("               Type of Tsm_sm_matrix:",type(Tsm_sm_matrix))
        print("                 Tsm_sm_matrix shape:",Tsm_sm_matrix.shape)
        print("number of non-zeros in Tsm_sm_matrix:",Tsm_sm_matrix.nnz)
        print("number of     zeros in Tsm_sm_matrix:",M*Lx*Ly*M*Lx*Ly - Tsm_sm_matrix.nnz)
        print("               opacity of the matrix: {:7.4f}".format(Tsm_sm_matrix.nnz/(M*Lx*Ly*M*Lx*Ly) * 100.0), "%")
        # print("           Sparse matrix memory size:", Tsm_sm_matrix.data.nbytes/1e6, " MB")
        print("           Sparse matrix memory size:", Tsm_sm_matrix.data.nbytes/1e6 + Tsm_sm_matrix.indptr.nbytes/1e6 + Tsm_sm_matrix.indices.nbytes/1e6, " MB")
        print("Theoretical dense matrix memory size:", M*Lx*Ly*M*Lx*Ly*8/1e6, " MB")
        print('-'*77)

    if solver.solver_type_eta == 'iter':
        if solver.use_petsc:
            new_eta = solver.function_solver_iter(Tsm_sm_matrix, eta, rho0, gamma, action_size, M, Lx, Ly, solver.ksp_type, solver.pc_type, solver, device=solver.device,verbose=verbose)
        else :
            new_eta = solver.function_solver_iter(Tsm_sm_matrix,eta,gamma,M,Lx,Ly,rho0,device=solver.device,verbose=verbose)

    if solver.solver_type_eta == 'direct':
        if solver.use_petsc:
            new_eta = solver.function_solver_direct(Tsm_sm_matrix, eta, rho0, gamma, action_size, M, Lx, Ly, tol, max_iter, solver.ksp_type,solver.pc_type, solver, device=solver.device, verbose=verbose)
        else :
            new_eta = solver.function_solver_direct(Tsm_sm_matrix, eta, rho0, gamma, M, Lx, Ly ,tol, max_iter, device=solver.device, verbose=verbose)

        # solver.solver_type_eta = 'iter'    

    # if timing:
    if timing and (mpi_rank in [0,1,mpi_size/2,mpi_size-1]):
        timer_step[2] = time.time()
        total_time = timer_step[-1] - timer_step[0]

        diff_time = np.zeros(timer_step.size)
        diff_time[0] = timer_step[0].copy()

        for i in range(1,timer_step.size):
            diff_time[i] = timer_step[i] - timer_step[i-1]

        timer_step = diff_time / total_time
        print("Rank:",mpi_rank,
            "Set T:",   "{:.2f} %".format(timer_step[1]),
            "Solve eta:", "{:.2f} %".format(timer_step[2]),
            "Total:",   "{:.4f}".format(total_time))


    return new_eta, Tsm_sm_matrix

def get_next_state(state : np.ndarray, action : int, act_hdl : AgentActions, move = None):
    """
    This function should return the next state of the agent after the action.

    Parameters
    ----------
    state : numpy.ndarray
        State of the agent.
    action : int
        Action of the agent.
    act_hdl : AgentActions
        Handler of the actions of the agent.
    move : list
        Move of the agent.

    Returns
    -------
    state : numpy.ndarray
        State of the agent.

    """

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
def linear_solve_Q(agent, env, optim, Tsm_sm_matrix_sp, V, reward, source_as_zero, verbose=False, solver=None):
    """
    This function computes the Q function from the reward and the transition matrix

    Parameters
    ----------
    agent : parameters_agent
        Parameters of the agent.
    env : parameters_enviroment
        Parameters of the enviroment.
    optim : parameters_optimization
        Parameters of the optimization.
    Tsm_sm_matrix_sp : numpy.ndarray
        Transition matrix.
    V : numpy.ndarray
        Value function.
    reward : numpy.ndarray
        Reward function.
    source_as_zero : numpy.ndarray
        Position of the source.
    verbose : bool
        Print information of the optimization.
    solver : solver
        Solver object that contains the solver to use.

    Returns
    -------
    Q : numpy.ndarray
        shape (Lx*Ly*M*A)
        Q function of the agent.

    """
    # ------------------------------------------------------------
    # Environment parameters
    Lx = env.Lx   ; Ly = env.Ly

    # Agent parameters
    M = agent.M ; A = agent.A
    act_hdl = agent.act_hdl
    gamma = agent.gamma
    cost_move = agent.cost_move

    # Optimization parameters
    tol = optim.tol_Q 
    max_iter = optim.max_iter_method
    # ------------------------------------------------------------

    timing = False
    if timing:
        timer_step = np.zeros(4)
        timer_step[0] = time.time()

    if solver.mpi_rank == 0:
        reward = reward.reshape(M*Ly*Lx)
        Tsm_sm_matrix = Tsm_sm_matrix_sp.copy().tocsr()
        action_size = act_hdl.A
    else:
        Tsm_sm_matrix = None
        action_size = None
        Q = None

    if timing:
        timer_step[1] = time.time()
    if solver.solver_type_V == 'iter':
        if solver.use_petsc:
            solver.transpose = True
            V = solver.function_solver_iter(Tsm_sm_matrix,V,gamma,action_size,M,Lx,Ly,reward,solver.ksp_type,solver.pc_type,solver,device=solver.device)
            solver.transpose = True
        else :
            Tsm_sm_matrix = Tsm_sm_matrix.transpose()
            V = solver.function_solver_iter(Tsm_sm_matrix,V,gamma,M,Lx,Ly,reward,device=solver.device)

    if solver.solver_type_V == 'direct':
        if solver.use_petsc:
            solver.transpose = True
            # The transpose of Tsm_sm_matrix is doing by PETSc
            V = solver.function_solver_direct(Tsm_sm_matrix, V, reward, gamma, action_size, M, Lx, Ly, tol, max_iter,solver.ksp_type, solver.pc_type, solver, device=solver.device)
            solver.transpose = False
        else :
            Tsm_sm_matrix = Tsm_sm_matrix.transpose()
            V = solver.function_solver_direct(Tsm_sm_matrix, V, reward, gamma, M, Lx, Ly, tol, max_iter, device=solver.device)
        # solver.solver_type_V = 'iter'

    if timing:
        timer_step[2] = time.time()

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

    # if timing:
    if timing and (solver.mpi_rank in [0,1,solver.mpi_size/2,solver.mpi_size-1]):
        timer_step[3] = time.time()
        total_time = timer_step[-1] - timer_step[0]

        diff_time = np.zeros(timer_step.size)
        diff_time[0] = timer_step[0].copy()

        for i in range(1,timer_step.size):
            diff_time[i] = timer_step[i] - timer_step[i-1]

        timer_step = diff_time / total_time
        print("Rank:",solver.mpi_rank,
            "Transpose T:",   "{:.2f} %".format(timer_step[1]),
            "Solve V:",   "{:.2f} %".format(timer_step[2]),
            "Compute Q:", "{:.2f} %".format(timer_step[3]),
            "Total:",   "{:.4f}".format(total_time))

                
    return V.flatten(), Q

# ----------------------------------------------------------------------------
# Functions to compute the gradient, value
def find_grad(pi, Q, eta, L, PObs_lim):
    """ Compute the gradient of the cost function.

    Parameters
    ----------
    pi : numpy.ndarray
        Policy of the agent.
    Q : numpy.ndarray
        Q function of the agent.
    eta : numpy.ndarray
        Eta of the agent.
    L : int
        Number of states.
    PObs_lim : numpy.ndarray
        Probability of Observation matrix.

    Returns
    -------
    gradpi : numpy.ndarray
        shape (O, M, AxM)
        Gradient of the cost function.

    """
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
    """ Compute the value function of the agent.

    Parameters
    ----------
    Q : numpy.ndarray
        Q function of the agent.
    pi : numpy.ndarray
        Policy of the agent.
    PObs_lim : numpy.ndarray
        Probability of Observation matrix.
    L : int
        Number of states.
    rho0 : numpy.ndarray

    Returns
    -------
    value : float
        Value function of the agent.

    """

    O, M, a_size = pi.shape
    # We assume rho_starting is uniform on L but only in memory 0
    pi_L_M0 = np.tile(pi[:, 0, :], (1,1,L) ).reshape(O,-1)
    # We want pi_L_M0 to be the policy for M0 and O0/O1 repeated for L states 

    PObs_rep = np.repeat(PObs_lim[:,:L]*rho0[:L], a_size, axis=1)

    true_pi = np.sum(pi_L_M0 * PObs_rep[:, :], axis=0)
    value = np.sum(true_pi*Q[:L*a_size])
    return value


# ----------------------------------------------------------------------------
# Functions to compute the trajectory of the agent
from types import new_class
def one_step(a, x, y, Lx, Ly, act_hdl):
    """ Compute the next state of the agent.

    Parameters
    ----------
    a : int
        Action of the agent.
    x : int
        Position of the agent in the x direction.
    y : int
        Position of the agent in the y direction.
    Lx : int
        Size of the environment in the x direction.
    Ly : int
        Size of the environment in the y direction.
    act_hdl : AgentActions
        Handler of the actions of the agent.

    Returns
    -------
    m : int
        Memory of the agent.
    newx : int
        New position of the agent in the x direction.
    newy : int
        New position of the agent in the y direction.

    """

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

def single_traj_obs(pi, Lx, Ly, Lx0, Ly0, find_range, Tmax, PObs, rho0, act_hdl, progress_bar=False,save_traj=True):
    """ Compute the trajectory of the agent.

    Parameters
    ----------
    pi : numpy.ndarray
        Policy of the agent.
    Lx : int
        Size of the environment in the x direction.
    Ly : int
        Size of the environment in the y direction.
    Lx0 : int
        Position of the source in the x direction.
    Ly0 : int
        Position of the source in the y direction.
    find_range : float
        Range of the agent.
    Tmax : int
        Maximum number of steps.
    PObs : numpy.ndarray
        Probability of Observation matrix.
    rho0 : numpy.ndarray
        Initial distribution of the agent.
    act_hdl : AgentActions
        Handler of the actions of the agent.
    progress_bar : bool
        Print progress bar.
    save_traj : bool
        Save trajectory.

    Returns
    -------
    trj : numpy.ndarray
        Trajectory of the agent.
    success : int
        1 if the agent found the source, 0 otherwise.
    time_steps : int
        Number of steps of the agent.

    """

    fixed_time = True if Tmax > 1 else False
    
    O, M, a_size = pi.shape
    m = 0
    rho_map = rho0.reshape((Ly,Lx,M))
    rho_map = rho_map.sum(axis=2)

    s = np.random.choice(Lx*Ly, p=rho_map.flatten())
    x , y = (s%Lx, s//Lx)
    
    done = False
    trj=np.zeros((1,5))
    r = 0
    success = 0
    r = np.random.choice(PObs.shape[0], p=PObs[:,s])
    o = r
    #if r > O-1: o = O-1
    ##############
    # found = False
    # if (x+1-Lx0)**2 + (y+1-Ly0)**2 < find_range**2: found = True
    # if found:
    #   ret += 1
    #   done = True
    #   #print('Found!')
    # if fixed_time and t == Tmax:
    #   done = True
    #   #print('not found')
    # #if np.random.rand()<1-gamma:
    # #  done = True
    # trj = np.append(trj, [[a,x,y,m,r]], axis=0)
    # return trj, ret, t
    ##########
    #print(pi.shape)
    s0 = Lx0-1 + (Ly0-2)*Lx
    time_steps = 0 

    if progress_bar:
        pbar = tqdm(total=Tmax)

    while (not done):

        if progress_bar:
            pbar.update(1)

        time_steps += 1
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
        if (x+1-Lx0)**2 + (y+1-Ly0)**2 < find_range**2: 
            found = True
            success     = 1
            done = True
            #print('Found!')
        if fixed_time and time_steps == Tmax:
            done = True
        #if np.random.rand()<1-gamma:
            #done = True
        if save_traj:
            trj = np.append(trj, [[a,x,y,m,r]], axis=0)

    if progress_bar:
        pbar.close()
        
    return trj, success, time_steps

def compute_trajectory_single(i, pi, Lx, Ly, Lx0, Ly0, find_range, maxT, PObs_lim, rho0, act_hdl):
    """ Wrapper for the trajectory computation to use with multiprocessing.
    """
    _, result, trj_steps = single_traj_obs(pi, Lx, Ly, Lx0, Ly0, find_range, maxT, PObs_lim, rho0, act_hdl, progress_bar=False, save_traj=False)
    return result, trj_steps

def compute_trajectory_future(pi, Lx, Ly, Lx0, Ly0, find_range, maxT, PObs_lim, rho0, act_hdl, Nep=10):
    """ Compute the trajectory of the agent in parallel.
    """
    av_ret = 0
    Total_trj = []

    pbar = tqdm(total=Nep, desc='Success Rate: {:.3f}%'.format(0), position=0, leave=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_trajectory_single, i, pi, Lx, Ly, Lx0, Ly0, find_range, maxT, PObs_lim, rho0, act_hdl) for i in range(Nep)]

        for future in concurrent.futures.as_completed(futures):
            result, trj_steps = future.result()
            av_ret += result
            Total_trj.append(trj_steps)

            pbar.update(1)
            pbar.set_description('Success Rate: {:.3f}%'.format(100*av_ret / (len(Total_trj))))

    return av_ret, Total_trj
