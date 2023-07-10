#-------------------------------------------------------------
### Fake inversion 
#-------------------------------------------------------------
# How to run:
#
#  $ mpirun -np N python3 fake_inversion_sparse.py M
#
#      N = Number of cores that run in parallel
#      M = Multiplicative factor to increase the size of the system
#-------------------------------------------------------------

import numpy as np
import itertools as it
import sys
sys.path.append('../Comm/')
from utils import iterative_solve_eta as itsol
import matplotlib.pyplot as plt
from scipy.special import softmax

import time
#-------------------------------------------------------------
petsc4py_available = False
try :
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
    petsc4py_available = True

    OptDB = PETSc.Options()
    PETSc.Options().setValue('cuda_device', '0')

    mpi_rank = PETSc.COMM_WORLD.getRank()
    mpi_size = PETSc.COMM_WORLD.getSize()
except ImportError:
    mpi_rank = 0
    mpi_size = 1
    print('Petsc4py not available')
#-------------------------------------------------------------
scipy_sparse_available = False
try :
    import scipy.sparse as sparse
    scipy_sparse_available = True
except ImportError:
    print('Scipy sparse not available')
#-------------------------------------------------------------
cupy_scipy_available = False
try : 
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    import cupyx.scipy.sparse.linalg
    cupy_scipy_available = True
except ImportError:
    print('Cupy sparse not available')
#-------------------------------------------------------------
#-------------------------------------------------------------
## Utilities functions
#-------------------------------------------------------------
#-------------------------------------------------------------

clip = lambda x, l, u: l if x < l else u if x > u else x

def index_six_to_two(index,M,Ly,Lx):
    # Convert the index of an array of 6 dim to 2 dim
    # new_index_x = index[0] * Lx * Ly + index[1] * Lx + index[2]
    # new_index_y = index[3] * Lx * Ly + index[4] * Lx + index[5]
    # increse the performance of the new_index_x and new_index_y
    new_index_x = Lx * ( index[0] * Ly + index[1] ) + index[2]
    new_index_y = Lx * ( index[3] * Ly + index[4] ) + index[5]

    return (new_index_x,new_index_y)

def build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,action_size,p_a_mu_m_xy):

    # Get the maximum number of non zeros values per row for Tsm_sm
    count_non_zeros_in_row = np.ones(M*Ly*Lx,dtype=int) * action_size * M
    count_non_zeros_in_row[0] -= M
    count_non_zeros_in_row[-1] -= M

    # Set the amount of non zero values to zero depending on the distance to the center
    yxs = it.product(np.arange(Ly), np.arange(Lx))
    yx_founds = it.filterfalse(lambda x: (x[0]-Ly0)**2 + (x[1]-Lx0)**2 > find_range**2, yxs)

    for iy, ix in yx_founds:
        for im in range(M):
            index = np.zeros(6,dtype=int)
            index[0] = im
            index[1] = iy 
            index[2] = ix
            index_mat = index_six_to_two(index,M,Ly,Lx)
            count_non_zeros_in_row[index_mat[0]] = 0

    # Create the sparse matrix
    Tsm_sm_sp = sparse.lil_matrix((M*Ly*Lx, M*Ly*Lx), dtype=np.double)

    # Fill the sparse matrix
    for im_new in range(M):
        
        left = [ (im_new, iy, clip(ix-1, 0, Lx-1), im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        left_act = [ (0, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        
        for t, a in zip(left, left_act):
            indexes = index_six_to_two(t,M,Ly,Lx)
            if Tsm_sm_sp[indexes] == None :
                Tsm_sm_sp[indexes] =  p_a_mu_m_xy[a]
            else:
                Tsm_sm_sp[indexes] +=  p_a_mu_m_xy[a]
                
        right = [ (im_new, iy, clip(ix+1, 0, Lx-1), im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        right_act = [ (1, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]

        for t, a in zip(right, right_act):
            indexes = index_six_to_two(t,M,Ly,Lx)
            if Tsm_sm_sp[indexes] == None :
                Tsm_sm_sp[indexes] =  p_a_mu_m_xy[a]
            else:
                Tsm_sm_sp[indexes] +=  p_a_mu_m_xy[a]

        up = [ (im_new, clip(iy+1, 0, Ly-1), ix,  im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        up_act = [ (2, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]

        for t, a in zip(up, up_act):
            indexes = index_six_to_two(t,M,Ly,Lx)
            if Tsm_sm_sp[indexes] == None :
                Tsm_sm_sp[indexes] =  p_a_mu_m_xy[a]
            else:
                Tsm_sm_sp[indexes] +=  p_a_mu_m_xy[a]
        
        down = [ (im_new, clip(iy-1, 0, Ly-1), ix, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        down_act = [ (3, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]

        for t, a in zip(down, down_act):
            indexes = index_six_to_two(t,M,Ly,Lx)
            if Tsm_sm_sp[indexes] == None :
                Tsm_sm_sp[indexes] =  p_a_mu_m_xy[a]
            else:
                Tsm_sm_sp[indexes] +=  p_a_mu_m_xy[a]

    # Delete the rows and columns that have been set to zero depending on the distance to the initial position
    yxs = it.product(np.arange(Ly), np.arange(Lx))
    yx_founds = it.filterfalse(lambda x: (x[0]-Ly0)**2 + (x[1]-Lx0)**2 > find_range**2, yxs)

    for yx_found in yx_founds:
        for im in range(M):
            # rows to set zero
            indexes = np.zeros(6,dtype=int)
            indexes[0] = indexes[3] = im
            indexes[1] = indexes[4] = yx_found[0]
            indexes[2] = indexes[5] = yx_found[1]
            indexes = index_six_to_two(indexes,M,Ly,Lx)
            # rows to set zero
            Tsm_sm_sp[indexes[0],:] = 0
            # columns to set zero
            Tsm_sm_sp[:,indexes[1]] = 0

    return Tsm_sm_sp

def solution_test(eta,T,rho,gamma):
    mpi_rank = PETSc.COMM_WORLD.getRank()
    if mpi_rank == 0 :
        if sparse.issparse(T):
            identity = sparse.eye(T.shape[0])
        else:
            identity = np.eye(T.shape[0])
        sol = (identity - gamma * T) @ eta
        # plt.plot(sol-rho)
        # plt.show()
        return np.allclose(sol,rho)
    else :
        return False

#-------------------------------------------------------------
#-------------------------------------------------------------
## Functions to the $\eta$ with different libraries
#-------------------------------------------------------------
#-------------------------------------------------------------

def eta_petsc(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,ks_type,ps_type,verbose=False,device='cpu'):
    """
    Solve linear system for eta using PETSc
    """
    timing = False
    if timing :
        timer_step = np.zeros(6)
        timer_step[0] = time.process_time()

    if device == 'gpu':
        petsc4py.PETSc.Options().setValue('cuda', '1')
        mat_type = PETSc.Mat.Type.AIJCUSPARSE
    else:
        mat_type = PETSc.Mat.Type.AIJ

    mpi_rank = PETSc.COMM_WORLD.getRank()
    mpi_size = PETSc.COMM_WORLD.getSize()

    mat_size = M*Lx*Ly

    # Get the maximum number of non zeros values per row for Tsm_sm
    count_non_zeros_in_row = np.ones(mat_size,dtype=np.int32) * 4 * M
    count_non_zeros_in_row[0] -= M
    count_non_zeros_in_row[-1] -= M

    if timing :
        timer_step[1] = time.process_time()

    # Create PETSc matrix
    A = PETSc.Mat()
    A.create(comm=PETSc.COMM_WORLD)

    A.setSizes(  ( (PETSc.DECIDE, mat_size), (PETSc.DECIDE, mat_size) ) )
    A.setType(mat_type)
    A.setFromOptions()
    A.setPreallocationNNZ(count_non_zeros_in_row[mpi_rank*mat_size//mpi_size:(mpi_rank+1)*mat_size//mpi_size])
    # A.setUp()

    # Fill PETSc matrix
    non_zeros_Tsm = np.transpose(np.nonzero(Tsm_sm_matrix))
    
    rstart, rend = A.getOwnershipRange()
    local_non_zeros_Tsm = non_zeros_Tsm[non_zeros_Tsm[:,0]>=rstart]
    local_non_zeros_Tsm = local_non_zeros_Tsm[local_non_zeros_Tsm[:,0]<rend]

    for index in local_non_zeros_Tsm:
        # print("Rank: ", mpi_rank, "Index: ", index)
        A.setValue(index[0],index[1],Tsm_sm_matrix[index[0],index[1]])

    A.assemblyBegin()

    # eye matrix with PETSc
    ones = PETSc.Mat()
    ones.create(comm=PETSc.COMM_WORLD)
    ones.setSizes( ( (PETSc.DECIDE, mat_size), (PETSc.DECIDE, mat_size) ) )
    ones.setType(mat_type)    
    ones.setFromOptions()
    ones.setPreallocationNNZ(1)

    rstart, rend = ones.getOwnershipRange()
    for row in range(rstart, rend):
        ones.setValues(row,row,1.0)

    ones.assemblyBegin()
    A.assemblyEnd()
    ones.assemblyEnd()

    if timing :
        timer_step[2] = time.process_time()

    # matrix_A =  I - gamma * T
    A.aypx(-gamma,ones)

    if timing :
        timer_step[3] = time.process_time()

    # print('Matrix A assembled', A.size)
    # print(A.getInfo())
    # print("Options set", A.getOptionsPrefix())

    x, b =  A.createVecs()
    b.setValues(np.arange(rho0.shape[0],dtype=np.int32),rho0)

    b.assemble()
    x.assemble()
    
    # Start the solver
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    ksp.setType(ks_type)

    ksp.getPC().setType(ps_type)
    ksp.setTolerances(rtol=1e-10)

    ksp.setFromOptions()

    ksp.setUp()
    ksp.solve(b, x)

    A.destroy()
    ones.destroy()
    ksp.destroy()
    b.destroy()

    if timing :
        timer_step[4] = time.process_time()

    # Collect all the values across the processors 
    scatter, eta = PETSc.Scatter.toZero(x)
    scatter.scatter(x, eta, False, PETSc.Scatter.Mode.FORWARD)
    PETSc.COMM_WORLD.barrier()

    x.destroy()
    scatter.destroy()
    eta = eta.getArray()

    if not mpi_rank == 0 :
        eta = np.zeros(1)

    if timing :
        timer_step[5] = time.process_time()

    if timing :
        total_time = timer_step[-1] - timer_step[0]
        
        diff_time = np.zeros(timer_step.size)
        diff_time[0] = timer_step[0]
        
        for i in range(1,timer_step.size):
            diff_time[i] = timer_step[i] - timer_step[i-1]

        timer_step = diff_time / total_time
        print("Rank:",mpi_rank,
            "Get row nnz:","{:.2f}".format(timer_step[1]),
            "Fill A:",     "{:.2f}".format(timer_step[2]),
            "Compute A:",  "{:.2f}".format(timer_step[3]),
            "Solve:",      "{:.2f}".format(timer_step[4]),
            "Scatter:",    "{:.2f}".format(timer_step[5]),
            "Total:",      "{:.2f}".format(total_time))

    return eta

def eta_scipy_sparce_solve(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,verbose=False,device='cpu'):
    """
    Solve linear system using scipy sparce
    """
    mpi_rank = PETSc.COMM_WORLD.getRank()
    
    timing = False
    if timing :
        timer_step = np.zeros(4)
        timer_step[0] = time.process_time()


    if mpi_rank == 0:
        Tsm_sm_matrix_sparce = sparse.csr_matrix(Tsm_sm_matrix)
        
        if timing :
            timer_step[1] = time.process_time()

        Tsm_sm_matrix_sparce *= -gamma
        Tsm_sm_matrix_sparce += sparse.eye(M*Ly*Lx)

        if timing :
            timer_step[2] = time.process_time()

        new_eta = sparse.linalg.spsolve(Tsm_sm_matrix_sparce,rho0)

        if timing :
            timer_step[3] = time.process_time()

        if verbose :
            print("      Tsm_sm size: ", Tsm_sm_matrix_sparce.nnz, " Memory size: ", Tsm_sm_matrix_sparce.data.nbytes/1e6, " MB")
            print("        rho0 size: ", rho0.shape, " Memory size: ", rho0.nbytes/1e6, " MB")
            print("Total memory size: ", (Tsm_sm_matrix_sparce.data.nbytes + rho0.nbytes)/1e6, " MB")
        if timing :
            total_time = timer_step[-1] - timer_step[0]

            diff_time = np.zeros(timer_step.size)
            diff_time[0] = timer_step[0]

            for i in range(1,timer_step.size):
                diff_time[i] = timer_step[i] - timer_step[i-1]

            timer_step = diff_time / total_time
            print("Rank:",mpi_rank,
                "To csr:","{:.2f}".format(timer_step[1]),
                "Compute A:",  "{:.2f}".format(timer_step[2]),
                "Solve:",      "{:.2f}".format(timer_step[3]),
                "Total:",      "{:.2f}".format(total_time))

    else:
        new_eta = np.zeros(1)

    return new_eta

def eta_cupy_sparce_solve(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,verbose=False,device='gpu'):
    """
    Solve linear system using cupy scipy sparce
    """
    mpi_rank = PETSc.COMM_WORLD.getRank()
    
    timing = False
    if timing :
        timer_step = np.zeros(4)
        timer_step[0] = time.process_time()


    if mpi_rank == 0:
        
        if timing :
            timer_step[1] = time.process_time()
    
        Tsm_sm_matrix_cupy = cusparse.csr_matrix(Tsm_sm_matrix)

        to_invert = cusparse.eye(M*Ly*Lx) - gamma * Tsm_sm_matrix_cupy
        rho0_cupy = cp.asarray(rho0)

        if timing :
            timer_step[2] = time.process_time()

        new_eta_cupy = cusparse.linalg.spsolve(to_invert,rho0_cupy)
        new_eta = cp.asnumpy(new_eta_cupy)

        if timing :
            timer_step[3] = time.process_time()

        if verbose :
            print("      Tsm_sm size: ", Tsm_sm_matrix_cupy.nnz, " Memory size: ", Tsm_sm_matrix_cupy.data.nbytes/1e6, " MB")
            print("        rho0 size: ", rho0.shape, " Memory size: ", rho0.nbytes/1e6, " MB")
            print("Total memory size: ", (Tsm_sm_matrix_cupy.data.nbytes + rho0.nbytes)/1e6, " MB")
        if timing :
            total_time = timer_step[-1] - timer_step[0]

            diff_time = np.zeros(timer_step.size)
            diff_time[0] = timer_step[0]

            for i in range(1,timer_step.size):
                diff_time[i] = timer_step[i] - timer_step[i-1]

            timer_step = diff_time / total_time
            print("Rank:",mpi_rank,
                "To csr:","{:.2f}".format(timer_step[1]),
                "Compute A:",  "{:.2f}".format(timer_step[2]),
                "Solve:",      "{:.2f}".format(timer_step[3]),
                "Total:",      "{:.2f}".format(total_time))

    else:
        new_eta = np.zeros(1)

    return new_eta

#-------------------------------------------------------------
#-------------------------------------------------------------
## Solve the problem
#-------------------------------------------------------------
#-------------------------------------------------------------

def data_creation(M,O,Lx,Ly,a_size):

    comm = PETSc.COMM_WORLD.tompi4py()

    mpi_rank = PETSc.COMM_WORLD.getRank()
    mpi_size = PETSc.COMM_WORLD.getSize()

    PETSc.Sys.Print("Data creation...")
    if mpi_rank == 0:
        pi = softmax( np.random.rand(O,M,M*a_size), 2)
        PObs_lim = np.random.rand(O, M*Lx*Ly)
        PObs_lim[1] = 1-PObs_lim[0]
        rho0 = np.random.rand(M*Lx*Ly)
        rho0[Lx:] = 0
        rho0 /= np.sum(rho0)
        eta0 = np.random.rand(M*Lx*Ly)
    else:
        pi = None
        PObs_lim = None
        rho0 = None
        eta0 = None

    # Broadcast the data
    pi = comm.bcast(pi, root=0)
    PObs_lim = comm.bcast(PObs_lim, root=0)
    rho0 = comm.bcast(rho0, root=0)
    eta0 = comm.bcast(eta0, root=0)

    return pi, PObs_lim, rho0, eta0

#-------------------------------------------------------------

def solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=eta_scipy_sparce_solve, verbose=False, device='cpu'):
    """
    This function should solve the following:
    --> New_eta = (1 - gamma T)^-1 rho
    """

    timing = False
    if timing :
        timer_step = np.zeros(4)
        timer_step[0] = time.process_time()

    comm = PETSc.COMM_WORLD.tompi4py()
    mpi_rank = PETSc.COMM_WORLD.getRank()
    mpi_size = PETSc.COMM_WORLD.getSize()

    # O = Observations
    # M = Memory    
    # A = Actions
    O, M, A = pi.shape
    # L = Dimension of the environment
    L = Lx * Ly
    new_eta = np.zeros(M*L)
    
    # PY has size ~ 10^5
    PY = PObs_lim.reshape(O, M, Ly, Lx)
    # PY has size ~ 10^2
    PAMU = pi.reshape(O, M, M, A//M)
    
    p_a_mu_m_xy = np.einsum( 'omyx, omna -> anmyx', PY, PAMU)
    # T [ s'm'  sm] = sum_a, mu p(s'm' | sm a mu) p(a mu | sm)
    #               = sum_a, mu p(s'm' | sm a mu) sum_y f(y | s) pi(a mu | y m)
    if timing :
        timer_step[1] = time.process_time()

    # Tsm_sm has size ~ 10^5 x 10^5 or more
    # if mpi_rank == 0:
    #     Tsm_sm_matrix = build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,A//M,p_a_mu_m_xy)
    # else:
    #     Tsm_sm_matrix = None
    
    # Tsm_sm_matrix = comm.bcast(Tsm_sm_matrix, root=0)
    Tsm_sm_matrix = build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,A//M,p_a_mu_m_xy)

    if timing :
        timer_step[2] = time.process_time()

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
        # plt.imshow(Tsm_sm_matrix.toarray())
        # plt.show()
        timer_step = time.process_time()
        Tsm_sm_matrix = build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,A//M,p_a_mu_m_xy)
        print("Time to build the matrix: ", time.process_time() - timer_step)
        print("-"*50)


    test_ksp = False
    if func_eta == eta_petsc and test_ksp:
        # list_ksp_type = ['cg','gmres','bcgs','bcgsl','bcgsr','bicg','bicgsta','tfqmr','cr','minres','symmlq','lgmres']
        # list_pc_type = ['none','jacobi','sor','lu','cholesky','bjacobi','mg','eisenstat','ilu','icc','asm','gasm','ksp','preonly']

        # list_pc_type = ['none','jacobi','sor','lu','cholesky','bjacobi','mg','ilu','icc','asm','gasm','ksp','preonly']
        
        list_ksp_type = [PETSc.KSP.Type.BCGS,PETSc.KSP.Type.BCGSL,PETSc.KSP.Type.BICG,PETSc.KSP.Type.CG,PETSc.KSP.Type.CGLS,PETSc.KSP.Type.CGNE,PETSc.KSP.Type.CGS,PETSc.KSP.Type.CHEBYSHEV,PETSc.KSP.Type.CR,PETSc.KSP.Type.DGMRES,PETSc.KSP.Type.FBCGS,PETSc.KSP.Type.FBCGSR,PETSc.KSP.Type.FCG,PETSc.KSP.Type.FETIDP,PETSc.KSP.Type.FGMRES,PETSc.KSP.Type.GCR,PETSc.KSP.Type.GLTR,PETSc.KSP.Type.GMRES,PETSc.KSP.Type.GROPPCG,PETSc.KSP.Type.HPDDM,PETSc.KSP.Type.IBCGS,PETSc.KSP.Type.LCD,PETSc.KSP.Type.LGMRES,PETSc.KSP.Type.LSQR,PETSc.KSP.Type.MINRES,PETSc.KSP.Type.NASH,PETSc.KSP.Type.NONE,PETSc.KSP.Type.PGMRES,PETSc.KSP.Type.PIPEBCGS,PETSc.KSP.Type.PIPECG,PETSc.KSP.Type.PIPECG2,PETSc.KSP.Type.PIPECGRR,PETSc.KSP.Type.PIPECR,PETSc.KSP.Type.PIPEFCG,PETSc.KSP.Type.PIPEFGMRES,PETSc.KSP.Type.PIPEGCR,PETSc.KSP.Type.PIPELCG,PETSc.KSP.Type.PIPEPRCG,PETSc.KSP.Type.PREONLY,PETSc.KSP.Type.PYTHON,PETSc.KSP.Type.QCG,PETSc.KSP.Type.QMRCGS,PETSc.KSP.Type.RICHARDSON,PETSc.KSP.Type.STCG,PETSc.KSP.Type.SYMMLQ,PETSc.KSP.Type.TCQMR,PETSc.KSP.Type.TFQMR,PETSc.KSP.Type.TSIRM]

        list_pc_type = [PETSc.PC.Type.ASM,PETSc.PC.Type.BDDC,PETSc.PC.Type.BFBT,PETSc.PC.Type.BJACOBI,PETSc.PC.Type.CHOLESKY,PETSc.PC.Type.CHOWILUVIENNACL,PETSc.PC.Type.COMPOSITE,PETSc.PC.Type.CP,PETSc.PC.Type.DEFLATION,PETSc.PC.Type.EISENSTAT,PETSc.PC.Type.EXOTIC,PETSc.PC.Type.FIELDSPLIT,PETSc.PC.Type.GALERKIN,PETSc.PC.Type.GAMG,PETSc.PC.Type.GASM,PETSc.PC.Type.H2OPUS,PETSc.PC.Type.HMG,PETSc.PC.Type.HPDDM,PETSc.PC.Type.HYPRE,PETSc.PC.Type.ICC,PETSc.PC.Type.ILU,PETSc.PC.Type.JACOBI,PETSc.PC.Type.KSP,PETSc.PC.Type.LMVM,PETSc.PC.Type.LSC,PETSc.PC.Type.LU,PETSc.PC.Type.MAT,PETSc.PC.Type.MG,PETSc.PC.Type.ML,PETSc.PC.Type.NN,PETSc.PC.Type.NONE,PETSc.PC.Type.PARMS,PETSc.PC.Type.PATCH,PETSc.PC.Type.PBJACOBI,PETSc.PC.Type.PFMG,PETSc.PC.Type.PYTHON,PETSc.PC.Type.QR,PETSc.PC.Type.REDISTRIBUTE,PETSc.PC.Type.REDUNDANT,PETSc.PC.Type.ROWSCALINGVIENNACL,PETSc.PC.Type.SAVIENNACL,PETSc.PC.Type.SHELL,PETSc.PC.Type.SOR,PETSc.PC.Type.SPAI,PETSc.PC.Type.SVD,PETSc.PC.Type.SYSPFMG,PETSc.PC.Type.TELESCOPE,PETSc.PC.Type.TFS,PETSc.PC.Type.VPBJACOBI,PETSc.PC.Type.KACZMARZ]


        list_ksp_pc_type = [["      fbcgsr"," pbjacobi     "],
        ["      fbcgsr"," sor          "],
        ["      fbcgsr"," none         "],
        ["       fbcgs"," pbjacobi     "],
        ["       ibcgs"," pbjacobi     "],
        ["       ibcgs"," jacobi       "],
        ["       fbcgs"," none         "],
        ["       ibcgs"," none         "],
        ["       fbcgs"," jacobi       "],
        ["       fbcgs"," sor          "],
        ["         cgs"," jacobi       "],
        ["         cgs"," pbjacobi     "],
        ["        bicg"," jacobi       "],
        ["       tfqmr"," pbjacobi     "],
        ["       tfqmr"," jacobi       "],
        ["        bicg"," none         "],
        ["       tfqmr"," none         "],
        ["         cgs"," none         "],
        ["        cgls"," pbjacobi     "],
        ["        cgls"," mat          "],
        ["      fbcgsr"," jacobi       "],
        ["        cgls"," jacobi       "],
        ["      qmrcgs"," sor          "],
        ["        cgls"," kaczmarz     "],
        ["        cgls"," composite    "],
        ["        lsqr"," none         "],
        ["       bcgsl"," pbjacobi     "],
        ["       ibcgs"," bjacobi      "],
        ["        cgls"," shell        "],
        ["      fbcgsr"," kaczmarz     "],
        ["        bicg"," pbjacobi     "],
        ["       gmres"," sor          "],
        ["      qmrcgs"," pbjacobi     "],
        ["        cgls"," sor          "],
        ["      qmrcgs"," jacobi       "],
        ["         cgs"," bjacobi      "],
        ["       fbcgs"," bjacobi      "],
        ["        cgls"," none         "],
        ["         cgs"," kaczmarz     "],
        ["        cgne"," pbjacobi     "],
        ["      dgmres"," sor          "],
        ["        cgne"," none         "],
        ["       tfqmr"," sor          "],
        ["      lgmres"," sor          "],
        ["        cgne"," jacobi       "],
        ["       tfqmr"," bjacobi      "],
        ["       gmres"," jacobi       "],
        ["       gmres"," pbjacobi     "],
        ["      dgmres"," none         "],
        ["      qmrcgs"," none         "],
        ["        none"," redundant    "],
        ["       gmres"," none         "],
        ["      dgmres"," pbjacobi     "],
        ["       fbcgs"," kaczmarz     "],
        ["         gcr"," jacobi       "],
        ["        lsqr"," pbjacobi     "],
        ["        bicg"," bjacobi      "],
        ["  richardson"," redundant    "],
        ["        cgls"," ksp          "],
        ["      fgmres"," sor          "],
        ["        lsqr"," jacobi       "],
        ["        cgls"," bjacobi      "],
        ["      lgmres"," pbjacobi     "],
        ["        cgls"," lmvm         "],
        ["        nash"," redundant    "],
        ["       tfqmr"," kaczmarz     "],
        ["         gcr"," sor          "],
        ["      minres"," redundant    "],
        ["      fbcgsr"," redundant    "],
        ["         cgs"," redundant    "],
        ["         gcr"," none         "],
        ["      symmlq"," redundant    "],
        ["       fbcgs"," redundant    "],
        ["          cg"," redundant    "],
        ["          cr"," redundant    "],
        ["        cgne"," redundant    "],
        ["      dgmres"," jacobi       "],
        ["      lgmres"," redundant    "],
        ["       tfqmr"," redundant    "],
        ["      fbcgsr"," gasm         "],
        ["         gcr"," pbjacobi     "],
        ["    pipebcgs"," bjacobi      "],
        ["      lgmres"," jacobi       "],
        ["      lgmres"," none         "],
        ["       gmres"," redundant    "],
        ["        gltr"," redundant    "],
        ["      fgmres"," pbjacobi     "],
        ["      qmrcgs"," bjacobi      "],
        ["      fbcgsr"," asm          "],
        ["     groppcg"," redundant    "],
        ["      pgmres"," sor          "],
        ["         fcg"," redundant    "],
        ["       gmres"," bjacobi      "],
        ["      qmrcgs"," redundant    "],
        ["       fbcgs"," gasm         "],
        ["       tcqmr"," redundant    "],
        ["      fgmres"," none         "],
        ["      pgmres"," pbjacobi     "],
        ["     pipecg2"," redundant    "],
        ["      fbcgsr"," mat          "],
        ["      fgmres"," jacobi       "],
        ["       ibcgs"," gasm         "],
        ["      pgmres"," jacobi       "],
        ["      dgmres"," bjacobi      "],
        ["       fbcgs"," asm          "],
        ["      pgmres"," none         "],
        ["        stcg"," redundant    "],
        ["      pgmres"," redundant    "],
        ["        bicg"," redundant    "],
        ["      fgmres"," redundant    "],
        ["       ibcgs"," redundant    "],
        # ["        none"," tfs          "],
        ["      lgmres"," asm          "],
        ["        cgls"," gasm         "],
        ["         gcr"," redundant    "],
        ["       ibcgs"," asm          "],
        ["         gcr"," bjacobi      "],
        ["       tfqmr"," gasm         "],
        ["       gmres"," asm          "],
        ["         cgs"," gasm         "],
        ["         cgs"," asm          "],
        ["      lgmres"," bjacobi      "],
        ["       tfqmr"," asm          "],
        ["        bicg"," asm          "],
        # ["        stcg"," tfs          "],
        # ["      fbcgsr"," tfs          "],
        # ["      minres"," tfs          "],
        ["       fbcgs"," mat          "],
        ["         lcd"," redundant    "],
        ["      qmrcgs"," kaczmarz     "],
        # ["      fgmres"," tfs          "],
        ["       ibcgs"," mat          "],
        # ["       gmres"," tfs          "],
        ["        cgne"," bjacobi      "],
        ["      fgmres"," ksp          "],
        # ["        gltr"," tfs          "],
        ["      dgmres"," redundant    "],
        # ["          cg"," tfs          "],
        ["       tcqmr"," pbjacobi     "],
        ["       fbcgs"," ksp          "],
        ["      dgmres"," asm          "],
        # ["          cr"," tfs          "],
        ["       tcqmr"," sor          "],
        # ["      qmrcgs"," tfs          "],
        # ["       fbcgs"," tfs          "],
        # ["  richardson"," tfs          "],
        # ["      dgmres"," tfs          "],
        # ["       tfqmr"," tfs          "],
        ["      dgmres"," kaczmarz     "],
        ["         cgs"," mat          "],
        # ["      lgmres"," tfs          "],
        ["      pgmres"," bjacobi      "],
        ["         gcr"," asm          "],
        ["        bicg"," mat          "],
        # ["       tcqmr"," tfs          "],
        ["       gmres"," kaczmarz     "],
        ["       gmres"," gasm         "],
        ["      qmrcgs"," asm          "],
        # ["         fcg"," tfs          "],
        ["      qmrcgs"," gasm         "],
        ["        cgls"," asm          "],
        ["        cgls"," telescope    "],
        ["      lgmres"," kaczmarz     "],
        ["      fgmres"," bjacobi      "],
        ["      dgmres"," gasm         "],
        # ["      pgmres"," tfs          "],
        # ["     pipecg2"," tfs          "],
        ["         gcr"," kaczmarz     "],
        ["      pgmres"," asm          "],
        # ["     groppcg"," tfs          "],
        ["      fbcgsr"," ksp          "],
        ["       tfqmr"," mat          "],
        ["        cgne"," asm          "],
        ["       tcqmr"," none         "],
        ["       tcqmr"," jacobi       "],
        ["        bicg"," gasm         "],
        ["      fbcgsr"," telescope    "],
        # ["         gcr"," tfs          "],
        ["        cgls"," redistribute "],
        ["      fgmres"," asm          "],
        ["      lgmres"," gasm         "],
        ["        cgls"," redundant    "],
        # ["         cgs"," tfs          "],
        ["    pipebcgs"," asm          "],
        ["      dgmres"," ksp          "],
        ["      fgmres"," kaczmarz     "],
        ["         gcr"," gasm         "],
        ["      lgmres"," ksp          "],
        ["         gcr"," ksp          "],
        ["       fbcgs"," telescope    "],
        # ["      symmlq"," tfs          "],
        ["          cr"," ksp          "],
        ["          cg"," ksp          "],
        ["       gmres"," ksp          "],
        ["      fgmres"," gasm         "],
        ["        gltr"," ksp          "],
        # ["         lcd"," tfs          "],
        ["        stcg"," ksp          "],
        ["       tcqmr"," bjacobi      "],
        ["      qmrcgs"," ksp          "],
        ["  richardson"," ksp          "],
        ["        nash"," ksp          "],
        ["        cgne"," gasm         "],
        ["      symmlq"," ksp          "],
        ["         cgs"," ksp          "],
        ["        bcgs"," sor          "],
        ["      minres"," ksp          "],
        ["        bcgs"," none         "],
        ["         cgs"," sor          "],
        # ["        cgls"," tfs          "],
        ["        bcgs"," jacobi       "],
        ["       tfqmr"," telescope    "],
        ["      pgmres"," gasm         "],
        ["         lcd"," ksp          "],
        ["       gmres"," telescope    "],
        ["       tcqmr"," asm          "],
        ["       bcgsl"," telescope    "],
        # ["     groppcg"," ksp          "],
        ["         lcd"," asm          "],
        ["         gcr"," telescope    "],
        ["      fbcgsr"," redistribute "],
        ["       fbcgs"," redistribute "],
        ["        bcgs"," pbjacobi     "],
        ["      pgmres"," kaczmarz     "],
        ["         cgs"," telescope    "],
        ["      qmrcgs"," mat          "],
        ["       ibcgs"," ksp          "],
        ["      fgmres"," redistribute "],
        ["      dgmres"," telescope    "],
        ["      fbcgsr"," bjacobi      "],
        # ["        nash"," tfs          "],
        ["      qmrcgs"," telescope    "],
        ["      fgmres"," telescope    "],
        ["      lgmres"," telescope    "],
        ["     preonly"," redundant    "],
        ["         cgs"," redistribute "],
        ["       bcgsl"," ksp          "],
        ["         lcd"," sor          "],
        ["      pgmres"," telescope    "],
        ["          cr"," asm          "],
        ["         gcr"," redistribute "],
        ["      qmrcgs"," redistribute "],
        ["       gmres"," redistribute "],
        ["          cg"," redistribute "],
        ["          cr"," redistribute "],
        ["  richardson"," redistribute "],
        ["      minres"," redistribute "],
        # ["     preonly"," tfs          "],
        ["        nash"," redistribute "],
        # ["     groppcg"," redistribute "],
        ["      lgmres"," redistribute "],
        ["        stcg"," redistribute "],
        ["        bcgs"," kaczmarz     "],
        ["       tfqmr"," ksp          "],
        ["      symmlq"," redistribute "],
        ["      pgmres"," ksp          "],
        ["      dgmres"," redistribute "],
        ["        gltr"," redistribute "],
        ["         lcd"," none         "],
        ["         lcd"," pbjacobi     "],
        ["       tcqmr"," gasm         "],
        ["       bcgsl"," bjacobi      "],
        ["         lcd"," jacobi       "],
        ["        bicg"," ksp          "],
        ["         lcd"," bjacobi      "],
        ["         lcd"," redistribute "],
        ["        bcgs"," redundant    "],
        ["        bcgs"," bjacobi      "],
        ["       tcqmr"," kaczmarz     "],
        ["       tcqmr"," telescope    "],
        ["        bcgs"," gasm         "],
        ["        cgne"," mat          "],
        ["         lcd"," gasm         "],
        ["       bcgsl"," redistribute "],
        # ["        bcgs"," tfs          "],
        ["       tcqmr"," ksp          "],
        ["        cgne"," ksp          "],
        ["       tfqmr"," redistribute "],
        ["      pgmres"," redistribute "],
        ["        bcgs"," mat          "],
        ["        bcgs"," ksp          "],
        ["         lcd"," telescope    "],
        ["        bcgs"," telescope    "],
        ["       tcqmr"," redistribute "],
        ["         lcd"," kaczmarz     "],
        ["        bcgs"," redistribute "],
        ["        bcgs"," asm          "],
        ["  richardson"," asm          "],
        ["       tcqmr"," mat          "],
        ["      lgmres"," mat          "],
        # ["      fgmres"," svd          "],
        # ["        gltr"," svd          "],
        # ["       gmres"," svd          "],
        # ["  richardson"," svd          "],
        # ["       fbcgs"," svd          "],
        # ["      fbcgsr"," svd          "],
        # ["         cgs"," svd          "],
        # ["          cg"," svd          "],
        # ["      lgmres"," svd          "],
        # ["      symmlq"," svd          "],
        # ["        nash"," svd          "],
        # ["      dgmres"," svd          "],
        # ["      minres"," svd          "],
        # ["        bicg"," svd          "],
        # ["         gcr"," svd          "],
        # ["        none"," svd          "],
        # ["         lcd"," svd          "],
        # ["       tfqmr"," svd          "],
        # ["       ibcgs"," svd          "],
        # ["      pgmres"," svd          "],
        # ["      qmrcgs"," svd          "],
        # ["         fcg"," svd          "],
        # ["        cgne"," svd          "],
        # ["        stcg"," svd          "],
        # ["     groppcg"," svd          "],
        # ["       tcqmr"," svd          "],
        # ["     pipecg2"," svd          "],
        # ["        cgls"," svd          "],
        ["  richardson"," bjacobi      "],
        # ["          cr"," svd          "],
        ["  richardson"," pbjacobi     "],
        ["  richardson"," jacobi       "],
        ["  richardson"," gasm         "],
        ["  richardson"," none         "],
        ["     pipelcg"," sor          "],
        ["     pipelcg"," jacobi       "],
        ["     pipelcg"," asm          "],
        # ["     preonly"," svd          "],
        ["     pipelcg"," pbjacobi     "],
        # ["        bcgs"," svd          "],
        ["     pipelcg"," none         "],
        ["       gmres"," mat          "],
        ["      dgmres"," mat          "],
        ["         fcg"," sor          "],
        ["         fcg"," none         "],
        ["         fcg"," pbjacobi     "],
        ["         fcg"," jacobi       "],
        ["         fcg"," asm          "],
        ["      fgmres"," mat          "],
        ["      pgmres"," mat          "],
        ["         lcd"," mat          "],
        ["         fcg"," bjacobi      "],
        ["         gcr"," mat          "],
        ["         fcg"," gasm         "],
        ["     pipelcg"," bjacobi      "],
        ["  richardson"," kaczmarz     "],
        ["         fcg"," kaczmarz     "],
        ["         fcg"," telescope    "],
        ["     pipelcg"," gasm         "],
        ["     pipelcg"," telescope    "],
        ["         fcg"," ksp          "],
        ["         fcg"," redistribute "],
        ["  richardson"," sor          "],
        ["         fcg"," mat          "],
        ["          cg"," asm          "],
        ["     groppcg"," asm          "]]


        class ksp_pc_monitor():
            def __init__(self,mpi_rank,ksp_type,pc_type,runtimes):
                self.rank = mpi_rank
                self.ksp_type = ksp_type
                self.pc_type = pc_type
                self.time_step = np.zeros(runtimes)
                self.succeed = False
                self.solution_trust = True

        benchmark_ksp_pc = []

        runtimes = 5
        # for ksp_type in list_ksp_type:
        #     for pc_type in list_pc_type:
        for ksp_type,pc_type in list_ksp_pc_type:
            ksp_type = ksp_type.strip()
            pc_type = pc_type.strip()
            if True :
                PETSc.Sys.Print("{:>12} {:<12}".format(ksp_type,pc_type),end=" ")

                benchmark_ksp_pc.append(ksp_pc_monitor(mpi_rank,ksp_type,pc_type,runtimes))

                for i in range(runtimes):

                    if ksp_type == PETSc.KSP.Type.PIPELCG and (pc_type == PETSc.PC.Type.REDUNDANT or pc_type == PETSc.PC.Type.KSP or pc_type == PETSc.PC.Type.REDISTRIBUTE or pc_type == PETSc.PC.Type.TFS or pc_type == PETSc.PC.Type.SVD or pc_type == PETSc.PC.Type.KACZMARZ) :
                        PETSc.Sys.Print("*_", end=" ", flush=True)
                        benchmark_ksp_pc[-1].succeed = False
                        continue
                    
                    PETSc.COMM_WORLD.barrier()
                    t = time.process_time()

                    try :
                        new_eta = func_eta(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,ksp_type,pc_type,device=device)
                        benchmark_ksp_pc[-1].succeed = True
                    except :
                        benchmark_ksp_pc[-1].succeed = False

                    benchmark_ksp_pc[-1].time_step[i] = time.process_time() - t

                    if benchmark_ksp_pc[-1].succeed :
                        benchmark_ksp_pc[-1].solution_trust *= solution_test(new_eta,Tsm_sm_matrix,rho0,gamma)

                    PETSc.Sys.Print("*_", end=" ", flush=True)
                if benchmark_ksp_pc[-1].succeed :
                    PETSc.Sys.Print("Average time: {:.4f} s".format(np.mean(benchmark_ksp_pc[-1].time_step)),"std dev: {:.4f} s".format(np.std(benchmark_ksp_pc[-1].time_step)),"| Trust in solution:",benchmark_ksp_pc[-1].solution_trust)
                else:
                    PETSc.Sys.Print("Failed")

        PETSc.Sys.Print("-"*50)
        PETSc.Sys.Print("Summary:")
        benchmark_ksp_pc.sort(key=lambda x: np.mean(x.time_step))
        for ksp_pc in benchmark_ksp_pc:
            if ksp_pc.succeed and ksp_pc.solution_trust :
                PETSc.Sys.Print("{:>12} {:<12}".format(ksp_pc.ksp_type,ksp_pc.pc_type),"Average time: {:.4f} s".format(np.mean(ksp_pc.time_step)),"std dev: {:.4f} s".format(np.std(ksp_pc.time_step)))

    elif func_eta == eta_petsc and not test_ksp:
        new_eta = func_eta(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,'bcgs','jacobi',device=device)
    else :
        if mpi_rank == 0 :
            new_eta = func_eta(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,device=device)
    
    if timing :
        timer_step[3] = time.process_time()

    if timing :
        total_time = timer_step[-1] - timer_step[0]
        
        diff_time = np.zeros(timer_step.size)
        diff_time[0] = timer_step[0]
        
        for i in range(1,timer_step.size):
            diff_time[i] = timer_step[i] - timer_step[i-1]
        timer_step = diff_time / total_time
        
        if func_eta is not eta_petsc and mpi_rank < 2 :
            print()
            print("Rank:",mpi_rank,
                "Initial data:","{:.2f}".format(timer_step[1]),
                "Tsm_sm:",  "{:.2f}".format(timer_step[2]),
                "Solve:",      "{:.2f}".format(timer_step[3]),
                "Total:",      "{:.2f}".format(total_time))
        
        if func_eta is eta_petsc :
            print()
            print("Rank:",mpi_rank,
                "Initial data:","{:.2f}".format(timer_step[1]),
                "Tsm_sm:",  "{:.2f}".format(timer_step[2]),
                "Solve:",      "{:.2f}".format(timer_step[3]),
                "Total:",      "{:.2f}".format(total_time))

    return new_eta, Tsm_sm_matrix

#-------------------------------------------------------------
#-------------------------------------------------------------
## TEST
#-------------------------------------------------------------
#-------------------------------------------------------------

if __name__ == "__main__":

    # Initial message
    if mpi_rank == 0:
        print('Running fake inversion')
        print("modules loaded")
        if petsc4py_available:
            print('Petsc4py available')
        if scipy_sparse_available:
            print('Scipy sparse available')
        if cupy_scipy_available:
            print('Cupy sparse available')


    # Parameters
    dim_factor = 1 if len(sys.argv) < 2 else int(sys.argv[1])

    M = 2 # O(10)  Memories
    O = 2 # 2/3  Observations
    Lx = 5 * dim_factor # O(100)-O(1000)    Box size in x
    Ly = 7 * dim_factor # O(100)-O(1000)    Box size in y
    a_size = 4  #  Number of actions
    Lx0 = 5 # Initial position of the agent in x
    Ly0 = 7 # Initial position of the agent in y

    gamma = 0.99
    find_range = 2.1
    tol = 1e-8

    ## Benchmarking of the solvers
    solvers_methods = [
        ["Scipy Sparse",'cpu',eta_scipy_sparce_solve ],
        ["Cupy Sparse", 'gpu',eta_cupy_sparce_solve  ],
        # ["Petsc GPU",   'gpu',eta_petsc              ],
        ["Petsc CPU",   'cpu',eta_petsc              ]
        ]

    PETSc.COMM_WORLD.barrier()
    PETSc.Sys.Print("-"*50)
    PETSc.Sys.Print("System parameters:")
    PETSc.Sys.Print("M:",M,"   O:",O,"   Lx:",Lx,"   Ly:",Ly,"   # actions:",a_size,"   dim_factor:",dim_factor)

    # np.random.seed(33+33)
    # Create the data
    pi, PObs_lim, rho0, eta0 = data_creation(M,O,Lx,Ly,a_size)

    inv_sol, T = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range, verbose=True,func_eta=eta_scipy_sparce_solve)

    if mpi_rank == 0 :
        iter_sol = itsol(pi, PObs_lim, gamma, rho0, eta0, tol, Lx, Ly, Lx0, Ly0, find_range)
    else:
        iter_sol = np.zeros(1)

    T = None
    
    runtimes = 2
    PETSc.Sys.Print("-"*50)
    PETSc.Sys.Print("Benchmarking of the solvers")
    PETSc.Sys.Print("Number of runs:",runtimes)

    name_solvers = [x[0] for x in solvers_methods]
    device_selector = [x[1] for x in solvers_methods]
    solvers = [x[2] for x in solvers_methods]    

    for solver, name_solver,device_sel in zip(solvers, name_solvers,device_selector):
        
        PETSc.Sys.Print("Method: {:>12}".format(name_solver), end=" ")

        total_time = 0.0
        times = np.zeros(runtimes)
        
        for i in range(runtimes):
            PETSc.COMM_WORLD.barrier()
            t = time.process_time()

            out_01, out_02 = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=solver, verbose=False,device=device_sel)
            out_01 = None
            out_02 = None

            times[i] = time.process_time() - t
            PETSc.Sys.Print("*_", end=" ", flush=True)
        
        PETSc.Sys.Print("Average time: {:.4f} s".format(np.mean(times)),"std dev: {:.4f} s".format(np.std(times)))

    PETSc.Sys.Print("\nIterative Solution", end=" ")

    total_time = 0.0
    times = np.zeros(runtimes)

    if mpi_rank == 0:   
        for i in range(runtimes):
            t = time.process_time()
            _ = itsol(pi, PObs_lim, gamma, rho0, eta0, tol, Lx, Ly, Lx0, Ly0, find_range)
            times[i] = time.process_time() - t
        
        print("Average time: {:.4f} s,".format(np.mean(times)),"std dev: {:.4f} s".format(np.std(times)))


    PETSc.Sys.Print("-"*50)
    PETSc.Sys.Print("Diff between iterative solution and solvers")

    memory = 0
    PETSc.Sys.Print("Shape     iter_sol:", iter_sol.shape[0],
          "max: {:.4f} min: {:.4f}".format(np.max(iter_sol), np.min(iter_sol)))

    for solver, name_solver, device_sel in zip(solvers, name_solvers,device_selector):
        PETSc.COMM_WORLD.barrier()
        inv_sol, T = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=solver, verbose=False,device=device_sel)
        PETSc.COMM_WORLD.barrier()
        if mpi_rank == 0:
            PETSc.Sys.Print("Shape {:>12}: {:} max: {:.4f} min: {:.4f}".format(name_solver, inv_sol.shape[0], np.max(inv_sol), np.min(inv_sol)), end=" ", flush=True)  
            PETSc.Sys.Print("|| Passing test of solver:",solution_test(inv_sol,T,rho0,gamma), end=" ", flush=True)
            PETSc.Sys.Print("|| Diff from iter_sol: {:.8f}".format(np.sum(np.abs(iter_sol-inv_sol))))

    PETSc.COMM_WORLD.barrier()
    PETSc.Sys.Print("-"*50)
    PETSc.Sys.Print("Diff between solvers")        
    PETSc.Sys.Print("{:>12}|".format(""), end=" ")
    [PETSc.Sys.Print("{:>12}".format(i), end=" ") for i in name_solvers]
    PETSc.Sys.Print("{:>12}".format("|"))
    for i in range(len(solvers)):
        inv_sol_i, T_i = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=solvers[i],verbose=False,device=device_selector[i])
        PETSc.Sys.Print("{:>12}|".format(name_solvers[i]), end=" ")
        [PETSc.Sys.Print("{:12}".format(""), end=" ") for _ in range(i)]
        PETSc.Sys.Print("{:12.5e}".format(0.0), end=" ", flush=True)
        for j in range(i+1,len(solvers)):
            inv_sol_j, T_j = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=solvers[j],verbose=False,device=device_selector[j])
            if mpi_rank == 0:
                PETSc.Sys.Print("{:12.5e}".format(np.sum(np.abs(inv_sol_i-inv_sol_j))), end=" ", flush=True) 
        PETSc.Sys.Print("{:>12}".format("|"))



