### Fake inversion 

import numpy as np
import itertools as it
import sys
sys.path.append('../Comm/')
from utils import iterative_solve_eta as itsol
import matplotlib.pyplot as plt
from scipy.special import softmax

import time

print('Running fake inversion')
print("modules loaded")

torch_available = False
try :
    import torch as torch
    dtype = torch.double
    print('Pytorch available')
    if torch.cuda.is_available():
        device_torch = torch.device("cuda:1")
        print('Device: GPU')
    else:
        device_torch = torch.device("cpu")
        print('Device: CPU')
    torch_available = True
except ImportError:
    print('Pytorch not available')

petsc4py_available = False
try :
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
    petsc4py_available = True
    print('Petsc4py available')
    OptDB = PETSc.Options()
    PETSc.Options().setValue('cuda_device', '1 ')
except ImportError:
    print('Petsc4py not available')

scipy_sparse_available = False
try :
    import scipy.sparse as sparse
    scipy_sparse_available = True
    print('Scipy sparse available')
except ImportError:
    print('Scipy sparse not available')

## Utilities functions

clip = lambda x, l, u: l if x < l else u if x > u else x

def index_six_to_two(index,M,Ly,Lx):
    # Convert the index of an array of 6 dim to 2 dim
    new_index_x = index[0] * Lx * Ly + index[1] * Lx + index[2]
    new_index_y = index[3] * Lx * Ly + index[4] * Lx + index[5]

    return (new_index_x,new_index_y)

def build_Tsm_sm_dense(M,Lx,Ly,Lx0,Ly0,find_range,action_size,p_a_mu_m_xy):
    # Action Order
    # left, right, up, down
    
    # tuples to be populated
    #      T indices : "left / right / up / down"
    #      (m', y', x', m, y, x)
    #      pi indices : "left_act / ... "
    #      (m', a, m, y, x)

    # Create the matrix
    Tsm_sm = np.zeros( (M, Ly, Lx, M, Ly, Lx) )

    # Fill the matrix
    for im_new in range(M):
        
        left = [ (im_new, iy, clip(ix-1, 0, Lx-1), im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        left_act = [ (0, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        
        for l, la in zip(left, left_act):
            Tsm_sm[l] += p_a_mu_m_xy[la]
        
        right = [ (im_new, iy, clip(ix+1, 0, Lx-1), im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        right_act = [ (1, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        for r, ra in zip(right, right_act):
            Tsm_sm[r] += p_a_mu_m_xy[ra]
    
        up = [ (im_new, clip(iy+1, 0, Ly-1), ix,  im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        up_act = [ (2, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        for u, ua in zip(up, up_act):
            Tsm_sm[u] += p_a_mu_m_xy[ua]
        
        down = [ (im_new, clip(iy-1, 0, Ly-1), ix, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        down_act = [ (3, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        for d, da in zip(down, down_act):
            Tsm_sm[d] += p_a_mu_m_xy[da]
    
    # Delete the rows and columns that have been set to zero depending on the distance to the initial position
    yxs = it.product(np.arange(Ly), np.arange(Lx))
    yx_founds = it.filterfalse(lambda x: (x[0]-Ly0)**2 + (x[1]-Lx0)**2 > find_range**2, yxs)
    
    for yx_found in yx_founds:
        # all transitions starting from the source do not go anywhere
        Tsm_sm[:,yx_found[0],yx_found[1],:,:,:] = 0
        # all transitions ending in the source stop the episode
        Tsm_sm[:,:,:,:,yx_found[0],yx_found[1]] = 0
    
    Tsm_sm_matrix = np.reshape(Tsm_sm, (M*Ly*Lx, M*Ly*Lx))

    return Tsm_sm_matrix

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
    if sparse.issparse(T):
        identity = sparse.eye(T.shape[0])
    else:
        identity = np.eye(T.shape[0])
    sol = (identity - gamma * T) @ eta
    # plt.plot(sol-rho)
    # plt.show()
    return np.allclose(sol,rho)

## Functions to the $\eta$ with different libraries

def eta_petsc(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,ks_type,ps_type,verbose=False,device='gpu'):
    """
    Solve linear system for eta using PETSc
    """
    if device == 'gpu':
        petsc4py.PETSc.Options().setValue('cuda', '1')
        mat_type = PETSc.Mat.Type.AIJCUSPARSE
    else:
        mat_type = PETSc.Mat.Type.AIJ

    # Get non-zero indices from Tsm_sm_matrix per row 
    non_zeros_in_row = [np.nonzero(Tsm_sm_matrix[i,:])[0] for i in range(Tsm_sm_matrix.shape[0])]
    count_non_zeros_in_row = np.array([len(non_zeros_in_row[i]) for i in range(len(non_zeros_in_row))],dtype=np.int32)

    # print('Number of non-zero elements per row: ', np.sum(count_non_zeros_in_row))

    mat_size = M*Lx*Ly

    # Create PETSc matrix
    A = PETSc.Mat()
    A.create(comm=PETSc.COMM_WORLD)

    mpi_rank = PETSc.COMM_WORLD.getRank()
    mpi_size = PETSc.COMM_WORLD.getSize()
    comm = PETSc.COMM_WORLD

    A.setSizes( (mat_size , mat_size) )
    A.setType(mat_type)
    A.setPreallocationNNZ(count_non_zeros_in_row[mpi_rank*mat_size//mpi_size:(mpi_rank+1)*mat_size//mpi_size])
    # A.createAIJ([mat_size,mat_size], nnz=count_non_zeros_in_row)
    # 
    # Fill PETSc matrix
    non_zeros_Tsm = np.transpose(np.nonzero(Tsm_sm_matrix))
    # print("non_zeros_Tsm.shape: ", non_zeros_Tsm.shape)
    for index in non_zeros_Tsm:
        A.setValues(index[0],index[1],Tsm_sm_matrix[index[0],index[1]])
    
    # def index_to_grid(r):
    #     """Convert a row number into a grid point."""
    #     return (r // mat_size, r % mat_size)


    # rstart, rend = A.getOwnershipRange()
    # for index in non_zeros_Tsm:
    #     if rstart <= index[0] <= rend:
    #         A.setValues(index[0],index[1],Tsm_sm_matrix[index[0],index[1]])

    A.assemblyBegin()
    A.assemblyEnd()


    # print("Lllogo aqui")

    # eye matrix with PETSc
    ones = PETSc.Mat()
    ones.create(comm=PETSc.COMM_WORLD)
    ones.setSizes( (mat_size , mat_size) )
    ones.setType(mat_type)    
    ones.setPreallocationNNZ(1)

    for i in range(mat_size):
        ones.setValues(i,i,1.0)

    ones.assemblyBegin()
    ones.assemblyEnd()

    # matrix_A =  I - gamma * T
    A.aypx(-gamma,ones)

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

    # ksp.setConvergenceHistory()
    ksp.setFromOptions()
    # ksp.setInitialGuessNonzero(True)

    ksp.setUp()
    ksp.solve(b, x)

    # residuals = ksp.getConvergenceHistory()
    # plt.semilogy(residuals)
    # plt.show()

    A.destroy()
    ones.destroy()
    ksp.destroy()
    b.destroy()

    eta = x.getArray()
    x.destroy()

    return eta

def eta_numpy(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,verbose=False,device='cpu'):
    """
    Invert a matrix using numpy
    """


    Tsm_sm_matrix_local = Tsm_sm_matrix.copy()

    np.multiply(Tsm_sm_matrix_local,-gamma,out=Tsm_sm_matrix_local)
    for i in range(M*Ly*Lx):
        Tsm_sm_matrix_local[i,i] += 1.0

    inverted = np.linalg.inv(Tsm_sm_matrix_local)
    new_eta = inverted @ rho0

    if verbose :
        print("      Tsm_sm size: ", Tsm_sm_matrix.shape, " Memory size: ", Tsm_sm_matrix.nbytes/1e6, " MB")
        print("        rho0 size: ", rho0.shape, " Memory size: ", rho0.nbytes/1e6, " MB")
        print("    inverted size: ", inverted.shape, " Memory size: ", inverted.nbytes/1e6, " MB")
        print("Total memory size: ", (Tsm_sm_matrix.nbytes + rho0.nbytes + inverted.nbytes)/1e6, " MB")

    Tsm_sm_matrix_local = None
    return new_eta

def eta_numpy_sol(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,verbose=False,device='cpu'):
    """
    Solve linear system
    """
    Tsm_sm_matrix_local = Tsm_sm_matrix.copy()

    np.multiply(Tsm_sm_matrix_local,-gamma,out=Tsm_sm_matrix_local)
    for i in range(M*Ly*Lx):
        Tsm_sm_matrix_local[i,i] += 1.0

    new_eta = np.linalg.solve(Tsm_sm_matrix_local,rho0)

    if verbose :
        print("      Tsm_sm size: ", Tsm_sm_matrix.shape, " Memory size: ", Tsm_sm_matrix.nbytes/1e6, " MB")
        print("        rho0 size: ", rho0.shape, " Memory size: ", rho0.nbytes/1e6, " MB")
        print("Total memory size: ", (Tsm_sm_matrix.nbytes + rho0.nbytes)/1e6, " MB")

    Tsm_sm_matrix_local = None
    return new_eta

def eta_torch(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,verbose=False,device='gpu'):
    """
    Invert a matrix using torch
    """
    if device == 'cpu':
        device_torch = torch.device('cpu')
    else:
        device_torch = torch.device('cuda:1')

    try :
        Tsm_sm_matrix_local = Tsm_sm_matrix.copy()

        np.multiply(Tsm_sm_matrix_local,-gamma,out=Tsm_sm_matrix_local)
        for i in range(M*Ly*Lx):
            Tsm_sm_matrix_local[i,i] += 1.0

        Tsm_sm_matrix_torch = torch.from_numpy(Tsm_sm_matrix_local).type(dtype).to(device_torch)

        rho0_torch = torch.from_numpy(rho0).type(dtype).to(device_torch)

        inverted = torch.torch.linalg.inv(Tsm_sm_matrix_torch)
        new_eta = inverted @ rho0_torch

        new_eta = new_eta.cpu().numpy()
        Tsm_sm_matrix_local = None

        if torch.cuda.is_available():
            Tsm_sm_matrix_torch = None
            rho0_torch = None
            inverted = None
            torch.cuda.empty_cache()
    except:
        print("Error inverting matrix with torch")
        Tsm_sm_matrix_torch = None
        rho0_torch = None
        inverted = None
        torch.cuda.empty_cache()
        new_eta = np.zeros(M*Ly*Lx)

    return new_eta

def eta_torch_sol(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,verbose=False,device='gpu'):
    """
    Solve linear system using torch
    """
    if device == 'cpu':
        device_torch = torch.device('cpu')
    else:
        device_torch = torch.device('cuda:1')
    try :
        Tsm_sm_matrix_local = Tsm_sm_matrix.copy()

        np.multiply(Tsm_sm_matrix_local,-gamma,out=Tsm_sm_matrix_local)
        for i in range(M*Ly*Lx):
            Tsm_sm_matrix_local[i,i] += 1.0

        Tsm_sm_matrix_torch = torch.from_numpy(Tsm_sm_matrix_local).type(dtype).to(device_torch)

        rho0_torch = torch.from_numpy(rho0).type(dtype).to(device_torch)

        new_eta = torch.linalg.solve(Tsm_sm_matrix_torch,rho0_torch)

        new_eta = new_eta.cpu().numpy()
        Tsm_sm_matrix_local = None 

        if torch.cuda.is_available():
            Tsm_sm_matrix_torch = None
            rho0_torch = None
            torch.cuda.empty_cache()
    except:
        print("Error inverting matrix with torch")
        Tsm_sm_matrix_torch = None
        rho0_torch = None
        torch.cuda.empty_cache()
        new_eta = np.zeros(M*Ly*Lx)


    return new_eta

def eta_scipy_sparce_solve(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,verbose=False,device='cpu'):
    """
    Solve linear system using scipy sparce
    """
    Tsm_sm_matrix_sparce = sparse.csr_matrix(Tsm_sm_matrix)
    Tsm_sm_matrix_sparce *= -gamma
    Tsm_sm_matrix_sparce += sparse.eye(M*Ly*Lx)
    new_eta = sparse.linalg.spsolve(Tsm_sm_matrix_sparce,rho0)

    if verbose :
        print("      Tsm_sm size: ", Tsm_sm_matrix_sparce.nnz, " Memory size: ", Tsm_sm_matrix_sparce.data.nbytes/1e6, " MB")
        print("        rho0 size: ", rho0.shape, " Memory size: ", rho0.nbytes/1e6, " MB")
        print("Total memory size: ", (Tsm_sm_matrix_sparce.data.nbytes + rho0.nbytes)/1e6, " MB")

    return new_eta


## Solve the problem

def solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=eta_numpy, verbose=False, device='cpu'):
    """
    This function should solve the following:
    --> New_eta = (1 - gamma T)^-1 rho
    """
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
    
    need_sparse = False
    if func_eta == eta_scipy_sparce_solve or func_eta == eta_petsc:
        need_sparse = True
    # Tsm_sm has size ~ 10^5 x 10^5 or more
    if need_sparse:
        Tsm_sm_matrix = build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,A//M,p_a_mu_m_xy)
    else:
        Tsm_sm_matrix = build_Tsm_sm_dense(M,Lx,Ly,Lx0,Ly0,find_range,A//M,p_a_mu_m_xy)

    if verbose:
        print("-"*50)
        print("Solver info:")
        print("pi shape:",pi.shape)
        print("new_eta shape:",new_eta.shape)
        print("PY shape:",PY.shape, "PAMU shape:",PAMU.shape)
        print("p_a_mu_m_xy shape:",p_a_mu_m_xy.shape)
        
        if need_sparse:
            print("               Type of Tsm_sm_matrix:",type(Tsm_sm_matrix))
            print("number of non-zeros in Tsm_sm_matrix:",Tsm_sm_matrix.nnz)
            print("number of     zeros in Tsm_sm_matrix:",M*Lx*Ly*M*Lx*Ly - Tsm_sm_matrix.nnz)
            print("               opacity of the matrix: {:7.4f}".format(Tsm_sm_matrix.nnz/(M*Lx*Ly*M*Lx*Ly) * 100.0), "%")
            print("           Sparse matrix memory size:", Tsm_sm_matrix.data.nbytes/1e6, " MB")
            # plt.imshow(Tsm_sm_matrix.toarray())
            # plt.show()

        else :
            print("               Type of Tsm_sm_matrix:",type(Tsm_sm_matrix))
            print("number of non-zeros in Tsm_sm_matrix:",np.count_nonzero(Tsm_sm_matrix != 0))
            print("number of     zeros in Tsm_sm_matrix:",np.count_nonzero(Tsm_sm_matrix == 0))
            print("               opacity of the matrix: {:7.4f}".format(np.count_nonzero(Tsm_sm_matrix != 0)/Tsm_sm_matrix.size * 100.0), "%") 
            print("             Full matrix memory size:", Tsm_sm_matrix.nbytes/1e6, " MB")
            # plt.imshow(Tsm_sm_matrix)
            # plt.show()

    if func_eta == eta_petsc:
        list_ksp_type = ['cg','gmres','bcgs','bcgsl','bcgsr','bicg','bicgsta','tfqmr','cr','minres','symmlq','lgmres']
        list_pc_type = ['none','jacobi','sor','lu','cholesky','bjacobi','mg','eisenstat','ilu','icc','asm','gasm','ksp','preonly']

        # for ksp_type in list_ksp_type:
        #     for pc_type in list_pc_type:
        #         print(ksp_type,pc_type)
        #         try :
        #             %timeit func_eta(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,ksp_type,pc_type)
        #         except :
        #             print("Incompatible arrangement")
        # new_eta = func_eta(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,'bcgsl','sor')
        new_eta = func_eta(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,'bcgs','sor',device=device)
    else :
        new_eta = func_eta(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,device=device)
    # new_eta = func_eta(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0)
            
    # new_eta = func_eta(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,device=device)
    
    return new_eta, Tsm_sm_matrix

## TEST

if __name__ == "__main__":

    dim_factor = 1 if len(sys.argv) < 2 else int(sys.argv[1])

    M = 5 # O(10) 
    O = 2 # 2/3
    Lx = 11 * dim_factor # O(100)-O(1000)
    Ly = 15 * dim_factor # O(100)-O(1000)        
    a_size = 4

    print("-"*50)
    print("System parameters:")
    print("M:",M,"   O:",O,"   Lx:",Lx,"   Ly:",Ly,"   # actions:",a_size,"   dim_factor:",dim_factor)

    # np.random.seed(33+33)

    pi = softmax( np.random.rand(O,M,M*a_size), 2)

    PObs_lim = np.random.rand(O, M*Lx*Ly)
    PObs_lim[1] = 1-PObs_lim[0]
    Lx0 = 5
    Ly0 = 7
    rho0 = np.random.rand(M*Lx*Ly)
    rho0[Lx:] = 0
    rho0 /= np.sum(rho0)
    eta0 = np.random.rand(M*Lx*Ly)
    gamma = 0.99
    find_range = 2.1
    tol = 1e-8

    inv_sol, T = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range, verbose=True,func_eta=eta_scipy_sparce_solve)
    iter_sol = itsol(pi, PObs_lim, gamma, rho0, eta0, tol, Lx, Ly, Lx0, Ly0, find_range)

    T = None

    ## Benchmarking of the solvers

    # solvers = [ eta_numpy,eta_numpy_sol,eta_torch,eta_torch_sol, eta_torch, eta_torch_sol,eta_scipy_sparce_solve, eta_petsc,eta_petsc]
    # name_solvers =    ["Numpy", "Numpy Sol", "Torch CPU", "Torch SolCPU", "Torch", "Torch Sol", "Scipy", "Petsc GPU", "Petsc CPU"]
    # device_selector = [  'cpu',       'cpu',       'cpu',          'cpu',   'gpu',       'gpu',   'cpu',       'gpu',       'cpu']
    solvers = [eta_scipy_sparce_solve,eta_petsc]
    name_solvers =    ["Scipy", "Petsc CPU"]
    device_selector = [  'cpu',       'cpu']

    # print("-"*50)
    # print("Details of the solvers:")

    # for solver, name_solver, device_sel in zip(solvers, name_solvers,device_selector):
    #     print("Method: {:>12}".format(name_solver), end=" ")
    #     inv_sol, T = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=solver, verbose=True,device=device_sel)


    print("-"*50)
    print("Benchmarking of the solvers")
    runtimes = 2
    print("Number of runs:",runtimes)

    for solver, name_solver,device_sel in zip(solvers, name_solvers,device_selector):
        print("Method: {:>12}".format(name_solver), end=" ")
        total_time = 0.0
        times = np.zeros(runtimes)
        for i in range(runtimes):
            t = time.process_time()

            out_01, out_02 = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=solver, verbose=False,device=device_sel)
            out_01 = None
            out_02 = None

            times[i] = time.process_time() - t
            print("*_", end=" ", flush=True)
        print("Average time: {:.4f} s".format(np.mean(times)),"std dev: {:.4f} s".format(np.std(times)))

    print("\nIterative Solution", end=" ")

    total_time = 0.0
    times = np.zeros(runtimes)

    for i in range(runtimes):
        t = time.process_time()
        _ = itsol(pi, PObs_lim, gamma, rho0, eta0, tol, Lx, Ly, Lx0, Ly0, find_range)
        times[i] = time.process_time() - t
    print("Average time: {:.4f} s,".format(np.mean(times)),"std dev: {:.4f} s".format(np.std(times)))

    print("-"*50)
    print("Diff between iterative solution and solvers")

    memory = 0
    print("Shape     iter_sol:", iter_sol.shape[0],
          "max: {:.4f} min: {:.4f}".format(np.max(iter_sol), np.min(iter_sol)))

    for solver, name_solver, device_sel in zip(solvers, name_solvers,device_selector):
        inv_sol, T = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=solver, verbose=False,device=device_sel)
        print("Shape {:>12}: {:} max: {:.4f} min: {:.4f}".format(name_solver, inv_sol.shape[0], np.max(inv_sol), np.min(inv_sol)), end=" ", flush=True)  
        print("|| Passing test of solver:",solution_test(inv_sol,T,rho0,gamma), end=" ", flush=True)
        print("|| Diff from iter_sol: {:.8f}".format(np.sum(np.abs(iter_sol-inv_sol))))

    print("-"*50)
    print("Diff between solvers")        
    print("{:>12}|".format(""), end=" ")
    [print("{:>12}".format(i), end=" ") for i in name_solvers]
    print("{:>12}".format("|"))
    for i in range(len(solvers)):
        inv_sol_i, T_i = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=solvers[i],verbose=False,device=device_selector[i])
        print("{:>12}|".format(name_solvers[i]), end=" ")
        [print("{:12}".format(""), end=" ") for _ in range(i)]
        print("{:12.5e}".format(0.0), end=" ", flush=True)
        for j in range(i+1,len(solvers)):
            inv_sol_j, T_j = solve_eta(pi, PObs_lim, gamma, rho0, Lx, Ly, Lx0, Ly0, find_range,func_eta=solvers[j],verbose=False,device=device_selector[j])
            print("{:12.5e}".format(np.sum(np.abs(inv_sol_i-inv_sol_j))), end=" ", flush=True) 
        print("{:>12}".format("|"))



