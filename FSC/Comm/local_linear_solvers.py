import numpy as np
import itertools as it
import scipy.sparse as sparse

import numba

@numba.njit 
def index_six_to_two(index,M,Ly,Lx):
    # Convert the index of an array of 6 dim to 2 dim
    # new_index_x = index[0] * Lx * Ly + index[1] * Lx + index[2]
    # new_index_y = index[3] * Lx * Ly + index[4] * Lx + index[5]
    # increse the performance of the new_index_x and new_index_y
    new_index_x = Lx * ( index[0] * Ly + index[1] ) + index[2]
    new_index_y = Lx * ( index[3] * Ly + index[4] ) + index[5]

    return (new_index_x,new_index_y)

# @profile
def build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,action_size,p_a_mu_m_xy):

    clip = lambda x, l, u: l if x < l else u if x > u else x

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
            Tsm_sm_sp[indexes] +=  p_a_mu_m_xy[a]
                
        right = [ (im_new, iy, clip(ix+1, 0, Lx-1), im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        right_act = [ (1, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]

        for t, a in zip(right, right_act):
            indexes = index_six_to_two(t,M,Ly,Lx)
            Tsm_sm_sp[indexes] +=  p_a_mu_m_xy[a]

        up = [ (im_new, clip(iy+1, 0, Ly-1), ix,  im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        up_act = [ (2, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]

        for t, a in zip(up, up_act):
            indexes = index_six_to_two(t,M,Ly,Lx)
            Tsm_sm_sp[indexes] +=  p_a_mu_m_xy[a]
        
        down = [ (im_new, clip(iy-1, 0, Ly-1), ix, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]
        down_act = [ (3, im_new, im, iy, ix)   for ix in np.arange(Lx) for iy in np.arange(Ly) for im in np.arange(M)]

        for t, a in zip(down, down_act):
            indexes = index_six_to_two(t,M,Ly,Lx)
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

def eta_scipy_sparce_solve(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,verbose=False,device='cpu'):
    """
    Solve linear system using scipy sparce
    """
    Tsm_sm_matrix_sparce = sparse.csr_matrix(Tsm_sm_matrix)
        
    Tsm_sm_matrix_sparce *= -gamma
    Tsm_sm_matrix_sparce += sparse.eye(M*Ly*Lx)

    new_eta = sparse.linalg.spsolve(Tsm_sm_matrix_sparce,rho0)
    return new_eta

def eta_petsc(Tsm_sm_matrix,gamma,M,Lx,Ly,rho0,ks_type,ps_type,verbose=False,device='cpu'):
    pass

# @profile
def build_Tsm_sm_sparse_2(M,Lx,Ly,Lx0,Ly0,find_range,p_a_mu_m_xy,act_hdl,source_as_zero,verbose=0):

    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Create the sparse matrix
    Tsm_sm_sp = sparse.lil_matrix((M*Ly*Lx, M*Ly*Lx), dtype=np.double)

    if verbose >= 1:
        print("--- build_Tsm_sm_sparse_2 ---")
        print("M:",M,"Lx:",Lx,"Ly:",Ly,"Lx0:",Lx0,"Ly0:",Ly0)
    for im_new in range(M):
        for act in range(act_hdl.A):

            move = act_hdl.action_move(act)

            limits_x = [0,Lx]
            limits_y = [0,Ly]
            skid_x = 0
            skid_y = 0

            data2clip_range_x = [] #range(Lx)
            target2clipped_x  = [] #range(Lx)
            data2clip_range_y = [] #range(Ly)
            target2clipped_y  = [] #range(Ly)

            # move in x
            if move[-1] > 0: # right
                limits_x[0] += move[-1]
                skid_x -= move[-1]
                data2clip_range_x = [i for i in range(Lx+skid_x,Lx)]
                target2clipped_x = [Lx - 1 for _ in range(abs(move[-1]))]
            elif move[-1] < 0: # left
                limits_x[1] += move[-1]
                skid_x -= move[-1]
                data2clip_range_x = [i for i in range(skid_x)]
                target2clipped_x = [0 for _ in range(abs(move[-1]))]

            # move in y
            if move[-2] > 0: # up
                limits_y[0] += move[-2]
                skid_y -= move[-2]
                data2clip_range_y = [i for i in range(Ly+skid_y,Ly)]
                target2clipped_y = [Ly - 1 for _ in range(abs(move[-2]))]
            elif move[-2] < 0: # down
                limits_y[1] += move[-2]
                skid_y -= move[-2]
                data2clip_range_y = [i for i in range(skid_y)]
                target2clipped_y = [0 for _ in range(abs(move[-2]))]
            
            # move in z

            # Number of move directions
            move_dir = []
            if move[-1] != 0:
                move_dir.append(0)
            if move[-2] != 0:
                move_dir.append(1)
            # if move[-3] != 0:
            #     move_dir.append(2)

            if verbose >= 1:
                print("-"*50)
                print("act {:>5}".format(act_hdl.actions_names[act]),"limits y:",limits_y,"x:",limits_x,"move",move,"skid",skid_y,skid_x,"\nclipped",data2clip_range_y,data2clip_range_x,"\nntarget",target2clipped_y,target2clipped_x)

            #-----------------------------------------------------------------
            # Fill the direct values
            #-----------------------------------------------------------------
            index_policy_act = [ (act, im_new, im, iy+skid_y, ix+skid_x)  
                                 for im in range(M)  
                                    for iy in range(limits_y[0],limits_y[1])
                                        for ix in range(limits_x[0],limits_x[1]) 
                                        ]
            index_matrix_act = [ index_six_to_two((im_new, iy, ix, im, iy+skid_y, ix+skid_x),M,Ly,Lx)
                                 for im in range(M) 
                                    for iy in range(limits_y[0],limits_y[1])
                                        for ix in range(limits_x[0],limits_x[1]) 
                                        ]
            if verbose and False:
                for i in range(Ly+2):
                    print("index_policy_act",index_policy_act[i],"index_matrix_act",index_matrix_act[i])
            # index_matrix_act = [ index_six_to_two(index,M,Ly,Lx) for index in index_matrix_act]

            for t, a in zip(index_matrix_act, index_policy_act):
                Tsm_sm_sp[t] += p_a_mu_m_xy[a]

            #-----------------------------------------------------------------
            # Add the clipped values 
            #-----------------------------------------------------------------
            if verbose >= 1:
                print("move_dir",move_dir)
            
            limits_x = [0,Lx] # list of consecutive actions to clip

            for m_dir in move_dir:
                # clip in x
                if m_dir == 0:
                    range_x = data2clip_range_x
                    range_y = range(Ly)
                    target_x = target2clipped_x
                    target_y = [clip(i,0,Ly-1) for i in range(0-skid_y,Ly-skid_y)]

                    if range_x[0] == 0:
                        limits_x[0] = len(target_x)
                        limits_x[1] = Lx
                    else:   
                        limits_x[0] = 0
                        limits_x[1] = Lx - len(target_x) 

                # clip in y
                elif m_dir == 1:
                    range_x = range(limits_x[0],limits_x[1])
                    range_y = data2clip_range_y
                    target_x = [clip(i,0,Lx-1) for i in range(limits_x[0]-skid_x,limits_x[1]-skid_x)]
                    target_y = target2clipped_y 

                if verbose >= 1:
                    print("m_dir",m_dir,"range_x",range_x,"range_y",range_y,"target_x",target_x,"target_y",target_y)
                    print("limits_x",limits_x)
                

                index_policy_act = [ (act, im_new, im, iy, ix)  
                                    for im in range(M)  
                                        for iy in range_y
                                            for ix in range_x 
                                            ]
                index_matrix_act = [ index_six_to_two((im_new, iy_clip, ix_clip, im, iy, ix),M,Ly,Lx)
                                    for im in range(M) 
                                        for iy_clip, iy in zip(target_y,range_y)
                                            for ix_clip, ix in zip(target_x,range_x)
                                            ]
                if verbose >= 2:
                    print("policy_act size:",len(index_policy_act),"matrix size:",len(index_matrix_act))
                    for i in range(min(Ly+20000,len(index_policy_act))):
                        print("index_policy_act",index_policy_act[i],"index_matrix_act",index_matrix_act[i])

                # index_matrix_act = [ index_six_to_two(index,M,Ly,Lx) for index in index_matrix_act]

                for t, a in zip(index_matrix_act, index_policy_act):
                    Tsm_sm_sp[t] += p_a_mu_m_xy[a]

        # End for act
    # End for im_new

    # Delete the rows and columns that have been set to zero depending on the distance to the initial position

    indexes = np.zeros(6,dtype=int)
    for yx_found in source_as_zero:
        for im in range(M):
            indexes[0] = indexes[3] = im
            indexes[1] = indexes[4] = yx_found[0]
            indexes[2] = indexes[5] = yx_found[1]
            indexes_mat = index_six_to_two(indexes,M,Ly,Lx)
            # rows to set zero
            Tsm_sm_sp[indexes_mat[0],:] = 0
            # columns to set zero
            Tsm_sm_sp[:,indexes_mat[1]] = 0

    return Tsm_sm_sp