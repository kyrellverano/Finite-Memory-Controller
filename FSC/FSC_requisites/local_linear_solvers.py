import sys
import time
import numpy as np
import itertools as it
import scipy.sparse as sparse

# ----------------------------------------------------------------------------
def index_six_to_two(index,M,Ly,Lx):
    """ Convert the index of an array of 6 dim to 2 dim 

    Parameters
    ----------
    index : array of 6 dim
        index[0] = im
        index[1] = iy
        index[2] = ix
        index[3] = jm
        index[4] = jy
        index[5] = jx
    M : int
        Number of memory states
    Ly : int
        Number of rows
    Lx : int
        Number of columns

    Returns
    -------

    new_index : tuple
        index of the new array of 2 dim
        (new_index_x, new_index_y)
    
    """

    new_index_x = Lx * ( index[0] * Ly + index[1] ) + index[2]
    new_index_y = Lx * ( index[3] * Ly + index[4] ) + index[5]

    return (new_index_x,new_index_y)

def index_six_to_two_2(index,M,Ly,Lx):
    """ Convert the index of an array of 6 dim to 2 dim 

    Parameters
    ----------
    index : array of 6 dim
        index[0] = im
        index[1] = iy
        index[2] = ix
        index[3] = jm
        index[4] = jy
        index[5] = jx
    M : int
        Number of memory states
    Ly : int
        Number of rows
    Lx : int
        Number of columns

    Returns
    -------

    new_index : list
        index of the new array of 2 dim
        [new_index_x, new_index_y]
    
    """
    new_index_x = Lx * ( index[0] * Ly + index[1] ) + index[2]
    new_index_y = Lx * ( index[3] * Ly + index[4] ) + index[5]

    return [new_index_x,new_index_y]

# @profile
def build_Tsm_sm_sparse(M,Lx,Ly,Lx0,Ly0,find_range,action_size,p_a_mu_m_xy):
    """ Build the sparse matrix of the transition probabilities. 
        Initial version, not optimized.

    Parameters
    ----------
    M : int
        Number of memory states
    Lx : int
        Number of columns
    Ly : int
        Number of rows
    Lx0 : float
        Odor source position in x
    Ly0 : float
        Odor source position in y
    find_range : int
        Range of the search
    action_size : int
        Number of actions
    p_a_mu_m_xy : ndarray
        Probability of the action a, memory state mu, position x and y,
        to end in the memory state m and position x and y

    Returns
    -------
    Tsm_sm_sp : sparse matrix
        Sparse matrix of the transition probabilities

    """

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

# @profile
def build_Tsm_sm_sparse_2(M,Lx,Ly,Lx0,Ly0,find_range,p_a_mu_m_xy,act_hdl,source_as_zero,verbose=0):
    """ Build the sparse matrix of the transition probabilities. 
        Version 2, not optimized, general case.

    Parameters
    ----------
    M : int
        Number of memory states
    Lx : int
        Number of columns
    Ly : int
        Number of rows
    Lx0 : float
        Odor source position in x
    Ly0 : float
        Odor source position in y
    find_range : int
        Range of the search
    action_size : int
        Number of actions
    p_a_mu_m_xy : ndarray
        Probability of the action a, memory state mu, position x and y,
        to end in the memory state m and position x and y

    Returns
    -------
    Tsm_sm_sp : sparse matrix
        Sparse matrix of the transition probabilities

    """

    timing = True
    if timing:
        print("build 02")
        timer_step = np.zeros(9)
        timer_step[-1] = time.time()
        timer_snaps = np.zeros(9)

    clip = lambda x, l, u: l if x < l else u if x > u else x

    if verbose >= 1:
        print("--- build_Tsm_sm_sparse_2 ---")
        print("M:",M,"Lx:",Lx,"Ly:",Ly,"Lx0:",Lx0,"Ly0:",Ly0)

    # Collector of the information of the full matrix
    full_policy_act = np.array([],dtype=np.double)
    full_index_matrix = np.array([],dtype=int)

    for im_new in range(M):
        for act in range(act_hdl.A):
            
            if timing:
                timer_step[0] += time.time() - timer_snaps[4]
                timer_snaps[0] = time.time()

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
            
            if timing:
                timer_step[1] += time.time() - timer_snaps[0]
                timer_snaps[1] = time.time()
            #-----------------------------------------------------------------
            # Fill the direct values
            #-----------------------------------------------------------------
            # index_policy_act = [ (act, im_new, im, iy+skid_y, ix+skid_x)  
            #                      for im in range(M)  
            #                         for iy in range(limits_y[0],limits_y[1])
            #                             for ix in range(limits_x[0],limits_x[1]) 
            #                             ]
            full_policy_act = np.concatenate(
               (full_policy_act,
                np.array([ p_a_mu_m_xy[(act, im_new, im, iy+skid_y, ix+skid_x)]  
                                for im in range(M)  
                                    for iy in range(limits_y[0],limits_y[1])
                                        for ix in range(limits_x[0],limits_x[1]) 
                                        ]
                        )
                )
            )
            if timing:
                timer_step[2] += time.time() - timer_snaps[1]
                timer_snaps[2] = time.time()

            # index_matrix_act = [ index_six_to_two((im_new, iy, ix, im, iy+skid_y, ix+skid_x),M,Ly,Lx)
            #                      for im in range(M) 
            #                         for iy in range(limits_y[0],limits_y[1])
            #                             for ix in range(limits_x[0],limits_x[1]) 
            #                             ]
            full_index_matrix = np.concatenate(
                (full_index_matrix,
                 np.array([ index_six_to_two_2((im_new, iy, ix, im, iy+skid_y, ix+skid_x),M,Ly,Lx)
                                 for im in range(M) 
                                    for iy in range(limits_y[0],limits_y[1])
                                        for ix in range(limits_x[0],limits_x[1]) 
                                        ]
                        )
                )
            ,axis=None)

            if timing:
                timer_step[3] += time.time() - timer_snaps[2]
                timer_snaps[3] = time.time()

            if verbose and False:
                for i in range(Ly+2):
                    print("index_policy_act",index_policy_act[i],"index_matrix_act",index_matrix_act[i])
            # index_matrix_act = [ index_six_to_two(index,M,Ly,Lx) for index in index_matrix_act]

            # for t, a in zip(index_matrix_act, index_policy_act):
            #     Tsm_sm_sp[t] += p_a_mu_m_xy[a]

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
                

                # index_policy_act = [ (act, im_new, im, iy, ix)  
                #                     for im in range(M)  
                #                         for iy in range_y
                #                             for ix in range_x 
                #                             ]
                full_policy_act = np.concatenate(
                    (full_policy_act,
                    np.array([ p_a_mu_m_xy[act, im_new, im, iy, ix]  
                                    for im in range(M)  
                                        for iy in range_y
                                            for ix in range_x
                                            ]
                            )
                    )
                )

                # index_matrix_act = [ index_six_to_two((im_new, iy_clip, ix_clip, im, iy, ix),M,Ly,Lx)
                #                     for im in range(M) 
                #                         for iy_clip, iy in zip(target_y,range_y)
                #                             for ix_clip, ix in zip(target_x,range_x)
                #                             ]
                full_index_matrix = np.concatenate(
                    (full_index_matrix,
                     np.array([ index_six_to_two_2((im_new, iy_clip, ix_clip, im, iy, ix),M,Ly,Lx)
                                     for im in range(M) 
                                        for iy_clip, iy in zip(target_y,range_y)
                                            for ix_clip, ix in zip(target_x,range_x)
                                            ]
                            )
                    )
                ,axis=None)

                if verbose >= 2:
                    print("policy_act size:",len(index_policy_act),"matrix size:",len(index_matrix_act))
                    for i in range(min(Ly+20000,len(index_policy_act))):
                        print("index_policy_act",index_policy_act[i],"index_matrix_act",index_matrix_act[i])

                # index_matrix_act = [ index_six_to_two(index,M,Ly,Lx) for index in index_matrix_act]

                # for t, a in zip(index_matrix_act, index_policy_act):
                #     Tsm_sm_sp[t] += p_a_mu_m_xy[a]

            # End for m_dir
            if timing:
                timer_step[4] += time.time() - timer_snaps[3]
                timer_snaps[4] = time.time()
                

        # End for act
    # End for im_new

    if timing:
        timer_step[5] = time.time() - timer_step[-1]
        timer_snaps[5] = time.time()

    print("shape:",full_policy_act.shape,full_index_matrix.shape)
    # Reshape the arrays of indexes to be able to create the sparse matrix
    entries = full_policy_act.shape[0]
    full_index_matrix = full_index_matrix.reshape(entries,2)

    if verbose >= 1:
        print("full_policy_act size:",len(full_policy_act),"full_index_matrix size:",len(full_index_matrix))
        print("full_policy_act",full_policy_act.shape,"full_index_matrix",full_index_matrix.shape)

    # Create the sparse matrix and fill it with the values
    Tsm_sm_sp = sparse.coo_matrix((full_policy_act, (full_index_matrix[:,0], full_index_matrix[:,1])), shape=(M*Ly*Lx, M*Ly*Lx), dtype=np.double)
    # Convert to lil format to be able to set rows and columns to zero
    Tsm_sm_sp = Tsm_sm_sp.tolil() 

    if timing:
        timer_step[6] = time.time() - timer_snaps[5]
        timer_snaps[6] = time.time()

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

    if timing:
        timer_step[7] = time.time() - timer_snaps[6]
        total_time = time.time() - timer_step[-1]

        # diff_time = np.zeros(timer_step.size)
        # diff_time[0] = timer_step[0].copy()

        # for i in range(1,timer_step.size):
        #     diff_time[i] = timer_step[i] - timer_step[i-1]

        # timer_step = diff_time / total_time
        print("Rank:",0,
            "Move",   "{:.2f}".format(timer_step[1]),
            "Fill_policy:", "{:.2f}".format(timer_step[2]),
            "Fill matrix id:", "{:.2f}".format(timer_step[3]),
            "Clipped:", "{:.2f}".format(timer_step[4]),
            "For loop:", "{:.2f}".format(timer_step[5]),
            "Set matrix:", "{:.2f}".format(timer_step[6]),
            "Zeros:", "{:.2f}".format(timer_step[7]),
            "Total:",   "{:.2f}".format(total_time))

    print('*'*50)
    print('Tsm_sm_2',1)
    Tsm_sm_matrix = Tsm_sm_sp.copy().tocsr()
    col = Tsm_sm_matrix.indices
    row = Tsm_sm_matrix.indptr
    data = Tsm_sm_matrix.data
    print(col[0:10])
    print(row[0:10])
    print(data[0:10])
    print('*'*50)


    return Tsm_sm_sp

def set_index_4_Tsm_sm(M,Lx,Ly,Lx0,Ly0,act_hdl,verbose=0):
    """ Build the index for transition probabilities in sparse matrix Tsm_sm_sp.

    Parameters
    ----------
    M : int
        Number of memory states
    Lx : int
        Number of columns
    Ly : int
        Number of rows
    Lx0 : float
        Odor source position in x
    Ly0 : float
        Odor source position in y
    act_hdl : object
        Object with the actions information
    verbose : int
        Level of verbosity

    Returns
    -------
    full_policy_act : ndarray
        Array with the indexes for p_a_mu_m_xy
    full_index_matrix : ndarray
        Array with the index of the matrix Tsm_sm_sp

    """
    
    timing = False
    if timing:
        timer_step = np.zeros(9)
        timer_step[-1] = time.time()
        timer_snaps = np.zeros(9)

    clip = lambda x, l, u: l if x < l else u if x > u else x

    if verbose >= 1:
        print("M:",M,"Lx:",Lx,"Ly:",Ly,"Lx0:",Lx0,"Ly0:",Ly0)

    # Collector of the information of the full matrix
    full_policy_act = np.array([],dtype=int).reshape(0,5)
    full_index_matrix = np.array([],dtype=int)

    for im_new in range(M):
        for act in range(act_hdl.A):
            
            if timing:
                timer_step[0] += time.time() - timer_snaps[4]
                timer_snaps[0] = time.time()

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
            
            if timing:
                timer_step[1] += time.time() - timer_snaps[0]
                timer_snaps[1] = time.time()
            #-----------------------------------------------------------------
            # Fill the direct values
            #-----------------------------------------------------------------
            # index_policy_act = [ (act, im_new, im, iy+skid_y, ix+skid_x)  
            #                      for im in range(M)  
            #                         for iy in range(limits_y[0],limits_y[1])
            #                             for ix in range(limits_x[0],limits_x[1]) 
            #                             ]
            full_policy_act = np.concatenate(
               (full_policy_act,
                np.array([ [act, im_new, im, iy+skid_y, ix+skid_x]  
                                for im in range(M)  
                                    for iy in range(limits_y[0],limits_y[1])
                                        for ix in range(limits_x[0],limits_x[1]) 
                                        ]
                        )
                )
            )
            if timing:
                timer_step[2] += time.time() - timer_snaps[1]
                timer_snaps[2] = time.time()

            # index_matrix_act = [ index_six_to_two((im_new, iy, ix, im, iy+skid_y, ix+skid_x),M,Ly,Lx)
            #                      for im in range(M) 
            #                         for iy in range(limits_y[0],limits_y[1])
            #                             for ix in range(limits_x[0],limits_x[1]) 
            #                             ]
            full_index_matrix = np.concatenate(
                (full_index_matrix,
                 np.array([ index_six_to_two_2((im_new, iy, ix, im, iy+skid_y, ix+skid_x),M,Ly,Lx)
                                 for im in range(M) 
                                    for iy in range(limits_y[0],limits_y[1])
                                        for ix in range(limits_x[0],limits_x[1]) 
                                        ]
                        )
                )
            ,axis=None)

            if timing:
                timer_step[3] += time.time() - timer_snaps[2]
                timer_snaps[3] = time.time()

            if verbose and False:
                for i in range(Ly+2):
                    print("index_policy_act",index_policy_act[i],"index_matrix_act",index_matrix_act[i])
            # index_matrix_act = [ index_six_to_two(index,M,Ly,Lx) for index in index_matrix_act]

            # for t, a in zip(index_matrix_act, index_policy_act):
            #     Tsm_sm_sp[t] += p_a_mu_m_xy[a]

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
                

                # index_policy_act = [ (act, im_new, im, iy, ix)  
                #                     for im in range(M)  
                #                         for iy in range_y
                #                             for ix in range_x 
                #                             ]
                full_policy_act = np.concatenate(
                    (full_policy_act,
                    np.array([ [act, im_new, im, iy, ix] 
                                    for im in range(M)  
                                        for iy in range_y
                                            for ix in range_x
                                            ]
                            )
                    )
                )

                # index_matrix_act = [ index_six_to_two((im_new, iy_clip, ix_clip, im, iy, ix),M,Ly,Lx)
                #                     for im in range(M) 
                #                         for iy_clip, iy in zip(target_y,range_y)
                #                             for ix_clip, ix in zip(target_x,range_x)
                #                             ]
                full_index_matrix = np.concatenate(
                    (full_index_matrix,
                     np.array([ index_six_to_two_2((im_new, iy_clip, ix_clip, im, iy, ix),M,Ly,Lx)
                                     for im in range(M) 
                                        for iy_clip, iy in zip(target_y,range_y)
                                            for ix_clip, ix in zip(target_x,range_x)
                                            ]
                            )
                    )
                ,axis=None)

                if verbose >= 2:
                    print("policy_act size:",len(index_policy_act),"matrix size:",len(index_matrix_act))
                    for i in range(min(Ly+20000,len(index_policy_act))):
                        print("index_policy_act",index_policy_act[i],"index_matrix_act",index_matrix_act[i])

                # index_matrix_act = [ index_six_to_two(index,M,Ly,Lx) for index in index_matrix_act]

            # End for m_dir
            if timing:
                timer_step[4] += time.time() - timer_snaps[3]
                timer_snaps[4] = time.time()
        # End for act
    # End for im_new

    return full_policy_act, full_index_matrix

# @profile
def build_Tsm_sm_sparse_3(M,Lx,Ly,Lx0,Ly0,find_range,p_a_mu_m_xy,act_hdl,source_as_zero,solver,verbose=0):
    """ Build the sparse matrix of the transition probabilities. 
        Version 3, optimized without permanent allocation, general case.

    Parameters
    ----------
    M : int
        Number of memory states
    Lx : int
        Number of columns
    Ly : int
        Number of rows
    Lx0 : float
        Odor source position in x
    Ly0 : float
        Odor source position in y
    find_range : int
        Range of the search
    action_size : int
        Number of actions
    p_a_mu_m_xy : ndarray
        Probability of the action a, memory state mu, position x and y,
        to end in the memory state m and position x and y
    act_hdl : object
        Object with the actions information
    source_as_zero : list
        List of the positions to set to zero
    verbose : int
        Level of verbosity


    Returns
    -------
    Tsm_sm_sp : sparse matrix
        Sparse matrix of the transition probabilities

    """

    timing = False
    if timing:
        timer_step = np.zeros(5)
        timer_step[0] = time.time()

    # Compute only once the indexes of the matrix Tsm_sm_sp
    if solver.Tmatrix_index is None and solver.Tmatrix_p_index is None:
        solver.Tmatrix_p_index, solver.Tmatrix_index  = set_index_4_Tsm_sm(M,Lx,Ly,Lx0,Ly0,act_hdl,verbose=verbose)

        entries = solver.Tmatrix_p_index.shape[0]
        solver.Tmatrix_index = solver.Tmatrix_index.reshape(entries,2)

    if timing:
        timer_step[1] = time.time()

    # Reshape the arrays of indexes to be able to create the sparse matrix

    full_policy_act = p_a_mu_m_xy[solver.Tmatrix_p_index[:,0],solver.Tmatrix_p_index[:,1],solver.Tmatrix_p_index[:,2],solver.Tmatrix_p_index[:,3],solver.Tmatrix_p_index[:,4]]
    full_index_matrix = solver.Tmatrix_index

    if verbose >= 1:
        print("full_policy_act size:",len(full_policy_act),"full_index_matrix size:",len(full_index_matrix))
        print("full_policy_act",full_policy_act.shape,"full_index_matrix",full_index_matrix.shape)

    if timing:
        timer_step[2] = time.time()
    # Create the sparse matrix and fill it with the values

    if solver.Tsm_sm_sp is None:
        solver.Tsm_sm_sp = sparse.coo_matrix((full_policy_act, (full_index_matrix[:,0], full_index_matrix[:,1])), shape=(M*Ly*Lx, M*Ly*Lx), dtype=np.double)
    else :
        solver.Tsm_sm_sp.data = full_policy_act

    # Convert to lil format to be able to set rows and columns to zero
    Tsm_sm_sp = solver.Tsm_sm_sp.tolil() 

    if timing:
        timer_step[3] = time.time()

    # Delete the rows and columns that have been set to zero depending on the distance to the initial position
    if solver.Tsm_sm_zero is None:
        solver.Tsm_sm_zero = []
        indexes = np.zeros(6,dtype=int)
        for yx_found in source_as_zero:
            for im in range(M):
                indexes[0] = indexes[3] = im
                indexes[1] = indexes[4] = yx_found[0]
                indexes[2] = indexes[5] = yx_found[1]
                indexes_mat = index_six_to_two(indexes,M,Ly,Lx)
                solver.Tsm_sm_zero.append(indexes_mat)
        
        solver.Tsm_sm_zero = np.array(solver.Tsm_sm_zero)

    # rows to set zero
    Tsm_sm_sp[solver.Tsm_sm_zero[:,0],:] = 0.0
    # columns to set zero
    Tsm_sm_sp[:,solver.Tsm_sm_zero[:,1]] = 0.0

    if timing:
        timer_step[4] = time.time()
        total_time = timer_step[-1] - timer_step[0]

        diff_time = np.zeros(timer_step.size)
        diff_time[0] = timer_step[0].copy()

        for i in range(1,timer_step.size):
            diff_time[i] = timer_step[i] - timer_step[i-1]

        timer_step = diff_time / total_time
        print("Set T ---",
            "Rank:",0,
            "Set index",   "{:.2f}".format(timer_step[1]),
            "Fill_policy:", "{:.2f}".format(timer_step[2]),
            "Fill matrix id:", "{:.2f}".format(timer_step[3]),
            "Zeros", "{:.2f}".format(timer_step[4]),
            "Total:",   "{:.4f}".format(total_time))
        
    return Tsm_sm_sp

# @profile
def build_Tsm_sm_sparse_4(M,Lx,Ly,Lx0,Ly0,find_range,p_a_mu_m_xy,act_hdl,source_as_zero,solver,verbose=0):
    """ Build the sparse matrix of the transition probabilities. 
        Version 4, optimized with permanent allocation, general case.

    Parameters
    ----------
    M : int
        Number of memory states
    Lx : int
        Number of columns
    Ly : int
        Number of rows
    Lx0 : float
        Odor source position in x
    Ly0 : float
        Odor source position in y
    find_range : int
        Range of the search
    action_size : int
        Number of actions
    p_a_mu_m_xy : ndarray
        Probability of the action a, memory state mu, position x and y,
        to end in the memory state m and position x and y
    act_hdl : object
        Object with the actions information
    source_as_zero : list
        List of the positions to set to zero
    solver : object
        Object with the solver information
    verbose : int
        Level of verbosity

    Returns
    -------
    Tsm_sm_sp : sparse matrix
        Sparse matrix of the transition probabilities

    """

    timing = False
    if timing:
        timer_step = np.zeros(5)
        timer_step[0] = time.time()

    # Compute only once the indexes of the matrix Tsm_sm_sp
    if solver.Tmatrix_index is None and solver.Tmatrix_p_index is None:
        solver.Tmatrix_p_index, solver.Tmatrix_index  = set_index_4_Tsm_sm(M,Lx,Ly,Lx0,Ly0,act_hdl,verbose=verbose)

        entries = solver.Tmatrix_p_index.shape[0]
        solver.Tmatrix_index = solver.Tmatrix_index.reshape(entries,2)

    if timing:
        timer_step[1] = time.time()

    # Reshape the arrays of indexes to be able to create the sparse matrix

    full_policy_act = p_a_mu_m_xy[solver.Tmatrix_p_index[:,0],solver.Tmatrix_p_index[:,1],solver.Tmatrix_p_index[:,2],solver.Tmatrix_p_index[:,3],solver.Tmatrix_p_index[:,4]]

    if verbose >= 1:
        print("full_policy_act size:",len(full_policy_act),"full_index_matrix size:",len(full_index_matrix))
        print("full_policy_act",full_policy_act.shape,"full_index_matrix",full_index_matrix.shape)

    if timing:
        timer_step[2] = time.time()
    # Create the sparse matrix and fill it with the values

    if solver.Tsm_sm_sp is None:
        full_index_matrix = solver.Tmatrix_index
        solver.Tsm_sm_sp = sparse.coo_matrix((full_policy_act, (full_index_matrix[:,0], full_index_matrix[:,1])), shape=(M*Ly*Lx, M*Ly*Lx), dtype=np.double)
    else :
        solver.Tsm_sm_sp.data = full_policy_act

    # Convert to lil format to be able to set rows and columns to zero
    Tsm_sm_sp = solver.Tsm_sm_sp.copy().tocsr() #.tolil()

    if timing:
        timer_step[3] = time.time()

    # Delete the rows and columns that have been set to zero depending on the distance to the initial position
    if solver.Tsm_sm_zero is None:
        solver.Tsm_sm_zero = []
        indexes = np.zeros(6,dtype=int)
        for yx_found in source_as_zero:
            for im in range(M):
                indexes[0] = indexes[3] = im
                indexes[1] = indexes[4] = yx_found[0]
                indexes[2] = indexes[5] = yx_found[1]
                indexes_mat = index_six_to_two(indexes,M,Ly,Lx)
                solver.Tsm_sm_zero.append(indexes_mat)
        
        solver.Tsm_sm_zero = np.array(solver.Tsm_sm_zero)

        # find intersection between Tsm_sm_zero and Tmatrix_index  
        Tsm_sm_zero_index = np.array([],dtype=int).reshape(0,2)

        A = solver.Tmatrix_index
        row_zero = solver.Tsm_sm_zero[:,0]
        col_zero = solver.Tsm_sm_zero[:,1]
        row_A = A[:,0]
        col_A = A[:,1]

        inter_col = np.in1d(col_A,col_zero)
        inter_row = np.in1d(row_A,row_zero)

        Tsm_sm_zero_index = np.concatenate((Tsm_sm_zero_index,A[inter_row,:]),axis=0)
        Tsm_sm_zero_index = np.concatenate((Tsm_sm_zero_index,A[inter_col,:]),axis=0)

        Tsm_sm_zero_index = np.unique(Tsm_sm_zero_index,axis=0)

        solver.Tsm_sm_zero = Tsm_sm_zero_index

    # for index in solver.Tsm_sm_zero:
    #     Tsm_sm_sp[index[0],index[1]] = 0.0
    # Tsm_sm_sp[index[:,0],index[:,1]] = 0.0
    Tsm_sm_sp[solver.Tsm_sm_zero[:,0],solver.Tsm_sm_zero[:,1]] = 0.0

    # Tsm_sm_sp.eliminate_zeros()
    # Tsm_sm_sp.sum_duplicates()
    # Tsm_sm_sp.sort_indices()
    # Tsm_sm_sp.prune()
    # print("nnz per row",np.unique(Tsm_sm_sp.getnnz(axis=1)))
    # print("nnz per col",np.unique(Tsm_sm_sp.getnnz(axis=0)))
    # print(Tsm_sm_sp.getrow(0))
    # Tsm_sm_sp.tolil()

    if timing:
        timer_step[4] = time.time()
        total_time = timer_step[-1] - timer_step[0]

        diff_time = np.zeros(timer_step.size)
        diff_time[0] = timer_step[0].copy()

        for i in range(1,timer_step.size):
            diff_time[i] = timer_step[i] - timer_step[i-1]

        timer_step = diff_time / total_time
        print( "Set T ---",
            "Rank:",0,
            "Set index",   "{:.2f}".format(timer_step[1]),
            "Fill_policy:", "{:.2f}".format(timer_step[2]),
            "Fill matrix id:", "{:.2f}".format(timer_step[3]),
            "Zeros", "{:.2f}".format(timer_step[4]),
            "Total:",   "{:.4f}".format(total_time))

    return Tsm_sm_sp

def iterative_solver_sp(T, x, b, gamma, tol, max_iter=100, verbose = False, device='cpu'):
    """
    Solve linear system using scipy sparce with jacobi method

    Parameters
    ----------
    T : sparse matrix
        Sparse matrix of the transition probabilities
    x : ndarray
        Initial values of the solution
    b : ndarray
        Right hand side of the linear system
    gamma : float
        Discount factor
    tol : float
        Tolerance for the convergence
    max_iter : int
        Maximum number of iterations
    verbose : bool
        Level of verbosity
    device : str
        Device to use

    """

    # max_iter = 0
    if max_iter == 0 or tol == -1:
        return x, False

    x_sp = x.copy()
    success = False

    if (gamma < 1.):
        tol2 = tol/(1-gamma)*tol/(1-gamma)
    else:
        tol2 = tol*tol*1000000

    # Start the iterative method
    for i in range(max_iter):

        new_x_sp = b + gamma * T.dot(x_sp)
        
        delta = new_x_sp - x_sp
        delta = delta * delta

        if (delta.max() < tol2): 
            success = True
            # print('Scipy Sparse | Converged in {} iterations'.format(i))
            return new_x_sp, success

        x_sp = new_x_sp.copy()
    
    return x, success

def load_scipy():
    """
    Load scipy sparse matrix
    """

    eta_petsc = None

    # @profile
    def eta_scipy_sparse_solve(Tsm_sm_matrix, eta, rho0, gamma, M, Lx, Ly, tol, max_iter, verbose=False, device='cpu'):
        """
        Solve linear system using scipy sparce
        """
        timing = False
        if timing:
            timer_step = np.zeros(3)
            timer_step[0] = time.time()

        if sparse.isspmatrix_csr(Tsm_sm_matrix):
            Tsm_sm_matrix_sparse = Tsm_sm_matrix.copy() 
        else:
            Tsm_sm_matrix_sparse = Tsm_sm_matrix.copy().tocsr()

        # Tsm_sm_matrix_sparse = Tsm_sm_matrix
        # rho0_sparse = sparse.csr_array(rho0).T
        # eta_sparse = sparse.csr_array(eta).T
        rho0_sparse = rho0
        eta_sparse = eta.copy()
            
        if timing:
            timer_step[1] = time.time()

        iter_success = False
        new_eta, iter_success = iterative_solver_sp(Tsm_sm_matrix_sparse, eta_sparse, rho0_sparse, gamma, tol, max_iter=max_iter)

        if not iter_success:

            Tsm_sm_matrix_sparse *= -gamma
            Tsm_sm_matrix_sparse += sparse.eye(M*Ly*Lx)

            sparse.linalg.use_solver(useUmfpack=False)
            new_eta = sparse.linalg.spsolve(Tsm_sm_matrix_sparse,rho0_sparse,use_umfpack=False)
            
            # new_eta = sparse.linalg.lsqr(Tsm_sm_matrix_sparce,rho0, x0=eta)[0]
            # new_eta, info = sparse.linalg.bicg(Tsm_sm_matrix_sparce,rho0)

            # if tol == -1 :
            #     tol = 1e-8
            # new_eta_cupy, info =  sparse.linalg.gmres(Tsm_sm_matrix_sparse, rho0_sparse, x0=eta_sparse, tol=tol)
            # if info != 0 :
            #     print("info:",info)


        if timing:
            timer_step[2] = time.time()
            total_time = timer_step[-1] - timer_step[0]

            diff_time = np.zeros(timer_step.size)
            diff_time[0] = timer_step[0].copy()

            for i in range(1,timer_step.size):
                diff_time[i] = timer_step[i] - timer_step[i-1]

            timer_step = diff_time / total_time
            print('Scipy ---',
                "Rank: 0",
                "Set T sparse:",   "{:.2f}".format(timer_step[1]),
                "Solver:", "{:.2f}".format(timer_step[2]),
                "Total:",   "{:.4f}".format(total_time))


        return new_eta

    def eta_scipy_sparse_solve_jacobi(Tsm_sm_matrix,eta,gamma,M,Lx,Ly,rho0,verbose=False,device='cpu'):
        """
        Solve linear system using scipy sparce with jacobi method
        """

        timing = True
        if timing:
            timer_step = np.zeros(4)
            timer_step[0] = time.time()
        Tsm_sm_matrix_sparce = sparse.csc_matrix(Tsm_sm_matrix)
        # rho0_sparse = sparse.csr_matrix(rho0).T
        # eta_sparse = sparse.csr_matrix(eta).T

        Tsm_sm_matrix_sparce *= -gamma
        Tsm_sm_matrix_sparce += sparse.eye(M*Ly*Lx)

        if timing:
            timer_step[1] = time.time()

        """
        D_T = sparse.diags(Tsm_sm_matrix_sparce.diagonal()).tocsc()
        LU_T = Tsm_sm_matrix_sparce - D_T

        D_T = sparse.linalg.inv(D_T)

        J = -LU_T.dot(D_T)
        C = D_T.dot(rho0_sparse)

        max_iter = 10000
        convergence = 1e-5
        # Iteration loop
        tmp_eta = eta_sparse.copy()
        for i in range(max_iter):
            eta_sparse_new = J.dot(eta_sparse) + C

            if np.allclose(eta_sparse_new.toarray(), eta_sparse.toarray(), atol=convergence):
                break

            tmp_eta = eta_sparse.copy()
            eta_sparse = eta_sparse_new.copy()

        print("Iterations:",i)
        print("norm:",np.sum(eta_sparse_new.toarray().flatten()-tmp_eta.toarray().flatten()))
        return eta_sparse_new.toarray().flatten()

        """

        print("Iterative method")

        # Preconditioned
        ilu = sparse.linalg.spilu(Tsm_sm_matrix_sparce, fill_factor=15,drop_tol=1e-6)
        M = sparse.linalg.LinearOperator(Tsm_sm_matrix_sparce.shape, ilu.solve)
        
        
        # solver = sparse.linalg.factorized(Tsm_sm_matrix_sparce)

        if timing:
            timer_step[2] = time.time()

        new_eta,info = sparse.linalg.gmres(Tsm_sm_matrix_sparce, rho0, x0=eta, tol=1e-8, M=M)
        
        # new_eta = solver(rho0)
        # info = 0 

        # new_eta = sparse.linalg.spsolve(Tsm_sm_matrix_sparce, rho0, use_umfpack=True)
        # info = 0 



        if info != 0 :
            print("info:",info)
        # new_eta, info = sparse.linalg.bicg(Tsm_sm_matrix_sparce,rho0)
        
        
        if timing:
            timer_step[3] = time.time()
            total_time = timer_step[-1] - timer_step[0]

            diff_time = np.zeros(timer_step.size)
            diff_time[0] = timer_step[0].copy()

            for i in range(1,timer_step.size):
                diff_time[i] = timer_step[i] - timer_step[i-1]

            timer_step = diff_time / total_time
            print("Rank: 0",
                "Set T sparse:",   "{:.2f}".format(timer_step[1]),
                "Precondition:", "{:.2f}".format(timer_step[2]),
                "Solver:", "{:.2f}".format(timer_step[3]),
                "Total:",   "{:.2f}".format(total_time))

        
        
        return new_eta

    return eta_scipy_sparse_solve,eta_scipy_sparse_solve_jacobi

def load_petsc():
    """
    Load petsc4py
    """
    # ----------------------------------------------------------------------------
    # Try to import petsc4py
    petsc4py_available = False
    try :
        import petsc4py
        petsc4py.init(sys.argv)
        from petsc4py import PETSc
        petsc4py_available = True

        global PETSc

        # OptDB = PETSc.Options()
        # PETSc.Options().setValue('cuda_device', '1')

        mpi_rank = PETSc.COMM_WORLD.getRank()
        mpi_size = PETSc.COMM_WORLD.getSize()
        mpi_comm = PETSc.COMM_WORLD.tompi4py()

    except ImportError:
        mpi_rank = 0
        mpi_size = 1
        print('Petsc4py not available')
    # ----------------------------------------------------------------------------

    # @profile
    def eta_petsc(Tsm_sm_matrix_sp,eta,rho0,gamma,act,M,Lx,Ly,tol,max_iter,ks_type,ps_type,solver,verbose=False,device='cpu'):
        """
        Solve linear system for eta using PETSc
        """
        timing = False
        if timing :
            timer_step = np.zeros(7)
            timer_step[0] = time.time()

        # Set the Matrix type and device
        if device == 'gpu':
            petsc4py.PETSc.Options().setValue('cuda', '1')
            mat_type = PETSc.Mat.Type.AIJCUSPARSE
        else:
            mat_type = PETSc.Mat.Type.AIJ

        mpi_rank = PETSc.COMM_WORLD.getRank()
        mpi_size = PETSc.COMM_WORLD.getSize()
        mpi_comm = PETSc.COMM_WORLD.tompi4py()

        # Compute the solution using scipy sparce iterative method
        if mpi_rank == 0:
            Tsm_sm_matrix = Tsm_sm_matrix_sp.tocsr()
            rho0_sparse = rho0
            eta_sparse = eta.copy()

            if solver.transpose:
                Tsm_sm_matrix_iter = Tsm_sm_matrix.copy().transpose()
            else:
                Tsm_sm_matrix_iter = Tsm_sm_matrix.copy()

            new_eta, iter_success = iterative_solver_sp(Tsm_sm_matrix_iter, eta_sparse, rho0_sparse, gamma, tol, max_iter=max_iter)
        else:
            Tsm_sm_matrix = None
            new_eta = None
            iter_success = None
        
        if mpi_size > 1:
            iter_success = mpi_comm.bcast(iter_success, root=0)
            new_eta = mpi_comm.bcast(new_eta, root=0)
        
        if iter_success:
            return new_eta
        
        # If the iterative method did not converge, use PETSc

        if solver.A is None:
            solver.Tsm_sm_sp_petsc = mpi_comm.bcast(Tsm_sm_matrix, root=0)
            solver.var_list = mpi_comm.bcast([gamma,act,M], root=0)

        gamma,act,M = solver.var_list
        mat_size = M*Lx*Ly

        # broadcast the eta, rho0 and data of Tsm_sm_matrix to all the processors
        if mpi_rank  == 0:
            bcast_list = np.concatenate((eta,rho0,Tsm_sm_matrix.data))
            solver.Tsm_sm_sp_petsc.data = Tsm_sm_matrix.data
        else:
            bcast_list = None
        if mpi_size > 1:
            bcast_list = mpi_comm.bcast(bcast_list, root=0)
            eta = bcast_list[:mat_size]
            rho0 = bcast_list[mat_size:2*mat_size]
            solver.Tsm_sm_sp_petsc.data = bcast_list[2*mat_size:]

        if timing :
            timer_step[1] = time.time()

        # Create PETSc matrix
        if solver.A is None:
        # if True:
            # Get the maximum number of non zeros values per row for Tsm_sm
            count_non_zeros_in_row = solver.Tsm_sm_sp_petsc.getnnz(axis=1)

            # Create PETSc matrix   
            A = PETSc.Mat()
            A.create(comm=PETSc.COMM_WORLD)

            A.setSizes(  ( (PETSc.DECIDE, mat_size), (PETSc.DECIDE, mat_size) ) )
            A.setType(mat_type)
            A.setFromOptions()
            A.setPreallocationNNZ(count_non_zeros_in_row[mpi_rank*mat_size//mpi_size:(mpi_rank+1)*mat_size//mpi_size])
            A.setUp()
            solver.A = A
        else :
            solver.A.zeroEntries()

        # Fill PETSc matrix
        # print('PETSC -->',type(Tsm_sm_matrix))
        Tsm_sm_matrix = solver.Tsm_sm_sp_petsc  
        rstart, rend = solver.A.getOwnershipRange()

        indptr = Tsm_sm_matrix.indptr[rstart:rend+1]-Tsm_sm_matrix.indptr[rstart]
        indices = Tsm_sm_matrix.indices[Tsm_sm_matrix.indptr[rstart]:Tsm_sm_matrix.indptr[rend]]
        data = Tsm_sm_matrix.data[Tsm_sm_matrix.indptr[rstart]:Tsm_sm_matrix.indptr[rend]]

        solver.A.setValuesCSR(indptr, indices, data,addv=False)

        if timing :
            timer_step[2] = time.time()

        solver.A.assemblyBegin()
        solver.A.assemblyEnd()

        # Create PETSc identity matrix with ones
        if solver.ones is None:
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
            ones.assemblyEnd()

            solver.ones = ones

        # Compute: matrix_A =  I - gamma * T
        solver.A.aypx(-gamma,solver.ones)

        if timing :
            timer_step[3] = time.time()

        # print('Matrix A assembled', A.size)
        # print(A.getInfo())
        # print("Options set", A.getOptionsPrefix())

        # Create PETSc vectors x and b
        if solver.x is None and solver.b is None:
            x, b =  solver.A.createVecs()
            solver.x = x
            solver.b = b

        # Fill PETSc vectors x and b
        rstart, rend = solver.b.getOwnershipRange()
        solver.b.setValuesLocal(np.arange(rstart,rend,dtype=np.int32),rho0[rstart:rend])
        # print(eta.shape,eta)
        solver.x.setValuesLocal(np.arange(rstart,rend,dtype=np.int32),eta[rstart:rend])

        solver.b.assemble()
        solver.x.assemble()

        # Start the solver
        if solver.ksp is None:
            ksp = PETSc.KSP().create(comm=solver.A.getComm())
            ksp.setOperators(solver.A)
            ksp.setType(ks_type)

            ksp.getPC().setType(ps_type)
            if tol == -1 :
                ksp.setTolerances(rtol=1.0e-10)
            else:
                ksp.setTolerances(rtol=tol)

            ksp.setFromOptions()

            # print("KSP type:",ksp.getType())
            if ksp.getType() != 'preonly':
                ksp.setInitialGuessNonzero(True)

            ksp.setUp()
            solver.ksp = ksp
        
        solver.ksp.setUp()

        if solver.transpose:
            solver.ksp.solveTranspose(solver.b, solver.x)
        else:
            solver.ksp.solve(solver.b, solver.x)

        # A.destroy()
        # ones.destroy()
        # if timing :
        #     PETSc.Sys.Print(solver.ksp.view())
        # ksp.destroy()
        # b.destroy()

        if timing :
            timer_step[4] = time.time()

        # Collect all the values across the processors 
        scatter, eta = PETSc.Scatter.toZero(solver.x)
        scatter.scatter(solver.x, eta, False, PETSc.Scatter.Mode.FORWARD)
        PETSc.COMM_WORLD.barrier()

        # x.destroy()
        scatter.destroy()
        eta = eta.getArray()

        if not mpi_rank == 0 :
            eta = np.zeros(1)

        if timing :
            timer_step[5] = time.time()

        # broadcast eta to all the processors
        eta = mpi_comm.bcast(eta, root=0)

        if timing and (mpi_rank in [0,1,mpi_size/2,mpi_size-1]):
            timer_step[6] = time.time()
            
            total_time = timer_step[-1] - timer_step[0]

            diff_time = np.zeros(timer_step.size)
            diff_time[0] = timer_step[0]

            for i in range(1,timer_step.size):
                diff_time[i] = timer_step[i] - timer_step[i-1]

            timer_step = diff_time / total_time
            print( "Petsc ---",
                "Rank:",mpi_rank,
                "Get row nnz:","{:.2f}".format(timer_step[1]),
                "Fill A:",     "{:.2f}".format(timer_step[2]),
                "Compute A:",  "{:.2f}".format(timer_step[3]),
                "Solve:",      "{:.2f}".format(timer_step[4]),
                "Scatter:",    "{:.2f}".format(timer_step[5]),
                "Broadcast:",  "{:.2f}".format(timer_step[6]),
                "Total:",      "{:.2f}".format(total_time))

        return eta

    def free_memory(solver):
        """
        Free memory
        """
        if solver.A is not None:
            solver.A.destroy()
            solver.A = None
        if solver.ksp is not None:
            solver.ksp.destroy()
            solver.ksp = None
        if solver.b is not None:
            solver.b.destroy()
            solver.b = None
        if solver.x is not None:
            solver.x.destroy()
            solver.x = None
        if solver.ones is not None:
            solver.ones.destroy()
            solver.ones = None
        if solver.Tsm_sm_sp_petsc is not None:
            solver.Tsm_sm_sp_petsc = None
        if solver.var_list is not None:
            solver.var_list = None

    return eta_petsc, eta_petsc, mpi_rank, mpi_size, mpi_comm, free_memory

def load_cupy():        
    """
    Load cupy and cupyx
    """
    # Try to import cupy
    cupy_scipy_available = False
    try : 
        import cupy as cp
        import cupyx.scipy.sparse as cusparse
        import cupyx.scipy.sparse.linalg
        cupy_scipy_available = True

    except ImportError:
        print('Cupy sparse not available')
    
    eta_petsc = None

    # cp.cuda.Device(0).use()

    # 
    #----------------------------------------------------------------------------
    def eta_cupy_sparse_solve_iter(Tsm_sm_matrix,eta,gamma,M,Lx,Ly,rho0,device='gpu',verbose=False):
        """
        Solve linear system using cupy scipy sparce
        """
        print('Cupy iter')
        timing = False
        if timing :
            timer_step = np.zeros(4)
            timer_step[0] = time.time()

        Tsm_sm_matrix_cupy = cusparse.csr_matrix(Tsm_sm_matrix)

        if timing :
            timer_step[1] = time.time()

        to_invert = cusparse.eye(M*Ly*Lx) - gamma * Tsm_sm_matrix_cupy
        rho0_cupy = cp.asarray(rho0)
        eta_cupy = cp.asarray(eta)

        if timing :
            timer_step[2] = time.time()

        # new_eta_cupy = cusparse.linalg.spsolve(to_invert,rho0_cupy)
        # set the precondioioner as ilu
        M = cusparse.linalg.spilu(to_invert,fill_factor=1)
        # Solve the system with mgres with M precondiotioner
        new_eta_cupy =  cusparse.linalg.gmres(to_invert, rho0_cupy, x0=eta_cupy, tol=1e-08,  M=M)

        new_eta = cp.asnumpy(new_eta_cupy)


        if verbose :
            print("      Tsm_sm size: ", Tsm_sm_matrix_cupy.nnz, " Memory size: ", Tsm_sm_matrix_cupy.data.nbytes/1e6, " MB")
            print("        rho0 size: ", rho0.shape, " Memory size: ", rho0.nbytes/1e6, " MB")
            print("Total memory size: ", (Tsm_sm_matrix_cupy.data.nbytes + rho0.nbytes)/1e6, " MB")
        if timing :
            timer_step[3] = time.time()

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

        return new_eta

    def eta_cupy_sparse_solve(Tsm_sm_matrix, eta, rho0, gamma, M, Lx, Ly, tol, max_iter, device='gpu',verbose=False):
        """
        Solve linear system using cupy scipy sparce
        """

        timing = False
        if timing :
            timer_step = np.zeros(4)
            timer_step[0] = time.time()

        Tsm_sm_matrix_cupy = cusparse.csr_matrix(Tsm_sm_matrix)
        rho0_cupy = cp.asarray(rho0)
        eta_cupy = cp.asarray(eta)

        if timing :
            timer_step[1] = time.time()

        new_eta_cupy, iter_success = iterative_solver_sp(Tsm_sm_matrix_cupy, eta_cupy, rho0_cupy, gamma, tol,max_iter=max_iter)

        if not iter_success:

            to_invert = cusparse.eye(M*Ly*Lx) - gamma * Tsm_sm_matrix_cupy

            if tol == -1 :
            #     new_eta_cupy = cusparse.linalg.spsolve(to_invert,rho0_cupy)
            # else:            
                new_eta_cupy, info =  cusparse.linalg.gmres(to_invert, rho0_cupy, x0=eta_cupy, tol=tol,maxiter=10000)
                if info != 0 :
                    new_eta_cupy = cusparse.linalg.spsolve(to_invert,rho0_cupy)
                    print("info:",info)

        new_eta = cp.asnumpy(new_eta_cupy)

        if timing :
            timer_step[2] = time.time()

        if verbose :
            print("      Tsm_sm size: ", Tsm_sm_matrix_cupy.nnz, " Memory size: ", Tsm_sm_matrix_cupy.data.nbytes/1e6, " MB")
            print("        rho0 size: ", rho0.shape, " Memory size: ", rho0_cupy.nbytes/1e6, " MB")
            print("Total memory size: ", (Tsm_sm_matrix_cupy.data.nbytes + rho0_cupy.nbytes)/1e6, " MB")
        if timing :
            timer_step[3] = time.time()

            total_time = timer_step[-1] - timer_step[0]

            diff_time = np.zeros(timer_step.size)
            diff_time[0] = timer_step[0]

            for i in range(1,timer_step.size):
                diff_time[i] = timer_step[i] - timer_step[i-1]

            timer_step = diff_time / total_time
            print("Cupy | Rank: 0",
                "To csr:","{:.2f}".format(timer_step[1]),
                "Compute A:",  "{:.2f}".format(timer_step[2]),
                "Solve:",      "{:.2f}".format(timer_step[3]),
                "Total:",      "{:.2f}".format(total_time))

        return new_eta

    return eta_cupy_sparse_solve, eta_cupy_sparse_solve_iter

def load_numpy_inv():
    """
    Load numpy inverse
    """
    # Try to import cupy
    numpy_available = False
    try : 
        numpy_available = True

    except ImportError:
        print('Numpy sparse not available')
    
    eta_petsc = None

    # ----------------------------------------------------------------------------
    def eta_numpy_solve(Tsm_sm_matrix, eta, rho0, gamma, M, Lx, Ly, tol, max_iter, device='cpu',verbose=False):
        """
        Solve linear system using cupy scipy sparce
        """

        timing = False
        if timing :
            timer_step = np.zeros(4)
            timer_step[0] = time.time()

        Tsm_sm_matrix_local = Tsm_sm_matrix.toarray()

        if timing :
            timer_step[1] = time.time()

        np.multiply(Tsm_sm_matrix_local,-gamma,out=Tsm_sm_matrix_local)
        for i in range(M*Ly*Lx):
            Tsm_sm_matrix_local[i,i] += 1.0

        if timing :
            timer_step[2] = time.time()

        inverted = np.linalg.inv(Tsm_sm_matrix_local)
        new_eta = inverted @ rho0


        if verbose :
            print("      Tsm_sm size: ", Tsm_sm_matrix_local.shape, " Memory size: ", Tsm_sm_matrix_local.nbytes/1e6, " MB")
            print("        rho0 size: ", rho0.shape, " Memory size: ", rho0.nbytes/1e6, " MB")
            print("    inverted size: ", inverted.shape, " Memory size: ", inverted.nbytes/1e6, " MB")
            print("Total memory size: ", (Tsm_sm_matrix_local.nbytes + rho0.nbytes + inverted.nbytes)/1e6, " MB")

        if timing :
            timer_step[3] = time.time()

            total_time = timer_step[-1] - timer_step[0]

            diff_time = np.zeros(timer_step.size)
            diff_time[0] = timer_step[0]

            for i in range(1,timer_step.size):
                diff_time[i] = timer_step[i] - timer_step[i-1]

            timer_step = diff_time / total_time
            print("Rank:",0,
                "To array:","{:.2f}".format(timer_step[1]),
                "Compute A:",  "{:.2f}".format(timer_step[2]),
                "Solve:",      "{:.2f}".format(timer_step[3]),
                "Total:",      "{:.2f}".format(total_time))

        return new_eta

    return eta_numpy_solve, eta_numpy_solve

def load_numpy():
    """
    Load numpy inverse
    """
    # Try to import cupy
    numpy_available = False
    try : 
        numpy_available = True

    except ImportError:
        print('Numpy sparse not available')
    
    eta_petsc = None

    # ----------------------------------------------------------------------------
    def eta_numpy_solve(Tsm_sm_matrix, eta, rho0, gamma, M, Lx, Ly, tol, max_iter, device='cpu',verbose=False):
        """
        Solve linear system using cupy scipy sparce
        """

        timing = False          
        if timing :
            timer_step = np.zeros(4)
            timer_step[0] = time.time()

        Tsm_sm_matrix_local = Tsm_sm_matrix.toarray()

        if timing :
            timer_step[1] = time.time()

        np.multiply(Tsm_sm_matrix_local,-gamma,out=Tsm_sm_matrix_local)
        for i in range(M*Ly*Lx):
            Tsm_sm_matrix_local[i,i] += 1.0

        if timing :
            timer_step[2] = time.time()

        new_eta = np.linalg.solve(Tsm_sm_matrix_local,rho0)

        if verbose :
            print("      Tsm_sm size: ", Tsm_sm_matrix_local.shape, " Memory size: ", Tsm_sm_matrix_local.nbytes/1e6, " MB")
            print("        rho0 size: ", rho0.shape, " Memory size: ", rho0.nbytes/1e6, " MB")
            print("Total memory size: ", (Tsm_sm_matrix_local.nbytes + rho0.nbytes)/1e6, " MB")

        if timing :
            timer_step[3] = time.time()

            total_time = timer_step[-1] - timer_step[0]

            diff_time = np.zeros(timer_step.size)
            diff_time[0] = timer_step[0]

            for i in range(1,timer_step.size):
                diff_time[i] = timer_step[i] - timer_step[i-1]

            timer_step = diff_time / total_time
            print("Rank:",0,
                "To array:","{:.2f}".format(timer_step[1]),
                "Compute A:",  "{:.2f}".format(timer_step[2]),
                "Solve:",      "{:.2f}".format(timer_step[3]),
                "Total:",      "{:.2f}".format(total_time))

        return new_eta

    return eta_numpy_solve, eta_numpy_solve

def load_torch_inv():
    """
    Load Pytorch inverse
    """
    # Try to import PyTorch
    torch_available = False
    try : 
        import torch as torch
        dtype = torch.double
        if torch.cuda.is_available():
            device_torch = torch.device("cuda:0")
        else:
            device_torch = torch.device("cpu")
    
        torch_available = True

    except ImportError:
        print('Pytorch not available')
    
    eta_petsc = None
    # ----------------------------------------------------------------------------
    def eta_torch(Tsm_sm_matrix, eta, rho0, gamma, M, Lx, Ly, tol, max_iter, device='gpu',verbose=False):
        """
        Invert a matrix using torch
        """
        if device == 'cpu':
            device_torch = torch.device('cpu')
        else:
            device_torch = torch.device('cuda:0')

        try :
            Tsm_sm_matrix_local = Tsm_sm_matrix.toarray()

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
        except Exception as error:
            print("Error inverting matrix with torch")
            print(error)
            Tsm_sm_matrix_torch = None
            rho0_torch = None
            inverted = None
            torch.cuda.empty_cache()
            new_eta = np.zeros(M*Ly*Lx)

        return new_eta
    
    return eta_torch, eta_torch

def load_torch():
    """
    Load Pytorch
    """
    # Try to import PyTorch
    torch_available = False
    try : 
        import torch as torch
        dtype = torch.double
        if torch.cuda.is_available():
            device_torch = torch.device("cuda:0")
        else:
            device_torch = torch.device("cpu")
    
        torch_available = True

    except ImportError:
        print('Pytorch not available')
    
    eta_petsc = None
    # ----------------------------------------------------------------------------
    def eta_torch(Tsm_sm_matrix, eta, rho0, gamma, M, Lx, Ly, tol, max_iter, device='gpu',verbose=False):
        """
        Invert a matrix using torch
        """
        if device == 'cpu':
            device_torch = torch.device('cpu')
        else:
            device_torch = torch.device('cuda:0')

        try :
            Tsm_sm_matrix_local = Tsm_sm_matrix.toarray()

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
                inverted = None
                torch.cuda.empty_cache()
        except Exception as error:
            print("Error inverting matrix with torch")
            print(error)
            Tsm_sm_matrix_torch = None
            rho0_torch = None
            inverted = None
            torch.cuda.empty_cache()
            new_eta = np.zeros(M*Ly*Lx)

        return new_eta
    
    return eta_torch, eta_torch
