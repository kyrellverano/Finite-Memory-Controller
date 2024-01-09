subroutine mult_eta_five_2d(pi, eta, pobs, Lx0, Ly0, find_range, Lx, Ly, O, M, neweta)
! ===========================================
! multiplies T eta
! Lx0 and Ly0 are coordinates of source - episode ends.
! ===========================================
implicit none
integer, intent(in) :: Lx, Ly, O, M
real*8, intent(in)  :: pi(O,M,5*M), eta(Lx*Ly*M), pobs(O, M*Lx*Ly), find_range, Lx0, Ly0
real*8, intent(out) :: neweta(Lx*Ly*M)


real*8 :: p_r(M), p_l(M), p_u(M), p_d(M), p_s(M), find_range2
integer :: i, j, k, s, sl, sr, su, sd, mem, L
integer :: A = 5, MA, AL, Lkm1, ix, iy
!real :: s0

neweta = 0.d0
L = Lx*Ly
MA = M*A
AL = A*L

!s0 = Lx0 + (Ly0-1)*Lx

find_range2 = find_range * find_range
  !sx = mod(i -1 , Lx) + 1
  !sy = (i-1) / Lx + 1

do i=1,(L*M)
  ! Here I cycle over s, m states  
  p_l = 0.d0
  p_r = 0.d0
  p_u = 0.d0
  p_d = 0.d0
  p_s = 0.d0

  ! s goes from 1 (if i is 1) to L (if i is L)
  s = mod(i-1,L) +1 

  ! "circular" attraction basin
  ! if inside attraction basin, does not move out -> terminal state.
  ix = mod(s -1 , Lx)
  iy = (s-1) / Lx
  if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 ) then
    
    mem = (i-1) / L + 1
    
    do j=1,O
    ! Here I cycle over possible obs
        p_l = p_l + pi(j,mem,1:MA:A)*pobs( j, i)
        p_r = p_r + pi(j,mem,2:MA:A)*pobs( j, i)
        p_u = p_u + pi(j,mem,3:MA:A)*pobs( j, i)
        p_d = p_d + pi(j,mem,4:MA:A)*pobs( j, i)
        p_s = p_s + pi(j,mem,5:MA:A)*pobs( j, i)
    enddo

    sl = s - 1
    if ( mod(sl, Lx) == 0) sl = s
    sr = s + 1
    if ( mod(sr, Lx) == 1) sr = s
    su = s + Lx
    if ( su > L) su = s
    sd = s - Lx
    if ( sd < 1) sd = s
    
    do k=1,M
        Lkm1 = L*(k-1)
        neweta(sl+Lkm1) = neweta(sl+Lkm1) + eta(i) * p_l(k)  
        neweta(sr+Lkm1) = neweta(sr+Lkm1) + eta(i) * p_r(k)
        neweta(su+Lkm1) = neweta(su+Lkm1) + eta(i) * p_u(k)  
        neweta(sd+Lkm1) = neweta(sd+Lkm1) + eta(i) * p_d(k)
        neweta(s+Lkm1) = neweta(s+Lkm1) + eta(i) * p_s(k)
    enddo
  endif  
enddo

do i=1,(L*M)
    s = mod(i-1,L) +1 
    if ( (mod(s -1 , Lx) + 1 - Lx0)**2 + ((s-1) / Lx + 1 - Ly0)**2  < find_range2 ) then
        do k = 1, M
            neweta(s+L*(k-1)) = 0.d0
        enddo
    endif 
enddo
    
end subroutine

subroutine mult_eta_four_2d(pi, eta, pobs, Lx0, Ly0, find_range, Lx, Ly, O, M, neweta)
! ===========================================
! multiplies T eta
! Lx0 and Ly0 are coordinates of source - episode ends.
! ===========================================
implicit none
integer, intent(in) :: Lx, Ly, O, M
real*8, intent(in)  :: pi(O,M,4*M), eta(Lx*Ly*M), pobs(O, M*Lx*Ly), find_range, Lx0, Ly0
real*8, intent(out) :: neweta(Lx*Ly*M)


real*8 :: p_r(M), p_l(M), p_u(M), p_d(M), find_range2
integer :: i, j, k, s, sl, sr, su, sd, mem, L
integer :: A = 4, MA, AL, Lkm1, ix, iy
!integer :: s0

neweta = 0.d0
L = Lx*Ly
MA = M*A
AL = A*L

!s0 = Lx0 + (Ly0-1)*Lx

find_range2 = find_range * find_range
  !sx = mod(i -1 , Lx) + 1
  !sy = (i-1) / Lx + 1

do i=1,(L*M)
  ! Here I cycle over s, m states  
  p_l = 0.d0
  p_r = 0.d0
  p_u = 0.d0
  p_d = 0.d0

  ! s goes from 1 (if i is 1) to L (if i is L)
  s = mod(i-1,L) +1 

  ! "circular" attraction basin
  ! if inside attraction basin, does not move out -> terminal state.
  ix = mod(s -1 , Lx)
  iy = (s-1) / Lx
  if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 ) then
    
    mem = (i-1) / L + 1
    
    do j=1,O
    ! Here I cycle over possible obs
        p_l = p_l + pi(j,mem,1:MA:A)*pobs( j, i)
        p_r = p_r + pi(j,mem,2:MA:A)*pobs( j, i)
        p_u = p_u + pi(j,mem,3:MA:A)*pobs( j, i)
        p_d = p_d + pi(j,mem,4:MA:A)*pobs( j, i)
    enddo

    sl = s - 1
    if ( mod(sl, Lx) == 0) sl = s
    sr = s + 1
    if ( mod(sr, Lx) == 1) sr = s
    su = s + Lx
    if ( su > L) su = s
    sd = s - Lx
    if ( sd < 1) sd = s
    
    do k=1,M
        Lkm1 = L*(k-1)
        neweta(sl+Lkm1) = neweta(sl+Lkm1) + eta(i) * p_l(k)  
        neweta(sr+Lkm1) = neweta(sr+Lkm1) + eta(i) * p_r(k)
        neweta(su+Lkm1) = neweta(su+Lkm1) + eta(i) * p_u(k)  
        neweta(sd+Lkm1) = neweta(sd+Lkm1) + eta(i) * p_d(k)
    enddo
  endif  
enddo

do i=1,(L*M)
    s = mod(i-1,L) +1 
    ix = mod(s -1 , Lx)
    iy = (s-1) / Lx
    if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  < find_range2 ) then
        do k = 1, M
            neweta(s+L*(k-1)) = 0.d0
        enddo
    endif 
enddo
    
end subroutine

subroutine mult_q_four_2d(pi, Q, pobs, Lx0, Ly0, find_range, Lx, Ly, O, M, newQ)
! ===========================================
! multiplies T Q
! ===========================================
implicit none
real*8, intent(in)  :: pi(O,M,4*M), Q(Lx*Ly*M*4*M), pobs(O, M*Lx*Ly), find_range, Lx0, Ly0
real*8, intent(out) :: newQ(Lx*Ly*M*4*M)
integer, intent(in) :: Lx, Ly, O, M

real*8 :: p_a, find_range2, Qi_pa
integer :: i, j, k, ia, s, mem, act, find_range_int, ix, iy
integer :: SP_A = 4, A,  L, SP_A_mem(4), ALkm1
!integer :: s0

logical :: s_not_found

newQ = 0.d0
A = SP_A*M
L = Lx*Ly

find_range_int = int(find_range) + 2

!s0 = Lx0 + (Ly0-1)*Lx

!open(7)
find_range2 = find_range ** 2

! I am computing 
! Q(s,a) = sum s'a' p(a'|s') Q(s',a')
! The cycle is over s',a'


do i=1,(L*M*A)
! Here I cycle over s', mem, a' states
    
    p_a = 0.d0
    
    act = mod(i-1, A) + 1
    s = mod((i-1)/A, L) + 1
    mem = (i-1)/(L*A) + 1
    
    SP_A_mem(1) = SP_A*(mem-1) + 1
    SP_A_mem(2) = SP_A*(mem-1) + 2
    SP_A_mem(3) = SP_A*(mem-1) + 3
    SP_A_mem(4) = SP_A*(mem-1) + 4
    
    do j=1,O
    ! Here I cycle over possible obs
        p_a = p_a + pi(j,mem,act)*pobs( j, s)
    enddo

    Qi_pa = Q(i) * p_a

    ! From where did I arrive here?
    
    ! First of all, we are now in M=mem, so for sure a' = [A*(mem-1) .. A*mem]
    
    ! First check, if s is already inside the attraction basin.
    ! If it is, there is no contribution to Q, given it has arrived to a terminal state.
    ix = mod(s -1 , Lx)
    iy = (s-1) / Lx
    s_not_found = ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )
    
   ! Q(i) must be zero in terminal states.
    if (s_not_found .eqv. .true.) then
    
        do k=1,M

            ALkm1 = A*L*(k-1)

            if (mod(s-1, Lx)+1 .ne. Lx) then
            ! I arrived coming from left action
            ! a = A*(mem-1) + 1
            ! old s = s + 1
                ix = mod(s, Lx)
                iy = (s) / Lx
    
                if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )  &
                newQ(SP_A_mem(1) + A*s + ALkm1) = &
                    newQ(SP_A_mem(1) + A*s + ALkm1) + Qi_pa  
                
                if (mod(s, Lx) == 1) then
                    !'left special'
                    if ( s_not_found ) newQ(SP_A_mem(1) + A*(s-1) + ALkm1) = &
                        newQ(SP_A_mem(1) + A*(s-1) + ALkm1) + Qi_pa  
                endif
            endif
           

            if (mod(s, Lx) .ne. 1) then
            ! I arrived coming from right action
            ! a = A*(mem-1) + 2
            ! old s = s - 1
                ix = mod(s-2, Lx)
                iy = (s-2) / Lx
    
                if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )  &
                newQ(SP_A_mem(2) + A*(s-2) + ALkm1) = &
                    newQ(SP_A_mem(2) + A*(s-2) + ALkm1) + Qi_pa  
                
                if (mod(s, Lx) == 0) then
                    !'right special'            
                    if ( s_not_found ) newQ(SP_A_mem(2) + A*(s-1) + ALkm1) = &
                        newQ(SP_A_mem(2) + A*(s-1) + ALkm1) + Qi_pa  
                endif
            endif
        
            if (s > Lx) then
            ! If I arrived coming from up action
            ! a = A*(mem-1) + 3
            ! old s = s - Lx
           
                !write(7,*), 'up', i, s,  SP_A*(mem-1) + 3 + A*(s - Lx - 1) + ALkm1
                ix = mod(s-1-Lx, Lx)
                iy = (s-Lx-1) / Lx
    
                if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )  &
                newQ(SP_A_mem(3) + A*(s - Lx - 1) + ALkm1) = &
                    newQ(SP_A_mem(3) + A*(s - Lx - 1) + ALkm1) + Qi_pa  
                if (s > L - Lx) then
                    if (s_not_found) newQ(SP_A_mem(3) + A*(s-1) + ALkm1) = &
                        newQ(SP_A_mem(3) + A*(s-1) + ALkm1) + Qi_pa  
                endif
            endif

            if (s < L - Lx + 1) then
            ! If I arrived coming from down action
            ! a = A*(mem-1) + 1
            ! old _s = s + Lx
           
                !write(7,*), 'down', i, s,  SP_A*(mem-1) + 4 + A*(s + Lx - 1) + ALkm1
                ix = mod(s-1+Lx, Lx)
                iy = (s+Lx-1) / Lx
    
                if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )  &
                newQ(SP_A_mem(4) + A*(s + Lx - 1) + ALkm1) = &
                    newQ(SP_A_mem(4) + A*(s + Lx - 1) + ALkm1) + Qi_pa  
                if (s < Lx+1) then
                    if (s_not_found) newQ(SP_A_mem(4) + A*(s-1) + ALkm1) = &
                        newQ(SP_A_mem(4) + A*(s-1) + ALkm1) + Qi_pa  
                endif
            endif
        
            
        
        enddo
    endif
    
enddo

!do i=1,(L*M*A)
 !Here I cycle over s', mem, a' states
!    s = mod((i-1)/A, L) + 1
    
!    if (((mod(s-1,Lx)+1 - Lx0)**2 + ((s-1)/Lx+1 - Ly0)**2) .lt. find_range2) newQ(i) = 0
!enddo
!close(7)

do i=-find_range_int, find_range_int
    do j = -find_range_int, find_range_int
    
        if ( (i)**2 + (j)**2  < find_range2 ) then
 
            do ia = 1, A
                s = (((j+int(Ly0)-1)*Lx + (i+int(Lx0)) - 1)*A + ia)
                do k = 1, M
                    ALkm1 = A*L*(k-1)
                    newQ(s+ALkm1) = 0.d0
                enddo
            enddo
        endif 
    
    enddo
enddo

end subroutine

subroutine mult_q_five_2d(pi, Q, pobs, Lx0, Ly0, find_range, Lx, Ly, O, M, newQ)
! ===========================================
! multiplies T Q
! ===========================================
implicit none
real*8, intent(in)  :: pi(O,M,5*M), Q(Lx*Ly*M*5*M), pobs(O, M*Lx*Ly), find_range, Lx0, Ly0
real*8, intent(out) :: newQ(Lx*Ly*M*5*M)
integer, intent(in) :: Lx, Ly, O, M

real*8 :: p_a, find_range2, Qi_pa
integer :: i, j, k, ia, s, mem, act, find_range_int, ix, iy
integer :: SP_A = 5, A,  L, SP_A_mem(5), ALkm1
!integer :: s0

logical :: s_not_found

newQ = 0.d0
A = SP_A*M
L = Lx*Ly

find_range_int = int(find_range) + 2

!s0 = Lx0 + (Ly0-1)*Lx

!open(7)
find_range2 = find_range ** 2

! I am computing 
! Q(s,a) = sum s'a' p(a'|s') Q(s',a')
! The cycle is over s',a'


do i=1,(L*M*A)
! Here I cycle over s', mem, a' states
    
    p_a = 0.d0
    
    act = mod(i-1, A) + 1
    s = mod((i-1)/A, L) + 1
    mem = (i-1)/(L*A) + 1
    
    SP_A_mem(1) = SP_A*(mem-1) + 1
    SP_A_mem(2) = SP_A*(mem-1) + 2
    SP_A_mem(3) = SP_A*(mem-1) + 3
    SP_A_mem(4) = SP_A*(mem-1) + 4
    SP_A_mem(5) = SP_A*(mem-1) + 5
    
    do j=1,O
    ! Here I cycle over possible obs
        p_a = p_a + pi(j,mem,act)*pobs( j, s)
    enddo

    Qi_pa = Q(i) * p_a

    ! From where did I arrive here?
    
    ! First of all, we are now in M=mem, so for sure a' = [A*(mem-1) .. A*mem]
    
    ! First check, if s is already inside the attraction basin.
    ! If it is, there is no contribution to Q, given it has arrived to a terminal state.
    ix = mod(s -1 , Lx)
    iy = (s-1) / Lx
    s_not_found = ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )
    
   ! Q(i) must be zero in terminal states.
    if (s_not_found .eqv. .true.) then
    
        do k=1,M

            ALkm1 = A*L*(k-1)

            if (mod(s-1, Lx)+1 .ne. Lx) then
            ! I arrived coming from left action
            ! a = A*(mem-1) + 1
            ! old s = s + 1
                ix = mod(s, Lx)
                iy = (s) / Lx
    
                if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )  &
                newQ(SP_A_mem(1) + A*s + ALkm1) = &
                    newQ(SP_A_mem(1) + A*s + ALkm1) + Qi_pa  
                
                if (mod(s, Lx) == 1) then
                    !'left special'
                    if ( s_not_found ) newQ(SP_A_mem(1) + A*(s-1) + ALkm1) = &
                        newQ(SP_A_mem(1) + A*(s-1) + ALkm1) + Qi_pa  
                endif
            endif
           

            if (mod(s, Lx) .ne. 1) then
            ! I arrived coming from right action
            ! a = A*(mem-1) + 2
            ! old s = s - 1
                ix = mod(s-2, Lx)
                iy = (s-2) / Lx
    
                if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )  &
                newQ(SP_A_mem(2) + A*(s-2) + ALkm1) = &
                    newQ(SP_A_mem(2) + A*(s-2) + ALkm1) + Qi_pa  
                
                if (mod(s, Lx) == 0) then
                    !'right special'            
                    if ( s_not_found ) newQ(SP_A_mem(2) + A*(s-1) + ALkm1) = &
                        newQ(SP_A_mem(2) + A*(s-1) + ALkm1) + Qi_pa  
                endif
            endif
        
            if (s > Lx) then
            ! If I arrived coming from up action
            ! a = A*(mem-1) + 3
            ! old s = s - Lx
           
                !write(7,*), 'up', i, s,  SP_A*(mem-1) + 3 + A*(s - Lx - 1) + ALkm1
                ix = mod(s-1-Lx, Lx)
                iy = (s-Lx-1) / Lx
    
                if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )  &
                newQ(SP_A_mem(3) + A*(s - Lx - 1) + ALkm1) = &
                    newQ(SP_A_mem(3) + A*(s - Lx - 1) + ALkm1) + Qi_pa  
                if (s > L - Lx) then
                    if (s_not_found) newQ(SP_A_mem(3) + A*(s-1) + ALkm1) = &
                        newQ(SP_A_mem(3) + A*(s-1) + ALkm1) + Qi_pa  
                endif
            endif

            if (s < L - Lx + 1) then
            ! If I arrived coming from down action
            ! a = A*(mem-1) + 4
            ! old _s = s + Lx
           
                !write(7,*), 'down', i, s,  SP_A*(mem-1) + 4 + A*(s + Lx - 1) + ALkm1
                ix = mod(s-1+Lx, Lx)
                iy = (s+Lx-1) / Lx
    
                if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )  &
                newQ(SP_A_mem(4) + A*(s + Lx - 1) + ALkm1) = &
                    newQ(SP_A_mem(4) + A*(s + Lx - 1) + ALkm1) + Qi_pa  
                if (s < Lx+1) then
                    if (s_not_found) newQ(SP_A_mem(4) + A*(s-1) + ALkm1) = &
                        newQ(SP_A_mem(4) + A*(s-1) + ALkm1) + Qi_pa  
                endif
            endif
        
            ! if (s < L - Lx + 1) then
            ! If I arrived coming from stay action
            ! a = A*(mem-1) + 5
            ! old _s = s 
            ix = mod(s -1 , Lx)
            iy = (s-1) / Lx
            if ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )  &
            newQ(SP_A_mem(5) + A*(s-1) + ALkm1) = &
                newQ(SP_A_mem(5) + A*(s-1) + ALkm1) + Qi_pa  
            ! endif           
        
        enddo
    endif
    
enddo

!do i=1,(L*M*A)
 !Here I cycle over s', mem, a' states
!    s = mod((i-1)/A, L) + 1
    
!    if (((mod(s-1,Lx)+1 - Lx0)**2 + ((s-1)/Lx+1 - Ly0)**2) .lt. find_range2) newQ(i) = 0
!enddo
!close(7)

do i=-find_range_int, find_range_int
    do j = -find_range_int, find_range_int
    
        if ( (i+int(Lx0)-Lx0)**2 + (j+int(Ly0)-Ly0)**2  < find_range2 ) then
 
            do ia = 1, A
                s = (((j+int(Ly0)-1)*Lx + (i+int(Lx0)) - 1)*A + ia)
                do k = 1, M
                    ALkm1 = A*L*(k-1)
                    newQ(s+ALkm1) = 0.d0
                enddo
            enddo
        endif 
    
    enddo
enddo


end subroutine


subroutine rewards_four_2d(M, Lx, Ly, Lx0, Ly0, find_range, cost_move, reward_find, maxO, RR)
! ===========================================
! multiplies T eta0
! ===========================================
implicit none
real*8, intent(in)  :: cost_move, reward_find, find_range, Lx0, Ly0
real*8, intent(out) :: RR(Lx*Ly*M*4*M)
integer, intent(in) :: Lx, Ly, M, maxO

integer :: i, k, s, new_s, act
integer :: A, L, spa(4), ix, iy!, s0
real*8 :: r, find_range2

logical :: s_not_found

A = 4*M
RR = 0.d0
L = Lx * Ly
!s0 = Lx0 + (Ly0-1)*Lx


spa(1) = -1
spa(2) = +1
spa(3) = +Lx
spa(4) = -Lx

find_range2 = find_range * find_range

! --------------------------------------
! HERE I HAD REWARDS FOR ARRIVING THERE 
! --------------------------------------

do i=1,(L*A)
! Here I cycle over s, a states -> memory is free
    
    ! act is spatial act only - memory is irrelevant for questions of reward
    act = mod(i-1, 4) + 1
    s = mod((i-1)/A, L) + 1
    
    !Where do I go
    new_s = s + spa(act)
    if (new_s<1) new_s = s
    if (new_s>L) new_s = s

    r = 0.d0
    ! If outside the source
    
    ix = mod(s -1 , Lx)
    iy = (s-1) / Lx
    s_not_found = ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )
    if (s_not_found) then
        r = 0.d0 - cost_move
        ! The source has been found!
        ix = mod(new_s -1 , Lx)
        iy = (new_s-1) / Lx
        if ( (ix + 1 - Lx0)**2 + (iy + 1 - Ly0)**2 < find_range2) &
            r = r + reward_find
    endif

    do k=1,M
        RR(i+A*L*(k-1)) = r    
       !write(8,*), 'act', act, 's', s, 'new_s', new_s, 'r', r, 'new_i', i+A*L*(k-1), 'L*A*M', L*2*M*M
    enddo

enddo

end subroutine

subroutine rewards_five_2d(M, Lx, Ly, Lx0, Ly0, find_range, cost_move, reward_find, maxO, RR)
! ===========================================
! multiplies T eta0
! ===========================================
implicit none
real*8, intent(in)  :: cost_move, reward_find, find_range, Lx0, Ly0
real*8, intent(out) :: RR(Lx*Ly*M*5*M)
integer, intent(in) :: Lx, Ly, M, maxO

integer :: i, k, s, new_s, act
integer :: A, L, spa(5), ix, iy!, s0
real*8 :: r, find_range2

logical :: s_not_found

A = 5*M
RR = 0.d0
L = Lx * Ly
!s0 = Lx0 + (Ly0-1)*Lx


spa(1) = -1
spa(2) = +1
spa(3) = +Lx
spa(4) = -Lx
spa(5) = 0

find_range2 = find_range * find_range

! --------------------------------------
! HERE I HAD REWARDS FOR ARRIVING THERE 
! --------------------------------------

do i=1,(L*A)
! Here I cycle over s, a states -> memory is free
    
    ! act is spatial act only - memory is irrelevant for questions of reward
    act = mod(i-1, 4) + 1
    s = mod((i-1)/A, L) + 1
    
    !Where do I go
    new_s = s + spa(act)
    if (new_s<1) new_s = s
    if (new_s>L) new_s = s

    r = 0.d0
    ! If outside the source
    
    ix = mod(s -1 , Lx)
    iy = (s-1) / Lx
    s_not_found = ( ( ix + 1 - Lx0)**2 + ( iy + 1 - Ly0)**2  .ge. find_range2 )
    if (s_not_found) then
        r = 0.d0 - cost_move
        ! The source has been found!
        ix = mod(new_s -1 , Lx)
        iy = (new_s-1) / Lx
        if ( (ix + 1 - Lx0)**2 + (iy + 1 - Ly0)**2 < find_range2) &
            r = r + reward_find
    endif

    do k=1,M
        RR(i+A*L*(k-1)) = r    
       !write(8,*), 'act', act, 's', s, 'new_s', new_s, 'r', r, 'new_i', i+A*L*(k-1), 'L*A*M', L*2*M*M
    enddo

enddo


end subroutine
