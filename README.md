# Finite-State-Controller
## Algorithm
## Instructions
You have two otpions: Notebooks or Python Scripts. The first step is to download the folder FSC_NB (Notebooks) or FSC_PS (Python Scripts), depending on your choice. For both modes, there are some preliminary steps. The implementation of the algorithm makes use of a fortran code for faster computations in some operations needed. Hence, one need to run the following in the command prompt in the folder where the notebook is run or where the python script will be run:
(instructions on creating environment - conda or without conda)
## Tasks
There are three different things that you can do with the given code:
1. Optimize : run the algorithm to solve the optimized policy by natural gradient descent; the user needs to input all required variables and initializations
2. Visualize : look at some statistics and sample trajectory from saved policies: optimized policies pre-saved and optimized policies from (1) 
3. TimeDistribution : evaluate saved policies for a large number of trajectories and plot the time distributions

## Pre-saved policies
There are two types of policies that are shown here: the best policy we have solved so far and a suboptimal policy that exhibits a different set of behaviors. The following are the variables you can change and teh possible values: 
1. coarse = 0 , 1 [0 for non-coarse, 1 for coarse]
2. memories = 1, 2, 3, 4 
3. threshold = 5, 12, 15 
4. sub = 0 , 1 [0 gives the best and 1 gives the suboptimal] 

The other variables are in the default_input.

## Notebooks
The notebooks are easily prepared and the instructions for inputs are explained there. There are three notebooks - depending on the task that is desired. All outputs will be automatically saved on outputs/NB.

## Python Scripts
Change the input variables in the file input.dat then run the python file that corresponds to the task you want:
1. python3 optimize.py input.dat
2. python3 visualize.py input.dat
3. python3 timedist.py input.dat
The outputs (plots, .dat files, or videos) can be seen in the outputs folder.
