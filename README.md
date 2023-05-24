# Finite-State-Controller
We approach a navigation problem as a memory augmented partially observable Markov decision process but highlighting the fact that we only use a low-dimensional finite memory. These memory states are the finite state controllers in optimizing the search process. In simple words, we derive an optimal searching behavior using an optimal strategy of updating memory states.
## Algorithm
We frame a source-tracking problem as a gridworld where: a searcher (agent) navigates through an environment where the odor source (target) emits intermittent and uncorrelated signals. The searcher does not have any knowledge of its spatial position. The agent then uses these signals (odors) as cues to reach the target. The idea is that the more cues an agent receives, the closer it is to the source. Our task is to answer the question: how are these intermittent cues exploited to formulate an efficient strategy to locate an odor source? The set-up is a 2-dimensional grid wherein the target is at a fixed position and the agent starts downwind from the target. The state of the agent is an ordered pair of its x and y positions. Actions of the agent are: left, right, up, down (the code also working for an additional action:stay). From either a model of odor detection or from data, we derive a probability distribution of signals (observations) received given a state. A model of the environment is known: state position of the agent after an action and what is the reward to be received. In the set-up, we set reward=-0.1 for each step until it reaches a target range centered at the source. In this way, maximizing the rewards also minimize the time it takes to reach the target. The \textit{policy} is how the agent acts given a memory state and observation. It is also embedded here how the agent updates its memory thus it is a mapping from the current memory state, observation received to the action and memory update. This policy is parameterized and we implement a policy gradient to maximize the average rewards given a distribution of signals or observations. 
## Conda Environment
You have two options: Notebooks or Python Scripts. The first step is to download the folder FSC_NB (Notebooks) or FSC_PS (Python Scripts), depending on your choice. For both modes, there are some preliminary steps to create an environment which have the required versions of packages. The implementation of the algorithm makes use of a fortran code which needs to be compiled. 
```
conda create -n fmc python=3.8.5
conda activate fmc
conda install numpy=1.19.2
conda install scipy=1.5.2
conda install -c jmcmurray os
conda install -c jmcmurray json
conda install -c jmcmurray matplotlib
``` 
Then, go to the directory where your codes are: `cd FSC_NB` or `cd FSC_PS` then compile the fortran script by running the following:
```
f2py -c fast_sparce_multiplications_2D.f90 -m fast_sparce_multiplications_2D
```
After doing the installation/compilation above, you just need to activate the environment every time you use it again: `conda activate fmc` .
## Tasks
There are two things that you can do with the given code/notebooks:
1. Optimize : run the algorithm to solve the optimized policy by natural gradient descent; the user needs to input all required variables and initializations
2. Visualize : look at some statistics and sample trajectory from saved policies: optimized policies pre-saved and optimized policies from (1) 


## Pre-saved policies
There are two types of policies that are shown here: the best policy we have solved so far and a suboptimal policy that exhibits a different set of behaviors. The following are the variables you can change and the possible values: 
1. coarse = 0 , 1 [0 for non-coarse, 1 for coarse]
2. memories = 1, 2, 3, 4 
3. threshold = 5, 12, 15 
4. sub = 0 , 1 [0 gives the best and 1 gives the suboptimal] 

The other variables are in the input.dat.

## Notebooks
The notebooks are prepared for more convenient view of the outputs. There are two notebooks - depending on the task that is desired. 

## Python Scripts
Change the input variables in the file input.dat then run the python file that corresponds to the task you want:
1. python3 optimize.py input.dat
2. python3 visualize.py input.dat

The outputs (plots, .dat files, or videos) can be seen in the outputs folder.
