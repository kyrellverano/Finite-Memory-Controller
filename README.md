# Finite-State-Controller
We approach a navigation problem as a memory augmented partially observable Markov decision process but highlighting the fact that we only use a low-dimensional finite memory. These memory states are the finite state controllers in optimizing the search process. In simple words, we derive an optimal search behavior using an optimal strategy of updating memory states. The paper entitled "Olfactory search with finite-state controllers" (https://doi.org/10.1073/pnas.2304230120) provides more details regarding the algorithm in this repository.
## Algorithm
We frame a source-tracking problem as a gridworld where a searcher (agent) navigates through an environment where the odor source (target) emits intermittent and uncorrelated signals. The searcher does not have any knowledge of its spatial position. The agent then uses these signals (odors) as cues to reach the target. The idea is that the more cues an agent receives, the closer it is to the source. Our task is to answer the question: how are these intermittent cues exploited to formulate an efficient strategy to locate an odor source? The set-up is a 2-dimensional grid wherein the target is at a fixed position and the agent starts downwind from the target. The state of the agent is an ordered pair of its x and y positions. Actions of the agent are: left, right, up, down (the code also working for an additional action:stay). From either a model of odor detection or from data, we derive a probability distribution of signals (observations) received given a state. A model of the environment is known: state position of the agent after an action and what is the reward to be received. In the set-up, we set a negative reward for each step until it reaches a target range centered at the source. In this way, maximizing the rewards also minimize the time it takes to reach the target. The policy is how the agent acts given a memory state and observation. It is also embedded here how the agent updates its memory thus it is a mapping from the current memory state, observation received to the action and memory update. This policy is parameterized and we implement a policy gradient to maximize the average rewards given a distribution of signals or observations. 
## Downloading the Files
You have two options: Notebooks or Python Scripts. The first step is to download the folder FSC_NB (Notebooks) or FSC_PS (Python Scripts), depending on your choice. Then, you have to download the FSC_requisites: this contains some data and common codes (utils and fortran codes) that are applicable to both options. After downloading the FSC_requisites, you have to copy all the contents of this to where the codes are: in FSC_NB or FSC_PS, depending on what you have chosen.
## Conda Environment
For both modes (NB or PY), there are some preliminary steps to create an environment which have the required versions of packages. The implementation of the algorithm makes use of a fortran code which needs to be compiled. 
```
conda create -n fmc python=3.8.5
conda activate fmc
conda install numpy=1.19.2
conda install scipy=1.5.2
conda install -c jmcmurray os
conda install -c jmcmurray json
conda install -c jmcmurray matplotlib
``` 
### PETSc installation
This code have the possibility to use PETSc for parallelization (tested with PETSc 3.18.1).
For use PETSc on python, you need to install the module *petsc4py*.
You can follow the instructions here: [petsc4py](https://petsc.org/release/petsc4py/install.html)

If your system has MPI installed, you can install *mpi4py*
```
conda install -c conda-forge mpi4py
```

If your system has PETSc installed, then you can link it to *petsc4py*.
First, you need to set the environment variable `PETSC_DIR` to the directory where PETSc is installed.
After you need the version of PETSc that you have installed.
And then you can install *petsc4py* with the following command:
```
export PETSC_DIR=/path/to/petsc
pip install --user petsc4py==[version of PETSc]
```

**Tip:** If you will use HPC infrastructure, you can load the module of PETSc and the environment variable `PETSC_DIR` will be set automatically. Then you can install *petsc4py*

If you don't have PETSc installed, you can install it with the following command:
We recommend to use the pip installation. 
```
pip install --user mpi4py
export PETSC_CONFIGURE_OPTIONS='--with-cuda=0 --with-debugging=0 --download-superlu --download-superlu_dist --download-parmetis --download-metis --download-ptscotch COPTFLAGS="-O3 " CXXOPTFLAGS="-O3 " FOPTFLAGS="-O3 "
pip install --user petsc # this will take a while
pip install --user petsc4py
```
### Fortran module
Then, go to the directory where your codes are: `cd FSC_requisites` then compile the fortran script by running the following:
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
2. threshold = 5, 12, 15 
3. sub = 0 , 1 [0 gives the best and 1 gives the suboptimal] 

The other variables are in the input.dat.

## Notebooks
The notebooks are prepared for more convenient view of the outputs.

## Python Scripts
Change the input variables in the file input.dat then run the python file that corresponds to the task you want:
1. python3 optimize.py --input_file input.dat
2. python3 visualize.py --dir /path/to/policies

## Optimization options
To see the options for optimization, run the following:
```
python3 optimize.py --help
```

## Visualization options
The outputs (plots, .dat files, or videos) can be seen in the outputs folder.

## Evaluation on Dynamic Plumes and Data Processing 
An additional feature is to be able to evaluate policies given a dynamic environment: for the detecton of signals, instead of using a probabilistic model derived from time-averaged snapshots of the plume, here we use the snapshots of the plume at each time [searcher receive a signal if concentration>threshold].

For this feature, one can download the folder FSC_eval_dataprocess and then add the contents of this to the original/home folder, with all the other requisite codes and data already in there. 

Note: to be able to use the `ProcessFramesFromRawData.ipynb` , one has to use .mat as source files. You can also use data from videos, extracting them first to get the .mat file which contains all pixel values in each frames. 
