# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project High Level Overview

This is **Spara** (also referred to as **groot** in config/naming), an efficient distributed GNN (Graph Neural Network) training system built on top of DGL (Deep Graph Library). 
groot is a multi-gpu parallelization framework, which abstracts communication involved in multi-GPU training. 
It introduces partition-aware sampling and feature loading to accelerate multi-GPU GNN training.

## Important documents and directory structure

1. Build and setup instructions of groot  are in the file @.claude/groot.md
2. We use groot as an external library in our experiments. This requires several utilities which are in the @.claude/experiment.md file. 


# Common setup.

1. We have to be in the project's conda environment before building or running.
   The exact path depends on the machine:

   - RunPod box:      `conda activate /home/groot_env`
   - Unity cluster:   `conda activate /work/pi_huiguan_umass_edu/sandeep/conda_environment/dgl_groot/`

   `build.sh` picks up the CUDA toolkit from whichever env is active
   (`CONDA_PREFIX`). A fresh machine is provisioned once with
   `bash environment_setup.sh`; see @.claude/groot.md for details.