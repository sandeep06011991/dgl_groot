# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project High Level Overview

This is **Spara** (also referred to as **groot** in config/naming), an efficient distributed GNN (Graph Neural Network) training system built on top of DGL (Deep Graph Library). 
groot is a multi-gpu parallelization framework, which abstracts communication involved in multi-GPU training. 
It introduces partition-aware sampling and feature loading to accelerate multi-GPU GNN training.

## Important documents and directory structure

1. Build and setup instructions of groot  are in the file @groot.md
2. We use groot as an external library in our experiments. This requires several utilities which are in the @experiment.md file. 


# Common setup.

1. We have to be in the conda environment 

conda activate /work/pi_huiguan_umass_edu/sandeep/conda_environment/dgl_groot/