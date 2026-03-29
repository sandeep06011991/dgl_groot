# Overview of module. 

This package runs a single gpu sampler, and model of with varying hidden dimensions to appropriately estimate the overhead of different forms of parallelism. 

For Ex. To evaluate the correct point of hybrid parallelism. 
Imagine a papers and a 10 layer model with fanout of 2. 
and batch size of 256. 

Option 1. Strict data parallelism would randomly iterate through the training nodes. 


Option 2. data parallelism and switch in the bottom layer. 

