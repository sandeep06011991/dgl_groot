## Required Tests
Phase 1: 
	FOCUS ONLY ON Slicing sample while considering 
	only WORKLOAD MAPPING
	Clear all experiments with 10 CLIQUE 

1. Shuffle of tensors using current communicator

	tensor[gpu_id] = [1,1,1,1] * gpu_id
	after shuffle 
	tensor[gpu_id] = [1,2,3,4] for all gpus

2. Tensor Slicing to Scatter Array
	tensor = [1,2,3,5,6,7,8]
	partition_map = [0,1,2,0,3,1,1]
	slice = [1,5,2,7,8,3,6]
	inverse = [0,3,
	offsets = [2,5,6,7]

3. Write a pseudo code for single layer of  src_to_dest and dest_to_src
	Assume a partition book mapping
	## SRC -> DEST
	target_nodes = seed_nodes
	for i in range(layers == 1):
		sampled_graph = g.sample_neighbours([target_nodes])	
		create_block(sampled_graph)
		scattered f = ScatterArray(frontier)
		target_nodes = shuffle(f.slice, f.offsets)
	## DEST -> SRC
	sampled_graph = g.sample_neighbours([target_nodes])
	src_nodes, dest_nodes = sampled_graph.coo()
	scattered f = ScatteredArray((src_nodes, dest_nodes))
	 	

4.
	a.  Put the Graph on GPU and sample 
	b.  fill up one layer of the 3 pseudo code.


################# CACHING #####################
1. Fill up feature cache of gpu, Review the feature storage code that DS wrote. 
2. Partition the feature vector across the gpus 
3. Write three layer GNN sampling 

################# TRAINING ####################
1. Hardcode a two layer forward block with shuffle function
:x

