# Step 1 get full profile
GRAPH_NAME=ogbn
nsys profile --trace-fork-before-exec true -t cuda,nvtx -o pr\
  --force-overwrite true  python3 groot_run.py --graph ogbn-products

#step 2 collect metrics
#nsys stats --help-reports
#nsys recipe nvtx_gpu_proj_sum --help

nsys stats --report nvtx_sum pr.nsys-rep
