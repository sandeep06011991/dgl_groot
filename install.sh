conda install pip 

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit                                                                                                                              
                                                                                                                                  
conda install -c conda-forge gxx==11.3                                                                          
# install pytorch v2.0
c

# install pyg
pip install torch_geometric

# install pyg lib
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# install dependencies
pip install torchmetrics jupyterlab numpy matplotlib pandas ogb
