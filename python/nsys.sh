nsys profile -o nsys-prof/products_batch_b256_p1 --force-overwrite=true python3 batch_sampling.py --graph_name ogbn-products --pool_size 1 --batch_layer=2
nsys profile -o nsys-prof/products_batch_b256_p2 --force-overwrite=true python3 batch_sampling.py --graph_name ogbn-products --pool_size 2 --batch_layer=2
nsys profile -o nsys-prof/products_batch_b256_p4 --force-overwrite=true python3 batch_sampling.py --graph_name ogbn-products --pool_size 4 --batch_layer=2
nsys profile -o nsys-prof/products_batch_b256_p8 --force-overwrite=true python3 batch_sampling.py --graph_name ogbn-products --pool_size 8 --batch_layer=2
nsys profile -o nsys-prof/products_batch_b256_p16 --force-overwrite=true python3 batch_sampling.py --graph_name ogbn-products --pool_size 16 --batch_layer=2
nsys profile -o nsys-prof/products_batch_b256_p32 --force-overwrite=true python3 batch_sampling.py --graph_name ogbn-products --pool_size 32 --batch_layer=2
nsys profile -o nsys-prof/products_batch_b256_p64 --force-overwrite=true python3 batch_sampling.py --graph_name ogbn-products --pool_size 64 --batch_layer=2
nsys profile -o nsys-prof/products_dgl_b256 --force-overwrite=true python3 dgl_sampling.py --graph_name ogbn-products
nsys profile -o nsys-prof/products_base_b256 --force-overwrite=true python3 base_sampling.py --graph_name ogbn-products