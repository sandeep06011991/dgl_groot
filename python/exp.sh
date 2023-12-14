# python3 batch_sampling.py --graph_name ogbn-products --pool_size 1 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-products --pool_size 2 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-products --pool_size 4 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-products --pool_size 8 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-products --pool_size 16 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-products --pool_size 32 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-products --pool_size 64 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-products --pool_size 128 --batch_layer=2

# python3 batch_sampling.py --graph_name ogbn-papers100M --pool_size 1 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-papers100M --pool_size 2 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-papers100M --pool_size 4 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-papers100M --pool_size 8 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-papers100M --pool_size 16 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-papers100M --pool_size 32 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-papers100M --pool_size 64 --batch_layer=2
# python3 batch_sampling.py --graph_name ogbn-papers100M --pool_size 128 --batch_layer=2

# python3 dgl_sampling.py --graph_name ogbn-products
# python3 dgl_sampling.py --graph_name ogbn-papers100M
python3 base_sampling.py --graph_name ogbn-products 
python3 base_sampling.py --graph_name ogbn-papers100M