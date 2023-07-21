import dgl
# f = dgl.groot.entry.wrap_function

n = 10
U = []
V = []
for u in range(n):
    for v in range(n):
        if u != v:
            U.append(u)
            V.append(v)

clique_n = dgl.DGLGraph((U,V))
print(clique_n)
# for i in range(10):
#     f()
