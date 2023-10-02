import torch.multiprocessing as mp

def test_shuffle(proc_id, n_gpus):
    scattered_array_object = ()
    features = ()
    

if __name__== "__main__":
    n_gpus = 4
    procs = []
    # assert(False)
    for proc_id in range(n_gpus):
        p = mp.Process(target=(test_shuffle),
                       args=(proc_id, n_gpus))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

