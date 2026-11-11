"""gsplit_pipepar -- pipeline + split training driver (Plan 2).

A DRIVER-ONLY variant of the split system (the model is NOT modified): each
optimizer step consumes TWO micro-batches (each HALF of config.batch_size, so a
step trains on the same number of seeds as the split baseline) run on TWO CUDA streams with a
one-stage-staggered schedule, so one micro-batch's split shuffle (all-to-all
comm) overlaps the other's conv (compute). Exactly one shuffle is in flight at a
time -> the single gsplit NCCL communicator is sufficient and cannot deadlock.
The two backward passes accumulate into one gradient; a single optimizer step is
taken per pair.

Reuses the split data path verbatim (SplitGraphLoader, get_distributed_model,
the per-layer Shuffle). Gradients are reduced manually (all_reduce mean) rather
than via DDP, because the staggered schedule drives the conv layers directly.

Caveat 1 (single global batch in the C++ sampler): both micro-batches are sampled
AND their features fetched up front (while each batch_id is current), so the
per-batch blocks/scattered-arrays/features are independent before any pipelined
compute. Features are cloned to survive the next GetFeature call.
"""
import torch
import torchmetrics.functional as MF
import torch.distributed as dist
from torch.multiprocessing import spawn
from utils import *
from dgl.dev.splitloader import *
from dgl.dev.splitmodel import get_distributed_model, Shuffle
from dgl.dev import CudaProfilerStart, CudaProfilerStop

# micro-batch ids within a pipelined step
MB_A, MB_B = 0, 1


class TwoStreamPipeline:
    """Runs the two micro-batches through the split model's layers with a
    one-stage-staggered schedule, overlapping one micro-batch's shuffle (comm)
    with the other's conv (compute):

        shuffle(A,0) | conv(A,0)‖shuffle(B,0) | conv(B,0)‖shuffle(A,1) | ...

    All shuffles are issued in one fixed order on a dedicated comm stream, so
    exactly one all-to-all is ever in flight -- the single gsplit NCCL
    communicator is enough and cannot deadlock. Each conv runs on its
    micro-batch's own compute stream and overlaps the next shuffle; CUDA events
    carry the cross-stream data dependencies. Streams are created once and reused
    across steps.
    """

    def __init__(self, model):
        self.layers = model.layers
        self.activation = model.activation
        self.num_layers = len(model.layers)
        self.rank = model.rank
        self.world_size = model.world_size
        self.comm_stream = torch.cuda.Stream()
        self.compute_streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    def run(self, blocks_pair, feat_pair):
        """blocks_pair / feat_pair are (micro-batch-A, micro-batch-B). Returns the
        two output tensors with their autograd graphs intact."""
        # per-step state (one pipelined step runs at a time, so storing it on self
        # is safe and keeps the schedule below free of nested closures)
        self._blocks = list(blocks_pair)
        self._layer_input = list(feat_pair)   # current tensor per micro-batch (0=A, 1=B)
        self._shuffled = {}                    # (mb, layer) -> shuffled input to that layer
        self._shuffle_done = {}                # (mb, layer) -> event: shuffle finished
        self._conv_done = {}                   # (mb, layer) -> event: conv finished

        self._issue_shuffle(MB_A, 0, wait_event=None)
        for layer in range(self.num_layers):
            self._issue_conv(MB_A, layer)                               # conv(A,i) ‖ shuffle(B,i)
            self._issue_shuffle(MB_B, layer, self._conv_done.get((MB_B, layer - 1)))
            self._issue_conv(MB_B, layer)                               # conv(B,i) ‖ shuffle(A,i+1)
            if layer + 1 < self.num_layers:
                self._issue_shuffle(MB_A, layer + 1, self._conv_done[(MB_A, layer)])

        current = torch.cuda.current_stream()
        current.wait_event(self._conv_done[(MB_A, self.num_layers - 1)])
        current.wait_event(self._conv_done[(MB_B, self.num_layers - 1)])
        torch.cuda.synchronize()
        return self._layer_input[MB_A], self._layer_input[MB_B]

    def _issue_shuffle(self, mb, layer, wait_event):
        """Run this micro-batch's layer-`layer` shuffle on the comm stream (after
        `wait_event`, the producing conv), and record its completion event."""
        if wait_event is not None:
            self.comm_stream.wait_event(wait_event)
        with torch.cuda.stream(self.comm_stream):
            self._shuffled[(mb, layer)] = Shuffle.apply(
                self._blocks[mb][layer].scattered_src, self._layer_input[mb],
                self.rank, self.world_size)
        done = torch.cuda.Event(); done.record(self.comm_stream)
        self._shuffle_done[(mb, layer)] = done

    def _issue_conv(self, mb, layer):
        """Run this micro-batch's layer-`layer` conv on its own compute stream
        (after its shuffle), and record its completion event."""
        stream = self.compute_streams[mb]
        stream.wait_event(self._shuffle_done[(mb, layer)])
        with torch.cuda.stream(stream):
            x = self.layers[layer](self._blocks[mb][layer], self._shuffled[(mb, layer)])
            if layer != self.num_layers - 1:
                x = self.activation(x)
            if x.dim() == 3:                   # GAT returns (N, heads, feat)
                x = x.flatten(-2)
        self._layer_input[mb] = x
        done = torch.cuda.Event(); done.record(stream)
        self._conv_done[(mb, layer)] = done


def bench_pipeline_split(config: Config):
    graph, feat, label, train_idx, valid_idx, test_idx, num_label = load_data(config, is_pinned=True)
    config.num_classes = num_label
    partition_map = None
    if "random" not in config.partition_type:
        partition_map = load_partition_map(config)
    else:
        print(f"using random with {config.num_partition} partitions")
        v_num = graph.num_nodes()
        partition_map = torch.randint(low=0, high=config.num_partition, size=(v_num,), dtype=torch.int8)
    try:
        unique_id = GetUniqueId()
        spawn(train_pipeline_split_ddp,
              args=(config, unique_id, graph, feat, label, train_idx, test_idx, partition_map),
              nprocs=config.world_size)
    except Exception as e:
        print(f"error encountered with {config=}:", e)
        exit(-1)


def train_pipeline_split_ddp(rank, config, unique_id, graph, feat, label,
                             train_idx, test_idx, partition_map):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    if rank == 0:
        print(config)
    graph = graph.pin_memory_()
    feat_handle = pin_memory_inplace(feat)
    label = label.to(device)
    assert partition_map.shape[0] == graph.num_nodes()

    model = get_distributed_model(rank, config.world_size, config.num_redundant_layer,
                                  config.model, feat.shape[1], len(config.fanouts),
                                  config.hid_size, config.num_classes).to(device)
    for param in model.parameters():                  # replicate initial weights
        dist.broadcast(param.data, src=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Each optimizer step consumes TWO micro-batches (A, B), so each micro-batch holds HALF
    # of the mini-batch: a step then trains on config.batch_size seeds per GPU, the same as
    # the split baseline (same nodes per optimizer step across GPUs, same live memory, same
    # number of optimizer steps per epoch). config.batch_size stays the per-step mini-batch
    # and is what gets logged. (Before this fix each micro-batch was a full batch_size,
    # i.e. 2x the seeds per step -- those sweep rows are flagged in gen_hybrid_table.py.)
    assert config.batch_size % 2 == 0, \
        f"gsplit_pipepar needs an even batch_size to split into 2 micro-batches, got {config.batch_size}"
    micro_batch_size = config.batch_size // 2
    sample_config = SampleConfig(rank=rank, num_partition=config.num_partition,
                                 batch_size=micro_batch_size, world_size=config.world_size,
                                 mode="uva", fanouts=config.fanouts, reindex=True, drop_last=True)
    dataloader = SplitGraphLoader(graph, partition_map, train_idx, unique_id, sample_config)

    local_ids = torch.empty([], dtype=torch.int64)
    if "G" in config.cache_size:
        ids = torch.arange(0, partition_map.shape[0])
        local_ids = ids[partition_map == rank]
        size = int(config.cache_size.removesuffix("G")) * (1024 ** 3)
        num_ids_cached = min(size // (feat.shape[1] * 4), local_ids.shape[0])
        local_ids = local_ids[:num_ids_cached].clone()
    dataloader.init_featloader(feat, local_ids)

    pipeline = TwoStreamPipeline(model)
    step = 0
    step_per_epoch = dataloader.max_step_per_epoch // 2   # loader yields micro-batches; a step = 2
    dist.barrier()
    CudaProfilerStart()
    timer = Timer()
    sampling_timers, feature_timers, forward_timers, backward_timers = [], [], [], []
    edges_computed = []

    for epoch in range(config.num_epoch):
        edges_computed_epoch = 0
        model.train()
        batches = iter(dataloader)
        sampling_timer = CudaTimer()
        while True:
            # sample + fetch BOTH micro-batches up front (caveat 1: the C++ sampler
            # keeps one _batch; fetch each feature while its batch_id is current, and
            # clone it so it survives the next GetFeature call).
            try:
                _, seeds_a, blocks_a = next(batches)
            except StopIteration:
                break
            feat_a = dataloader.get_feature().clone()
            labels_a = label[seeds_a]
            try:
                _, seeds_b, blocks_b = next(batches)
            except StopIteration:
                break                                 # drop the odd trailing micro-batch
            feat_b = dataloader.get_feature().clone()
            labels_b = label[seeds_b]
            step += 1
            sampling_timer.end(); sampling_timers.append(sampling_timer)
            for block in blocks_a + blocks_b:
                edges_computed_epoch += block.num_edges()

            # staggered compute‖comm forward of the two micro-batches
            forward_timer = CudaTimer()
            pred_a, pred_b = pipeline.run((blocks_a, blocks_b), (feat_a, feat_b))
            forward_timer.end(); forward_timers.append(forward_timer)

            # one accumulator over A+B (combined-batch mean), reduce once, single step
            backward_timer = CudaTimer()
            optimizer.zero_grad()
            (0.5 * torch.nn.functional.cross_entropy(pred_a, labels_a)).backward()
            (0.5 * torch.nn.functional.cross_entropy(pred_b, labels_b)).backward()
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= config.world_size
            backward_timer.end(); backward_timers.append(backward_timer)
            optimizer.step()
            sampling_timer = CudaTimer()
            log_step(rank, epoch, step, step_per_epoch, timer)
        edges_computed.append(edges_computed_epoch)

    dist.barrier()
    CudaProfilerStop()
    duration = timer.duration()
    sampling_time = get_duration(sampling_timers)
    feature_time = get_duration(feature_timers) if feature_timers else 0
    forward_time = get_duration(forward_timers)
    backward_time = get_duration(backward_timers)

    acc = 0
    if config.graph_name in ["products", "arxiv", "papers100M"]:
        try:
            t2 = Timer()
            print(f"testing model accuracy on {device}")
            model.eval()
            ys, y_hats = [], []
            dataloader.set_target_idx(test_idx)
            for input_nodes, output_nodes, blocks in dataloader:
                with torch.no_grad():
                    batch_feat = dataloader.get_feature()
                    ys.append(label[output_nodes])
                    y_hats.append(model(blocks, batch_feat))
            acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass",
                              num_classes=config.num_classes)
            acc = round(acc.item() * 100, 2)
            print(f"[rank {rank}] test accuracy={acc}% in {t2.duration()} secs")
        except Exception as e:
            print(f"WARN accuracy eval failed on rank {rank}: {e}")
            acc = 0

    if rank == 0:
        profiler = Profiler(duration=duration, sampling_time=sampling_time,
                            feature_time=feature_time, forward_time=forward_time,
                            backward_time=backward_time, test_acc=acc)
        profile_edge_skew(edges_computed, profiler, rank)
        write_to_csv(config.log_path, [config], [profiler])
    ddp_exit()
