import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from training import train_loop


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method='file:///f:/libtmp/some_file'
# dist.init_process_group(
#    'gloo',
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '3630'

    # initialize the process group
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(local_rank, node_rank, world_size):
    print(f'Running basic DDP example on machine: {node_rank} in gpu: {local_rank}.')
    setup(local_rank, world_size)

    train_loop.run_training_loop(local_rank=local_rank)

    cleanup()

def run_demo(demo_fn, gpus_per_node, node_rank, world_size):
    mp.spawn(demo_fn, args=(node_rank, world_size,), nprocs=gpus_per_node, join=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_per_node', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    args = parser.parse_args()
    world_size = args.gpu_per_node * args.nodes
    run_demo(demo_basic, args.gpu_per_node, args.node_rank, world_size)
