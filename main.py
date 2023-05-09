import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from training import train_loop
import socket
import time


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
    if os.getenv('MASTER_ADDR') is None:
        print('WARNING: could not find master address, setting it to local machine. if you are using multiple '
              'machines, this is an error')
        os.environ['MASTER_ADDR'] = 'localhost'

    if os.getenv('MASTER_PORT') is None:
        os.environ['MASTER_PORT'] = '3630'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def setup_and_run_single_process_train_code(local_rank, node_rank, gpus_per_node, world_size, train_loop_kwargs):
    print(f'Running basic DDP example on machine: {node_rank} in gpu: {local_rank}.')
    global_rnak = gpus_per_node * node_rank + local_rank
    setup(global_rnak, world_size)
    print(f'Starting training loop')
    try:
        train_loop.run_training_loop(local_rank=local_rank, global_rnak=global_rnak, **train_loop_kwargs)
    finally:
        cleanup()

def spawn_processes_for_this_node(setup_and_run_single_process_train_code, gpus_per_node, node_rank, world_size,
                                  **train_loop_kwargs):
    mp.spawn(setup_and_run_single_process_train_code, args=(node_rank, gpus_per_node, world_size, train_loop_kwargs),
             nprocs=gpus_per_node, join=True)

def manage_master_node_addr_and_port(node_rank, out_dir, unique_id):
    if os.getenv('MASTER_ADDR') is None:
        info_file = os.path.join(out_dir, f'{unique_id}_master_addr.txt')
        if node_rank == 0:
            hostname = socket.gethostname()
            ipaddr = socket.gethostbyname(hostname)
            with open(info_file, 'w') as f:
                f.write(ipaddr)
        else:
            while not os.path.exists(info_file):
                time.sleep(30)  # wait 30 seconds and check if the muster process has started
            with open(info_file) as f:
                ipaddr = f.readlines()[0]

        os.environ['MASTER_ADDR'] = ipaddr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpn', '--gpu_per_node', type=int, default=1)
    parser.add_argument('-n', '--nodes', type=int, default=1)
    parser.add_argument('-nr', '--node_rank', type=int, default=0)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-uid', '--unique_id', type=str, default=None)
    args = parser.parse_args()
    manage_master_node_addr_and_port(args.node_rank, args.output_dir, unique_id=args.unique_id)
    world_size = args.gpu_per_node * args.nodes
    spawn_processes_for_this_node(setup_and_run_single_process_train_code, args.gpu_per_node, args.node_rank,
                                  world_size, out_dir=args.output_dir)
