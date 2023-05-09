import os
import torch.distributed as dist
import torch.multiprocessing as mp

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

    os.system(f'fuser -n tcp -k {os.getenv("MASTER_PORT")}')
    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def setup_and_run_single_process_train_code(local_rank, node_rank, gpus_per_node, world_size, train_loop,
                                            train_loop_kwargs):
    print(f'Running basic DDP example on machine: {node_rank} in gpu: {local_rank}.')
    global_rank = gpus_per_node * node_rank + local_rank
    setup(global_rank, world_size)
    print(f'Starting training loop')
    try:
        train_loop(local_rank=local_rank, global_rank=global_rank, **train_loop_kwargs)
    finally:
        cleanup()

def spawn_processes_for_this_node(setup_and_run_single_process_train_code, gpus_per_node, node_rank, world_size,
                                  train_loop, **train_loop_kwargs):
    mp.spawn(setup_and_run_single_process_train_code,
             args=(node_rank, gpus_per_node, world_size, train_loop, train_loop_kwargs),
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

def manage_com_run_train_loop(node_rank, output_dir, unique_id, gpu_per_node, nodes, train_loop, **train_loop_kwargs):
    manage_master_node_addr_and_port(node_rank, output_dir, unique_id=unique_id)
    world_size = gpu_per_node * nodes
    spawn_processes_for_this_node(setup_and_run_single_process_train_code, gpu_per_node, node_rank,
                                  world_size, train_loop, **train_loop_kwargs)

