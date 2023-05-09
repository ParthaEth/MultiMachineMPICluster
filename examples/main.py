import os
import sys
sys.path.append(sys.path.append(os.getcwd()))
import argparse
from mmmc import launcher
from examples.training import train_loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpn', '--gpu_per_node', type=int, default=1)
    parser.add_argument('-n', '--nodes', type=int, default=1)
    parser.add_argument('-nr', '--node_rank', type=int, default=0)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-uid', '--unique_id', type=str, default=None)
    args = parser.parse_args()

    train_loop_kwargs = {}
    launcher.manage_com_run_train_loop(args.node_rank, args.output_dir, args.unique_id, args.gpu_per_node, args.nodes,
                                       train_loop.run_training_loop, **train_loop_kwargs)

