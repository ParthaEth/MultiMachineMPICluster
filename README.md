# MultiMachineMPICluster

## Goal

The primary goal of this repository is to enable users to run neural network training over multiple machines. As a starter, look up PyTorch native multi-machine support [here](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html).

### Additional goals

1. Use as lightweight (on IO) dataloading processes as possible.
2. Enable memory mapped caching to avoid data starving.
3. Enable exact training restart, i.e. stop a job and restart from where it left off.

## Dependencies

This repo will rely on native PyTorch functionality as much as possible to make it as easy to use. [Horovod](https://github.com/horovod/horovod) and other such tools/frameworks are known to provide useful wrappers and can, as such, be supported in the future in different branches. However, the main branch should be dedicated to native PyTorch support.

## PRs are highly encouraged

If you are using this repository, you are encouraged to contribute to it as well.

## Usage

Clone the repository and then make modifications to fit your specific needs; feel free to edit and make changes as you like. If you implement a generic feature that you think everyone can also use, make a PR.
Even if you are using **single machine** and multiple GPUs. Switching to this repo will **speed up your training** as compared to `torch.nn.DataParallel`.

To run this program you can use the following command
`./condor_submit_script.sh --bid=100 --num_cpus_per_node=2 --num_gpus_per_node=1 --cpu_ram_MB=64000 --cuda_device_name='NVIDIA A100-SXM4-80GB' --python_interp_path=<Full_path_to_python_executable_in_venv_or_conda> --nodes=2 --output_dir=<out_dir> --path_top_py_scpt=./examples/main.py`

## Known problems and fixes
1. To get way more info on what NCCL is doing set the following env. vars `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`
2. `NCCL WARN Call to ibv_create_cq failed with error Cannot allocate memory` - It is likely that `ulimit -l` is set too low

## Things to be double checked
1. do we need `has100GbE =?= true` - this condor flag?