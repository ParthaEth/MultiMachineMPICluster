# MultiMachineMPICluster
## Goal
The primary goal of this repository is to enable users to run neural network training over multiple machines. As a starter look up pytorch native multi machine support here - [https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html)
### Additional goals
1. Use as lightweight (on IO) dataloading process as possible
2. Enable memory mapped cacheing to avoid data starving
3. Enable exact training restart, i.e. stop a job and restart from where it left off

## Dependencies
This repo will rely as much as possible on native pytorch functionality to make it as easy as possible to use it. Horovod and others are known to privide useful wrappers and can as such be supported in the future in different brnaches but the main brnach should be dedicated to native pytorch support.

## PRs are highly encouraged
If you are using this repo you are encouraged to contribute to it aswell.

## Usage
Clone the repo as base repository then specialize according to your specific needs feel free to edit and modify as you like. If you implement a generic feature that you think everyone can also use make a PR.
