#! /bin/bash
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu_per_node=*)
      gpu_per_node="${1#*=}"
      ;;
    --nodes=*)
      nodes="${1#*=}"
      ;;
    --output_dir=*)
      output_dir="${1#*=}"
      ;;
    --python_interp_path=*)
      python_interp_path="${1#*=}"
      ;;
    --bid=*)
      bid="${1#*=}"
      ;;
    --num_cpus_per_node=*)
      num_cpus_per_node="${1#*=}"
      ;;
    --num_gpus_per_node=*)
      num_gpus_per_node="${1#*=}"
      ;;
    --cpu_ram_MB=*)
      cpu_ram_MB="${1#*=}"
      ;;
    --cuda_device_name=*)
      cuda_device_name="${1#*=}"
      ;;
    --path_top_py_scpt=*)
      path_top_py_scpt="${1#*=}"
      ;;
    *)
      printf "****************************\n"
      printf "* Error: Invalid argument. $1 *\n"
      printf "****************************\n"
      exit 1
  esac
  shift
done

command="condor_submit_bid $bid <(printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n' 'error=$output_dir/\$(Cluster)_\$(Process).err' 'output=$output_dir/\$(Cluster)_\$(Process).out' 'log=$output_dir/\$(Cluster)_\$(Process).log' 'request_cpus=$num_cpus_per_node' 'request_gpus=$num_gpus_per_node' 'request_memory=$cpu_ram_MB' 'requirements=CUDADeviceName==\"$cuda_device_name\"' 'executable=$python_interp_path' 'arguments=$path_top_py_scpt --gpu_per_node $num_gpus_per_node --nodes $nodes --node_rank \$(Process) --output_dir $output_dir --unique_id \$(Cluster)' 'queue $nodes')"
eval "$command"

