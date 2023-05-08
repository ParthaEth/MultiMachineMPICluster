#! /bin/bash
dir_to_sync='MultiMachineMPICluster'
usr=$1
while inotifywait -r --exclude '/\.' ../$dir_to_sync/*; do
  rsync --exclude=".*" -azhe "ssh -i ~/.ssh/id_rsa" ../$dir_to_sync/ $usr@brown.is.localnet:/is/cluster/$usr/repos/never_eidt_from_cluster_remote_edit_loc/$dir_to_sync &
  sleep 2
  echo "second sync."
  rsync --exclude=".*" -azhe "ssh -i ~/.ssh/id_rsa" ../$dir_to_sync/ $usr@brown.is.localnet:/is/cluster/$usr/repos/never_eidt_from_cluster_remote_edit_loc/$dir_to_sync
done
