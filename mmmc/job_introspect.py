import time
import os
import socket
import fcntl
from typing import Optional, Generator
from pathlib import Path
from contextlib import contextmanager


def get_cluster_process() -> tuple[int, int]:
    job_id = str(os.environ["JOB_ID"])
    cluster_process = job_id.split("#")[1]
    cluster, process = cluster_process.split(".")
    return int(cluster), int(process)


def get_htcondor_descriptors() -> dict[str, str]:
    descriptors_file = os.environ["_CONDOR_JOB_AD"]
    descriptors = {}
    with open(descriptors_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=")
                descriptors[key.strip()] = value.strip()
    return descriptors


def get_nb_nodes(descriptors: dict[str, str]) -> int:
    return int(descriptors["TotalSubmitProcs"])


def get_nb_gpus_per_node(descriptors: dict[str, str]) -> int:
    return int(descriptors["RequestGPUs"])


def get_log_directory(descriptors: dict[str, str]) -> Path:
    return Path(descriptors["UserLog"]).parent


def _get_master_address_file(log_directory: Path, cluster_id: int) -> Path:
    return log_directory / f"{cluster_id}_master_address.txt"


@contextmanager
def _lock_file(f) -> Generator[None, None, None]:
    fcntl.flock(f, fcntl.LOCK_EX)
    yield
    fcntl.flock(f, fcntl.LOCK_UN)


def share_master_address(log_directory: Path, cluster_id: int) -> str:
    shared_file = _get_master_address_file(log_directory, cluster_id)
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    # write the ip_ address to the shared file in a multiprocessing-safe way
    # (i.e. using a lock)
    with open(shared_file, "w") as f:
        with _lock_file(f):
            f.write(ip_address)
    return ip_address


def wait_for_master_address(
    log_directory: Path, cluster_id: int, timeout: Optional[float] = 60
) -> str:
    shared_file = _get_master_address_file(log_directory, cluster_id)
    time_start = time.time()
    while not shared_file.exists():
        time.sleep(0.1)
        if timeout is not None:
            if time.time() - time_start > timeout:
                raise TimeoutError("timeout while waiting for master address")
    with open(shared_file, "r") as f:
        with _lock_file(f):
            ip_address = f.read()
    return ip_address


def set_master_addr_env_var(
    log_directory: Path, cluster_id: int, node_rank: int, master_port: int
) -> None:
    if node_rank == 0:
        ip_address = share_master_address(log_directory, cluster_id)
    else:
        ip_address = wait_for_master_address(log_directory, cluster_id)
    os.environ["MASTER_ADDR"] = ip_address
    os.environ["MASTER_PORT"] = str(master_port)


class JobIntrospect:
    def __init__(self, master_port: int = 3630) -> None:
        descriptors = get_htcondor_descriptors()
        log_directory = get_log_directory(descriptors)
        self.cluster_id, self.node_rank = get_cluster_process()
        self.master_address_file = _get_master_address_file(
            log_directory, self.cluster_id
        )
        self.nb_nodes = get_nb_nodes(descriptors)
        self.nb_gpus_per_node = get_nb_gpus_per_node(descriptors)
        set_master_addr_env_var(
            log_directory, self.cluster_id, self.node_rank, master_port
        )
        self.master_port = master_port
