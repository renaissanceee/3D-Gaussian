# single-scale training and multi-scale testing setting proposed in mip-splatting

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

scenes = ["bicycle", "bonsai", "counter", "garden", "stump", "kitchen", "room", "flowers", "treehill"]
factors = [1, 1, 1, 1, 1, 1, 1, 1, 1]

excluded_gpus = set([])

output_dir = "benchmark_360v2_stmt_down"

dry_run = False

jobs = list(zip(scenes, factors))


def train_scene(gpu, scene, factor):
    for scale in [8, 4, 2]:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene} -r {scale}"
        print(cmd)
        if not dry_run:
            os.system(cmd)

    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)