# single-scale training and multi-scale testing setting proposed in mip-splatting

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
scenes = ["bicycle", "bonsai", "counter", "garden", "stump", "kitchen", "room", "flowers", "treehill"]
factors = [8, 8, 8, 8, 8, 8, 8, 8, 8]

excluded_gpus = set([])

output_dir = "360v2_ours_stmt_downsampl"

dry_run = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    get_folder = "/cluster/work/cvl/jiezcao/jiameng/3D-Gaussian_slurm/benchmark_360v2_stmt/"
    trained_gaussian = os.path.join(get_folder, scene, "point_cloud/iteration_30000/point_cloud.ply") # "./fused/"+scene+"_fused_x1.ply"
    for scale in [4, 2, 1]:
        pseudo_gt = os.path.join(get_folder, scene, "pseudo_gt/resize_x8")
        model_path= os.path.join(output_dir,scene,"resize_x"+str(scale))
        if scene == "bicycle":
            H, W = 3286/ scale, 4946 / scale
        if scene == "bonsai":
            H, W = 2078/ scale, 3118 / scale
        if scene == "counter":
            H, W = 2076/ scale, 3115 / scale
        if scene == "garden":
            H, W = 3361/ scale, 5187 / scale
        if scene == "stump":
            H, W = 3300/ scale, 4978 / scale
        if scene == "kitchen":
            H, W = 2078/ scale, 3115 / scale
        if scene == "room":
            H, W = 2075/ scale, 3114 / scale
        if scene == "flowers":
            H, W = 3312/ scale, 5025 / scale
        if scene == "treehill":
            H, W = 3326/ scale, 5068 / scale
        H, W = int(round(H)),int(round(W))
        
        cmd = f"python train_downsampl.py -s {pseudo_gt} -m {model_path} -r 1 --port {3009 + int(gpu)} --load_gaussian {trained_gaussian} " \
              f"--H {H} --W {W}"
        print(cmd)
        if not dry_run:
            os.system(cmd)

        cmd = f"python render_ours_downsampl.py -m {model_path} --scale {scale} -r 1 --data_device cpu --skip_train --iteration 100"
        print(cmd)
        if not dry_run:
            os.system(cmd)

        cmd = f"python render_ours_downsampl.py -m {model_path} --scale {scale} -r 1 --data_device cpu --skip_train --iteration 500"
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
    print(jobs)
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
            print(f"Job {job} has finished., releasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

