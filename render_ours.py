#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
from torchvision import transforms

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scale_factor, scale):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale}")
    # 360v2_ours_stmt_resize/bicycle/ -> benchmark_360v2_stmt/bicycle/test/ours_30000/gt_
    parts = model_path.split('/')
    root = parts[0].replace("360v2_ours", "benchmark_360v2")
    root = root.replace("_resize", "")
    scene = parts[1]
    gt_iteration = 30000
    gt_folder = os.path.join(root, scene, name, "ours_{}".format(gt_iteration), f"gt_{scale}")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        #################
        gt_path_now = os.path.join(gt_folder, view.image_name + ".png")
        gt = Image.open(gt_path_now)
        gt.resize((view.image_width, view.image_height))
        transform = transforms.ToTensor()
        gt = transform(gt)
        #################
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, scale: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scale_factor=scale_factor, scale=scale)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scale", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.scale)