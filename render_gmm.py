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
from shutil import copyfile
from generate_gmm import generate_gmm,concat_opacity,concat_fea,build_covariance_from_scaling_rotation
from utils.general_utils import strip_lowerdiag
import json
import numpy as np
from PIL import Image
from torchvision import transforms

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_H, render_W):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    "add GMM to gaussians"
    ######################################
    # "GMM for cov3D"
    "load from json"
    json_file_path = "gmm_low8.json"
    # json_file_path = "gmm_mean_cov_low8.json"

    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    gmm_mean = np.array(data["gmm_mean"])
    gmm_cov = np.array(data["gmm_cov"])
    gmm_opacities = np.array(data["opacities"])
    gmm_features_dc = np.array(data["features_dc"])
    gmm_features_rest =np.array(data["features_rest"])
    # to torch and cuda
    gaussians._xyz = torch.from_numpy(gmm_mean).float().cuda()
    cov3D_gmm = strip_lowerdiag(torch.from_numpy(gmm_cov)).float().cuda()
    gaussians._opacity = torch.from_numpy(gmm_opacities).cuda()
    gaussians._features_dc = torch.from_numpy(gmm_features_dc).cuda()
    gaussians._features_rest = torch.from_numpy(gmm_features_rest).cuda()
    gaussians._scaling = None
    gaussians._rotation = None

    print("GMM finished with num:", cov3D_gmm.size()[0])
    ######################################
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if render_W != 0:
            view.image_width, view.image_height = render_W, render_H
        pipeline.compute_cov3D_python = True
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))# gt = view.original_image[0:3, :, :]
        gt_path = "./tandt_db/tandt/train/images/"+view.image_name + ".jpg"
        gt=Image.open(gt_path)
        transform = transforms.ToTensor()
        gt = transform(gt)
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                render_H: int, render_W: int, save_HR_gt: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                           background, render_H, render_W)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                           background, render_H, render_W)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_H", default=0, type=int)
    parser.add_argument("--render_W", default=0, type=int)
    parser.add_argument("--save_HR_gt", type=str, default="")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.render_H, args.render_W, args.save_HR_gt)