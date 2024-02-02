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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_H, render_W):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if render_W != 0:
            view.image_width, view.image_height = render_W, render_H
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_set_HR_gt(model_path, name, iteration, views, gaussians, pipeline, background, render_H, render_W,
                     save_HR_gt):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    "add GMM to gaussians"
    ######################################
    # "GMM for cov3D"
    pipeline.compute_cov3D_python = True
    "generate GMM"
    # scales = gaussians.get_scaling
    # rotations = gaussians.get_rotation
    # xyz = gaussians.get_xyz
    # print("GMM starts with num:", xyz.size()[0])
    # # covariance_3d
    # covariance_matrix = build_covariance_from_scaling_rotation(scales, 1.0, rotations)
    # covariance_matrix = covariance_matrix.cpu()
    # xyz = xyz.cpu()
    # num_of_ply = xyz.size()[0]
    # gmm_mean,gmm_cov = generate_gmm(xyz, covariance_matrix, num_of_ply)
    # # save GMM in json
    # data_to_write = {"gmm_mean": gmm_mean.tolist(),"gmm_cov":gmm_cov.tolist()}
    # json_file_path = "gmm_mean_cov.json"
    # with open(json_file_path, "w") as json_file:
    #     json.dump(data_to_write, json_file)

    "load from json"
    json_file_path = "gmm_mean_cov.json"
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    gmm_mean_list = data["gmm_mean"]
    gmm_cov_list = data["gmm_cov"]
    gmm_mean = np.array(gmm_mean_list)
    gmm_cov = np.array(gmm_cov_list)

    # to torch and cuda
    gmm_cov = torch.from_numpy(gmm_cov).float()
    cov3D_gmm = strip_lowerdiag(gmm_cov).cuda()
    gaussians._xyz = torch.from_numpy(gmm_mean).float().cuda()

    # deplicate for others
    gaussians._opacity = concat_opacity(gaussians._opacity)
    gaussians._features_dc = concat_fea(gaussians._features_dc)
    gaussians._features_rest = concat_fea(gaussians._features_rest)
    gaussians._scaling = None
    gaussians._rotation = None

    print("GMM finished with num:", cov3D_gmm.size()[0])
    ######################################
    # render
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if render_W!=0:
            view.image_width, view.image_height =  render_W,render_H
        rendering = render(view, gaussians, pipeline, background, cov3D_gmm=cov3D_gmm)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    # gt: copy HR
    source_path = os.path.join(save_HR_gt, name, "ours_" + str(iteration), "gt")
    for file_name in os.listdir(source_path):
        source_file_path = os.path.join(source_path, file_name)
        gts_file_path = os.path.join(gts_path, file_name)
        copyfile(source_file_path, gts_file_path)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                render_H: int, render_W: int, save_HR_gt: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            if save_HR_gt != "":
                render_set_HR_gt(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians,
                                 pipeline, background, render_H, render_W, save_HR_gt)
            else:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                           background, render_H, render_W)

        if not skip_test:
            if save_HR_gt != "":
                render_set_HR_gt(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians,
                                 pipeline, background, render_H, render_W, save_HR_gt)
            else:
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