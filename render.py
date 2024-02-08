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
from PIL import Image
from torchvision import transforms

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_H, render_W, gt_folder):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    # render_path = os.path.join("./output/zoom_out_bicycle/low_4/baseline/render_down_2", name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join("./output/zoom_out_bicycle/low_4/baseline/render_down_2", name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if render_W != 0:
            view.image_width, view.image_height = render_W, render_H
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))

        gt = view.original_image[0:3, :, :]
        if gt_folder!="":
            # gt_path = "./tandt_db/tandt/train/images_8/"+view.image_name + ".jpg"
            gt_path_now = os.path.join(gt_folder, view.image_name + ".JPG")
            gt = Image.open(gt_path_now)
            gt.resize((view.image_width, view.image_height))
            transform = transforms.ToTensor()
            gt = transform(gt)
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))


# def render_set_HR_gt(model_path, name, iteration, views, gaussians, pipeline, background, render_H, render_W,
#                      save_HR_gt):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
# 
#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)
# 
#     # render
#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         if render_W!=0:
#             view.image_width, view.image_height =  render_W,render_H
#         # # zoom-out
#         # view.FoVx *=2
#         # view.FoVy *= 2
#         rendering = render(view, gaussians, pipeline, background)["render"]
#         torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
# 
#     # gt: copy HR
#     source_path = os.path.join(save_HR_gt, name, "ours_" + str(iteration), "gt")
#     for file_name in os.listdir(source_path):
#         source_file_path = os.path.join(source_path, file_name)
#         gts_file_path = os.path.join(gts_path, file_name)
#         copyfile(source_file_path, gts_file_path)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                render_H: int, render_W: int, gt_folder: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians,
                             pipeline, background, render_H, render_W, gt_folder)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians,
                             pipeline, background, render_H, render_W, gt_folder)



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
    parser.add_argument("--gt_folder", type=str, default="")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.render_H, args.render_W, args.gt_folder)