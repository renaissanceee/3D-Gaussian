import plyfile
from plyfile import PlyData, PlyElement
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.general_utils import build_rotation,build_scaling_rotation,strip_symmetric


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    # symm = strip_symmetric(actual_covariance)
    return actual_covariance

def process_ply(path):
    plydata = PlyData.read(path)
    max_sh_degree = 3
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    xyz = torch.from_numpy(xyz)
    opacities = torch.from_numpy(opacities)
    features_dc = torch.from_numpy(features_dc).transpose(1, 2).contiguous()
    features_rest = torch.from_numpy(features_extra).transpose(1, 2).contiguous()
    scaling = torch.from_numpy(scales)
    rotation = torch.from_numpy(rots)
    return xyz, opacities, features_dc, features_rest, scaling, rotation

def plot_ellipsoid(a, b, c, center, color, ax=None):
    """
    Plot an ellipsoid given its semi-axes (a, b, c) and center.
    """
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 50)

    x = center[0] + a * np.outer(np.cos(phi), np.sin(theta))
    y = center[1] + b * np.outer(np.sin(phi), np.sin(theta))
    z = center[2] + c * np.outer(np.ones(np.size(phi)), np.cos(theta))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    print(color)
    # ax.plot_surface(x, y, z, color=color, alpha=0.2, edgecolors=color)
    ax.plot_surface(x, y, z, color=color, alpha=0.2)
if __name__ == "__main__":
    path = "./output/low_8_train/point_cloud/iteration_30000/point_cloud.ply"
    xyz, opacities, features_dc, features_rest, scaling, rotation = process_ply(path)
    # activate
    opacities = torch.sigmoid(opacities).cpu()
    scaling = torch.exp(scaling).cpu()
    # rotation = torch.nn.functional.normalize(rotation)
    # covariance_3d
    covariance_matrix = build_covariance_from_scaling_rotation(scaling, 1.0, rotation)
    covariance_matrix=covariance_matrix.cpu()
    # print(rots.shape)#[283569, 4]
    # print(scaling.shape)#[283569, 3]
    # print(xyz.shape)#[283569, 3]
    # print(opacities.shape)#[283569, 1]
    "opacity top-k"
    # topk_values, topk_indices = torch.topk(opacities.squeeze(), k=100)
    # print("Top 100 values:", topk_values)
    # print("Top 100 indices:", topk_indices)
    "scaling top-k"
    # scaling_sum=torch.sum(scaling,dim=1)
    # topk_values, topk_indices = torch.topk(scaling_sum.squeeze(), k=100)
    "covariance top-k"
    diagonal_elements = torch.diagonal(covariance_matrix, dim1=1, dim2=2)
    diagonal_sum = torch.sum(diagonal_elements, dim=1)
    topk_values, topk_indices = torch.topk(diagonal_sum, k=100)
    # select top-k
    # topk_indices=torch.Tensor([0])
    print(topk_indices)
    topk_indices[0] = 0
    covariance_matrix = covariance_matrix[topk_indices]
    scaling = scaling[topk_indices]
    xyz = xyz[topk_indices]
    opacities = opacities[topk_indices]
    # to cpu
    covariance_matrix=covariance_matrix.numpy()
    scaling = scaling.numpy()
    xyz = xyz.numpy()
    opacities = opacities.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_list=["blue","green","red"]
    for i in range(len(xyz)):
        xyz_current = xyz[i]
        covariance = covariance_matrix[i]
        _, axes = np.linalg.eigh(covariance)# EVD to obtain semi-axes
        semi_axes = np.sqrt(1 / np.abs(_))
        plot_ellipsoid(semi_axes[0], semi_axes[1], semi_axes[2], center=xyz_current, color=color_list[i],ax=ax)# Plot the ellipsoid
        break
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    # plt.savefig('gaussian_'+"100"+'.png')
    plt.savefig('./single_gmm/gaussian_0.png')
    plt.close()