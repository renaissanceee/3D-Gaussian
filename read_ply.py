import open3d as o3d

def visualize_ply(ply_file_path):
    # 读取PLY文件
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    print("num_points = ", len(point_cloud.points))

    # 创建Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云到Visualizer
    vis.add_geometry(point_cloud)

    # 设置视角
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, 1, 0])
    view_control.set_lookat([0, 0, 0])

    # 运行Visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    ply_file_path = "./dataset/tandt_db/tandt/train_LR/sparse/0/points3D.ply"
    # LR
    # ply_file_path = "./output/bb2e3b89-2/input.ply"# initial
    # ply_file_path = "./output/bb2e3b89-2/point_cloud/iteration_7000/point_cloud.ply"# trained
    ply_file_path = "./output/bb2e3b89-2/point_cloud/iteration_30000/point_cloud.ply"# trained


    # HR
    # ply_file_path = "./output/745f977e-a/input.ply"# initial
    # ply_file_path = "./output/745f977e-a/point_cloud/iteration_7000/point_cloud.ply"# trained
    # ply_file_path = "./output/745f977e-a/point_cloud/iteration_30000/point_cloud.ply"# trained    

    visualize_ply(ply_file_path)
