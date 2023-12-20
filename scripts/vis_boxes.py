import open3d as o3d
import numpy as np

point_cloud = o3d.geometry.PointCloud()

point_cloud.points = o3d.utility.Vector3dVector(valid_points1[:, :])

point_cloud.colors = o3d.utility.Vector3dVector(colors[valid_points[:, 8].astype(np.int32)])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud)

for i, box in enumerate(bboxes):
    b = o3d.geometry.OrientedBoundingBox()
    b.center = box[:3]
    b.extent = box[3:6]
    # with heading
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0, box[6]))
    b.rotate(R, b.center)  
   
    vis.add_geometry(b)
vis.get_render_option().background_color = np.asarray([0, 0, 0]) # 设置一些渲染属性
vis.run()
vis.destroy_window()
