import open3d as o3d
import numpy as np

for i in range(100):
    mesh = o3d.io.read_triangle_mesh(f'/home/raphael/PycharmProjects/stam/meshes/box/box_{i}.stl')
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(1000)
    o3d.io.write_point_cloud(f"point_clouds/box/box_{i}.pcd", pcd)

# camera = [0, 0, diameter]
# radius = diameter * 100
#
# print("Get all points that are visible from given view point")
# _, pt_map = pcd.hidden_point_removal(camera, radius)
#
# print("Visualize result")
# pcd = pcd.select_by_index(pt_map)
# o3d.visualization.draw_geometries([pcd])
#
# import open3d as o3d
# import numpy as np
#
# mesh = o3d.io.read_triangle_mesh(f'/home/raphael/PycharmProjects/stam/meshes/cylinders/cylinder_10.stl')
# mesh.compute_vertex_normals()
# pcd = mesh.sample_points_poisson_disk(1000)
# diameter = np.linalg.norm(
#     np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
# o3d.visualization.draw_geometries([pcd])
#
# camera = [0.5, 0., diameter]
# radius = diameter * 100
#
# print("Get all points that are visible from given view point")
# _, pt_map = pcd.hidden_point_removal(camera, radius)
#
# print("Visualize result")
# pcd = pcd.select_by_index(pt_map)
# o3d.visualization.draw_geometries([pcd])