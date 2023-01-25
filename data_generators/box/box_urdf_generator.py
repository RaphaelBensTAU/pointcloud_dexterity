import numpy as np
from stl import mesh

xml_template =  '<?xml version="1.0" encoding="utf-8"?>  \
<robot name="cylinder_{}"> \
  <link name="cylinder_{}"> \
    <visual> \
      <geometry> \
        <mesh filename="../meshes/box/box_{}.stl" scale="1 1 1"/> \
      </geometry> \
    </visual> \
    <collision> \
      <geometry> \
        <mesh filename="../meshes/box/box_{}.stl" scale="1 1 1"/> \
      </geometry> \
    </collision> \
    <inertial>  \
      <origin \
        xyz="0 0 0" \
        rpy="0 0 0" /> \
      <mass value="{}"/> \
      <origin rpy="0 0 0" xyz="0 0 0"/> \
      <inertia ixx="{}" ixy="{}" ixz="{}" iyy="{}" iyz="{}" izz="{}"/> \
    </inertial> \
  </link>  \
</robot>'

for i in range(100):
    name = f'box_{i}'
    mesh_x = mesh.Mesh.from_file(f'../../meshes/box/box_{i}.stl')
    volume, cog, inertia = mesh_x.get_mass_properties()
    volume = volume * 1000
    inertia_ixx = inertia[0, 0] * 1000
    inertia_ixy = inertia[0, 1] * 1000
    inertia_ixz = inertia[0, 2] * 1000
    inertia_yy = inertia[1, 1] * 1000
    inertia_yz = inertia[1, 2] * 1000
    inertia_zz = inertia[2, 2] * 1000
    with open(f'../../urdfs/box/box_{i}.urdf', 'w+') as f:
        xml = xml_template.format(name, name, i, i, volume, inertia_ixx, inertia_ixy, inertia_ixz, inertia_yy, inertia_yz, inertia_zz)
        f.write(xml)