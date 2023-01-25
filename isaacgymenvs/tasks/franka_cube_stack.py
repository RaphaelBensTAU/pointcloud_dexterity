# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym.torch_utils import *
from Pointnet_Pointnet2_pytorch import my_pointnet

# A force sensor, measures force and torque. For each force sensor,
# its measurements are represented by a tensor with 6 elements of dtype float32.
# The first 3 element are the force measurements and the last 3 are the torque measurements.
N_ELEM_PER_FORCE_SENSOR = 6

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaCubeStack(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 17 + 1024
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.franka_default_dof_pos = to_torch(
            [0, 1.55, -2.0, 1.0, 0, 0, -0.7], device=self.device
        )

        # Set control limits
        self.cmd_limit = self._franka_effort_limits[:].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.projected_pointclouds_features = torch.zeros((self.num_envs, 1024), device=self.device)

        # Refresh tensors
        self._refresh()

        self.POINTCLOUD_NUM_POINTS = 512

        checkpoint = torch.load(
            '/home/raphael/PycharmProjects/stam/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth')
        self.point_net = my_pointnet.PointNet().cuda().eval()
        self.point_net.load_state_dict(checkpoint['model_state_dict'])

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "z1_pro/urdf/z1.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        # asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        stiffness = [50.0] * 32
        damping = [5.0] * 32

        # Create table asset
        table_pos = [-0.15, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.5, 0.7, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.01, 0.01, table_stand_height], table_opts)

        self.cubeA_size = 0.04
        self.cubeB_size = 0.05

        # cubeB_asset = self.gym.load_asset(self.sim, '', "/home/raphael/media/bensadoun/IsaacGymEnvs_v4_multicube/assets/urdf/objects/tray/urdf/tray.urdf", bottle_opts)
        # cubeB_color = gymapi.Vec3(1., 1., 1.)

        #
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = stiffness[i]
                franka_dof_props['damping'][i] = damping[i]

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

        #TODO wtf
        # self.franka_dof_speed_scales[[7, 8]] = 0.1
        # franka_dof_props['effort'][7] = 200
        # franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        tray_stand_start_pose = gymapi.Transform()
        tray_stand_start_pose.p = gymapi.Vec3(*[-0.35, 0., 1.03])
        tray_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4     # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 4     # 1 for table, table stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

        NUM_OBJECTS = 50
        cubeA_assets = []
        import open3d as o3d
        point_clouds = []
        for i in range(NUM_OBJECTS):
            point_cloud_cylinder = o3d.io.read_point_cloud(f"../point_clouds/cylinders/cylinder_{i}.pcd")
            point_cloud_box = o3d.io.read_point_cloud(f"../point_clouds/box/box_{i}.pcd")
            point_clouds.append(point_cloud_cylinder)
            point_clouds.append(point_cloud_box)
        bottle_opts = gymapi.AssetOptions()
        bottle_opts.vhacd_enabled = True
        bottle_opts.vhacd_params.resolution = 300000
        bottle_opts.vhacd_params.max_convex_hulls = 50
        bottle_opts.vhacd_params.max_num_vertices_per_ch = 64
        self.cylinder_heights = []
        self.cylinder_radius = []
        self.object_heights = []

        self.cube_heights = []
        for i in range(NUM_OBJECTS):
            print(f"Loading urdf {i}")
            cubeA_asset = self.gym.load_asset(self.sim, '../urdfs/cylinders', f"cylinder_{i}.urdf", bottle_opts)
            with open(f'../meshes/cylinders/cylinder_{i}.txt', 'r') as f:
                line = f.readline()
            line = line.strip().split(',')
            self.cylinder_heights.append(float(line[0]))
            self.cylinder_radius.append(float(line[1]))
            cubeA_assets.append(cubeA_asset)

            self.object_heights.append(float(line[0]))

            print(f"Loading urdf {i}")
            cubeA_asset = self.gym.load_asset(self.sim, '../urdfs/box', f"box_{i}.urdf", bottle_opts)
            with open(f'../meshes/box/box_{i}.txt', 'r') as f:
                line = f.readline()
            line = line.strip().split(',')
            self.cube_heights.append(float(line[2]))
            cubeA_assets.append(cubeA_asset)

            self.object_heights.append(float(line[2]))

        # Create environments
        for i in range(self.num_envs):

            cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                  1.0 + table_thickness / 2 + table_stand_height)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_assets[i% (NUM_OBJECTS * 2)], cubeA_start_pose, "cubeA", i, 4, 0)
            # self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            # self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        # self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # duplicate heights and radius according to modulo applied above
        # heights = self.cylinder_heights * (self.num_envs // NUM_OBJECTS)
        # heights += self.cylinder_heights[:self.num_envs % NUM_OBJECTS]
        #
        # radius = self.cylinder_radius * (self.num_envs // NUM_OBJECTS)
        # radius += self.cylinder_radius[:self.num_envs % NUM_OBJECTS]

        all_heights = self.object_heights * (self.num_envs // (NUM_OBJECTS * 2))
        all_heights += self.object_heights[: self.num_envs % (NUM_OBJECTS * 2)]

        self.point_clouds = {i : point_clouds[i%(NUM_OBJECTS * 2)] for i in range(self.num_envs)}

        self.first_iteration = True

        self.object_heights = torch.tensor(all_heights, device = self.device)



        # Setup data
        self.init_data()
    def oversample(self, pointcloud, num_points):
        oversample_indices = torch.multinomial(torch.tensor([i for i in range(len(pointcloud))]).float(), num_points - len(pointcloud), replacement=True)
        pointcloud = torch.cat((pointcloud, pointcloud[oversample_indices]), dim = 0)
        return pointcloud

    def undersample(self, pointcloud, num_points):
        undersample_indices = torch.multinomial(torch.tensor([i for i in range(len(pointcloud))]).float(), num_points, replacement=True)
        pointcloud = pointcloud[undersample_indices]
        return pointcloud

    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        i, j, k, r = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def normalize_pointcloud(self, points):
        centered_points = points - points.mean(dim = 0)
        max_dist = torch.max(torch.norm(centered_points, dim=-1, keepdim=True), dim = 1, keepdim=True)[0].repeat(1, centered_points.shape[1], centered_points.shape[2])
        centered_points = centered_points / max_dist
        return centered_points


    def project_pointcloud(self, env_ids, cube_quats):
        projected_pointclouds = torch.zeros((len(env_ids), 512, 3))
        euler_quats = self.quaternion_to_matrix(cube_quats)
        for i in range(len(env_ids)):
            pointcloud = self.point_clouds[env_ids[i].item()]
            pointcloud = pointcloud.rotate(np.array(euler_quats[i].cpu().numpy()))
            diameter = np.linalg.norm(np.asarray(pointcloud.get_max_bound()) - np.asarray(pointcloud.get_min_bound()))
            camera = [0.5, 0, diameter]
            radius = diameter * 100
            _, pt_map = pointcloud.hidden_point_removal(camera, radius)
            proj_pointcloud = torch.tensor(np.array(pointcloud.points))[pt_map]
            if proj_pointcloud.shape[0] < self.POINTCLOUD_NUM_POINTS:
                proj_pointcloud = self.oversample(proj_pointcloud, self.POINTCLOUD_NUM_POINTS)
            else:
                proj_pointcloud = self.undersample(proj_pointcloud, self.POINTCLOUD_NUM_POINTS)
            projected_pointclouds[i] = proj_pointcloud
        projected_pointclouds = self.normalize_pointcloud(projected_pointclouds)
        return projected_pointclouds

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            "end_effector": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "gripper_ee"),
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

        self._eef_state = self._rigid_body_state[:, self.handles["end_effector"], :]

        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # Initialize states
        self.states.update({
            "cubeA_size": self.object_heights / 2,
            "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,

        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self._arm_control = self._pos_control

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],

            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_vel" : self._cubeA_state[:, 7:],

            "to_target": self._cubeA_state[:, :3] - self._eef_state[:, :3],

            "pointcloud_feature": self.projected_pointclouds_features

        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )


    def compute_observations(self):
        self._refresh()
        obs = ["q", "eef_pos", "eef_quat", "to_target", "pointcloud_feature"]

        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        if not self.first_iteration:
            first_step_ids = torch.where(self.progress_buf == 0)[0]
        else:
            first_step_ids = torch.where(self.progress_buf == 1)[0]
            self.first_iteration = False
        if len(first_step_ids) > 0:
            projected_pointclouds = self.project_pointcloud(first_step_ids,
                                                            self.states['cubeA_quat'][first_step_ids]).cuda()
            with torch.no_grad():
                BATCH_SIZE = 256
                for i in range(len(first_step_ids) // BATCH_SIZE):
                    projected_pointclouds_features = self.point_net(
                        projected_pointclouds[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE].permute(0, 2, 1))
                    self.projected_pointclouds_features[first_step_ids[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]] = \
                    projected_pointclouds_features[1].squeeze(-1)
                i = len(first_step_ids) // BATCH_SIZE - 1
                if len(first_step_ids) % BATCH_SIZE != 0 and i != -1:
                    projected_pointclouds_features = self.point_net(
                        projected_pointclouds[i * BATCH_SIZE + BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE + (
                                    len(first_step_ids) % BATCH_SIZE)].permute(0, 2, 1))
                    self.projected_pointclouds_features[first_step_ids[
                                                        i * BATCH_SIZE + BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE + (
                                                                    len(first_step_ids) % BATCH_SIZE)]] = projected_pointclouds_features[1].squeeze(-1)
        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=False)

        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]

        reset_noise = torch.rand((len(env_ids), 7), device = self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        # Overwrite gripper init pos (no noise since these are always position controlled)

        # pos[:, DOF_LEFT_ELBOW] = 2
        # pos[:, DOF_RIGHT_ELBOW_PITCH] = -2
        # Reset the internal obs accordingly

        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        self._pos_control[env_ids, :] = pos

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            # other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.object_heights
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_cubeB_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        min_dists = ((self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0)

        # We scale the min dist by 2 so that the cubes aren't too close together
        min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2 + 0.01

        def randomize_heading(cube_state):
            theta = torch.rand((cube_state.shape[0],)) * (2 * torch.pi) - torch.pi
            w = torch.cos(theta / 2)
            z = torch.sin(theta / 2)
            sampled_cube_state[:, 6] = w
            sampled_cube_state[:, 5] = z

        randomize_heading(sampled_cube_state)

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
                # Check if sampled values are valid
                cube_dist = torch.linalg                .norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        this_cube_state_all[env_ids, :] = sampled_cube_state


    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        u_arm = self.actions

        scale = 0.5 * (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
        offset = 0.5 * (self.franka_dof_lower_limits + self.franka_dof_upper_limits)

        u_arm = offset + scale * actions
        u_arm = u_arm.clip(self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        self._arm_control[:, :] = u_arm

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
    cubeA_size = states["cubeA_size"]
    cubeB_size = states["cubeB_size"]

    d = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)
    # print(d)
    dist_reward = 1 - torch.tanh(4.0 * (d))

    # print(d.mean())

    cubeA_height = states["cubeA_pos"][:, 2] - 1.
    # cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
    # print(cubeA_height - cubeA_size)

    # print(lift_reward)
    print(dist_reward.mean().item())#, (cubeA_height - cubeA_size).float().mean().item())
    # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
    # offset = torch.zeros_like(states["cubeA_pos"])
    # offset[:, 2] = (cubeA_size + cubeB_size) / 2
    # d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
    # align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

    # dist_reward = torch.max(dist_reward, align_reward)

    # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
    # cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.05)
    # cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
    # gripper_away_from_cubeA = (d > 0.05)
    # stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA

    rewards = 1 * dist_reward #+ 1.5 * (cubeA_height - cubeA_size)

    # rewards = torch.where(
    #     stack_reward ,
    #     reward_settings["r_stack_scale"] * stack_reward  ,
    #     reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
    #         "r_align_scale"] * align_reward,
    # )

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
    cube_fall = states['cubeA_pos'][:, 2] < 1
    reset_buf |= cube_fall
    reset_buf |= (cubeA_height - cubeA_size) > 0.5
    # reset_buf |= (states['cubeA_pos'][:, 0] < -0.40)
    # reset_buf |= (states['cubeA_pos'][:, 1] < -0.6)
    # reset_buf |= (states['cubeA_pos'][:, 1] > 0.6)
    return rewards, reset_buf
