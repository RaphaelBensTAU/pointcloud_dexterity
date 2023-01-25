import math

import imageio
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# simple asset descriptor for selecting from a list


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=True):
        self.file_name = file_name

        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("mjcf/nv_humanoid.xml", False),
    AssetDesc("mjcf/nv_ant.xml", False),
    AssetDesc("urdf/cartpole.urdf", False),
    AssetDesc("urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf", False),
    AssetDesc("urdf/franka_description/robots/franka_panda.urdf", True),
    AssetDesc("urdf/kinova_description/urdf/kinova.urdf", False),
    AssetDesc("urdf/anymal_b_simple_description/urdf/anymal.urdf", True),
]


# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {
            "name": "--asset_id",
            "type": int,
            "default": 0,
            "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1),
        },
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
    ],
)

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
    quit()


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)
if sim is None:
    print("*** Failed to create sim")
    quit()

# # add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
# plane_params.distance = -3.2
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
# gym.set_light_parameters(sim, 0, intensity=gymapi.Vec3(1,1,1), ambient=gymapi.Vec3(1,1,1), direction=gymapi.Vec3(0,0,-1))
# gym.set_light_parameters(sim, 1, gymapi.Vec3(1,1,1), gymapi.Vec3(1,1,1), gymapi.Vec3(0,0,-1))
gym.set_light_parameters(
    sim, 3, gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(0, 0, -1)
)
# gym.set_light_parameters(sim, 1, gymapi.Vec3(0.5,0.5,0.5), gymapi.Vec3(0.5,0.5,0.5), gymapi.Vec3(0,1,0))

asset_root = asset_root_2 = "/home/raphael/media/bensadoun/IsaacGymEnvs_Tray_NEW/assets/urdf/new_sasson_model_gal/urdf/"
asset_file = asset_file_2 = "sasson.urdf"

# asset_root = asset_root_2 = "../../resources/robots/happybot_v2_wide/urdf"
# asset_file = asset_file_2 = "happy.urdf"

asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
# asset_options.flip_visual_attachments = False
# asset_options.use_mesh_materials = True

asset_options.disable_gravity = False
asset_options.collapse_fixed_joints = False
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = 3
asset_options.replace_cylinder_with_capsule = True
asset_options.flip_visual_attachments = False
asset_options.density = 0.001
asset_options.angular_damping = 0.0
asset_options.linear_damping = 0.0
asset_options.max_angular_velocity = 1000.0
asset_options.max_linear_velocity = 1000.0
asset_options.armature = 0.0
asset_options.thickness = 0.01
# asset_options.use_mesh_materials = True

asset_options.vhacd_enabled = False
asset_options.vhacd_params.resolution = 300000
asset_options.vhacd_params.max_convex_hulls = 50
asset_options.vhacd_params.max_num_vertices_per_ch = 64

# print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
# asset_2 = gym.load_asset(sim, asset_root_2, asset_file_2, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states["pos"]

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props["stiffness"]
dampings = dof_props["damping"]
armatures = dof_props["armature"]
has_limits = dof_props["hasLimits"]
lower_limits = dof_props["lower"]
upper_limits = dof_props["upper"]

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
# defaults = [0.337,0,0,0,-0.126,0,0,-0.337,0,0,0,0.126,0,0,0,0.77,0,1.0526,0,-0.77,0,-1.0526]
# defaults = [0.337, 0, 0, 0, -0.2269, 0.126, 0, -0.337, 0, 0, 0, 0.2269, -0.126, 0, 0, 0.77,0,1.0526,0,-0.77,0,-1.0526]
# defaults = np.array(
#     [
#         0.332,
#         -0.00524161,
#         0.178407,
#         0.21412,
#         -0.228917,
#         0.0544359,
#         -0.000953898,
#         -0.106437,
#         0.89488,
#         -0.00867663,
#         0.344684,
#         -0.332,
#         0.00523932,
#         -0.178411,
#         -0.214114,
#         0.222871,
#         -0.0544391,
#         0.000977788,
#         0.106339,
#         -0.894918,
#         0.00889888,
#         -0.344627,
#     ]
# )
defaults = np.zeros(num_dofs)
# defaults = [
#     0,
#     0.03490658504,
#     0,
#     0,
#     0.4651302457,
#     -0.2181661565,
#     0.0,
#     -0.1047197551,
#     -0.4625122518,
#     0.436332313,
#     0.0,
#     0.0,
# ]  # walky_v3.1
# defaults = [
#     0,
#     0.03490658504,
#     0,
#     0,
#     0.4651302457,
#     -0.2181661565,
#     0.0,
#     -0.1047197551,
#     -0.4625122518,
#     0.436332313,
#     0.0,
#     0.0,
# ]  # walky_v3.1
#
# defaults = [
#     0,
#     -0.12182,
#     0,
#     0,
#     0.48,
#     0,
#     0,
#     0.12182,
#     0,
#     0,
#     -0.48,
#     0,
#     0
# ]  # walky_v3.1
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    # if has_limits[i]:
    #     if dof_types[i] == gymapi.DOF_ROTATION:
    #         lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
    #         upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
    #     # make sure our default position is in range
    #     if lower_limits[i] > 0.0:
    #         defaults[i] = lower_limits[i]
    #     elif upper_limits[i] < 0.0:
    #         defaults[i] = upper_limits[i]
    # else:
    #     # set reasonable animation limits for unlimited joints
    #     if dof_types[i] == gymapi.DOF_ROTATION:
    #         # unlimited revolute joint
    #         lower_limits[i] = -math.pi
    #         upper_limits[i] = math.pi
    #     elif dof_types[i] == gymapi.DOF_TRANSLATION:
    #         # unlimited prismatic joint
    #         lower_limits[i] = -1.0
    #         upper_limits[i] = 1.0
    # set DOF position to default
    dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(
            2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi
        )
    else:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

# Print DOF properties
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# set up the env grid
num_envs = 1
num_per_row = 2
spacing = 0
env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)


# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0.0, 0.9)
    pose.r = gymapi.Quat(
        0.00778, -0.20365, 0.97934, 0
    )  # (-0.687,-0.144,0.698,-0.144) # (0.23335,-0.07849,0.96874,0.01768)  # gymapi.Quat(0.0, 0.0, 0.7071068, 0.7071068)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    # pose_2 = gymapi.Transform()
    # pose_2.p = gymapi.Vec3(0.0, 0.0, 0.0)  # gymapi.Vec3(-5.0, -5.0, 0.0)
    # pose_2.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    # actor_handle_2 = gym.create_actor(env, asset_2, pose_2, "kit", i, 1)
    actor_handles.append(actor_handle)
    # actor_handles.append(actor_handle_2)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

# position the camera
# cam_pos = gymapi.Vec3(17.2, 2.0, 16)
cam_pos = gymapi.Vec3(1.0, 0.0, 2)
cam_target = gymapi.Vec3(0.0, 0.0, 1.2)
# cam_target = gymapi.Vec3(0., 0., 0.)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

# Depth camera sensor
camera_props = gymapi.CameraProperties()
camera_props.width = 128
camera_props.height = 128
camera_props.enable_tensors = True

local_transform = gymapi.Transform()
local_transform.p = gymapi.Vec3(0.063, 0, -0.027)  # (-0.026, 0, -0,027)
local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(45.0))

env_handler = env
actor_handler = actor_handle
body_handler = gym.find_actor_rigid_body_handle(env_handler, actor_handler, "base")
camera_handler = gym.create_camera_sensor(env_handler, camera_props)
gym.attach_camera_to_body(
    camera_handler, env_handler, body_handler, local_transform, gymapi.FOLLOW_TRANSFORM
)

# Camera sensors

env_handler = env
camera_tensor = gym.get_camera_image_gpu_tensor(
    sim, env_handler, camera_handler, gymapi.IMAGE_DEPTH
)
torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)


# joint animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

# initialize animation state
anim_state = ANIM_SEEK_LOWER
current_dof = 0
print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

# gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    #
    # speed = speeds[current_dof]
    #
    # # animate the dofs
    # if anim_state == ANIM_SEEK_LOWER:
    #     dof_positions[current_dof] -= speed * dt
    #     if dof_positions[current_dof] <= lower_limits[current_dof]:
    #         dof_positions[current_dof] = lower_limits[current_dof]
    #         anim_state = ANIM_SEEK_UPPER
    # elif anim_state == ANIM_SEEK_UPPER:
    #     dof_positions[current_dof] += speed * dt
    #     if dof_positions[current_dof] >= upper_limits[current_dof]:
    #         dof_positions[current_dof] = upper_limits[current_dof]
    #         anim_state = ANIM_SEEK_DEFAULT
    # if anim_state == ANIM_SEEK_DEFAULT:
    #     dof_positions[current_dof] -= speed * dt
    #     if dof_positions[current_dof] <= defaults[current_dof]:
    #         dof_positions[current_dof] = defaults[current_dof]
    #         anim_state = ANIM_FINISHED
    # elif anim_state == ANIM_FINISHED:
    #     dof_positions[current_dof] = defaults[current_dof]
    #     current_dof = (current_dof + 1) % num_dofs
    #     anim_state = ANIM_SEEK_LOWER
    #     print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

    if args.show_axis:
        gym.clear_lines(viewer)

    # clone actor state in all of the environments
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

        if args.show_axis:
            # get the DOF frame (origin and axis)
            dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i], current_dof)
            frame = gym.get_dof_frame(envs[i], dof_handle)

            # draw a line from DOF origin along the DOF axis
            p1 = frame.origin
            p2 = frame.origin + frame.axis * 0.7
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    # d = torch_camera_tensor.cpu().numpy()
    # d[d == -np.inf] = 0
    # d[d < -10] = -10
    # norm_d = ((d / np.min(d + 1e-4)) * 255).astype(np.uint8)
    # imageio.imsave(f"/home/blau/tmp.png", norm_d)
    # gym.end_access_image_tensors(sim)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
