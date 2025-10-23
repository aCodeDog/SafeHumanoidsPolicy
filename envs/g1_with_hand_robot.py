
import numpy as np
import warp as wp
import random
import math
import trimesh
import matplotlib.pyplot as plt

# Try to import keyboard, but handle the case where it's not available or requires root
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except (ImportError, OSError):
    KEYBOARD_AVAILABLE = False
    print("Warning: keyboard module not available or requires root privileges. Keyboard control disabled.")

from utils.utils import to_torch
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import torch
from utils.utils import *
import genesis as gs
from envs.base_task import BaseTask
from sensors.warp.warp_kernels.warp_lidar_kernels import LidarWarpKernels
class G1HandRobot(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.enable_warp = True
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.headless = headless
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self._init_buffers()
        
        # Warp and sensor initialization
        self.draw_lidar = True
        if self.enable_warp:
            self._init_sensor_buffer()
        
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # Enhanced action processing for sim2sim compatibility
        actions_scaled = self.actions * self.cfg.control.action_scale
        target_dof_pos = actions_scaled + self.default_dof_pos
        
        for _ in range(self.cfg.control.decimation):
            if hasattr(self, '_apply_action_enhanced'):
                self._apply_action_enhanced(target_dof_pos)
            else:
                self._apply_action(self.actions)
            self.scene.step()

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _apply_action_enhanced(self, target_dof_pos):
        """Enhanced action application for sim2sim compatibility"""
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
    def update_buffer(self):
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat() # x, y, z, w
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=torch.float,
        )
        self.root_states = torch.cat((self.base_pos[:], self.base_quat[:], self.base_lin_vel[:], self.base_ang_vel[:]), dim=-1)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """        
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.update_buffer()
        self._post_physics_step_callback()

        # Warp sensor update
        if self.enable_warp:
            if self.cfg.sensor_config.use_warp and self.common_step_counter % self.cfg.sensor_config.warp_update_freq == 0:
                self.lidar_pos_tensor[:,:] = torch.cat((self.root_states[:,:2].view(self.num_envs,1,2), torch.ones(self.num_envs,1,1,device=self.device)*1.20,), dim=-1)
                self.lidar_position_array = wp.from_torch(self.lidar_pos_tensor,dtype=wp.vec3)
                self.lidar_quat_tensor[:,:] = torch.cat([self.root_states[:,4:7],self.root_states[:,3:4]],dim=-1).view(self.num_envs,1,4)
                self.lidar_quat_array = wp.from_torch(self.lidar_quat_tensor, dtype=wp.quat)

                self.sensor_update()
                if self.draw_lidar:
                    self._draw_debug_lidar()    
                self.update_warp_mesh()

        # compute observations, rewards, resets, ...
        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        

        self.get_observations()


        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        
        self.reset_buf |= self.time_out_buf
        # active_reset = keyboard.is_pressed('4')
        # self.reset_buf |= active_reset

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}



    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        mesh_type = self.cfg.terrain.mesh_type
        gs.init(logging_level="warning")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=False,
            ),
            show_viewer=self.show_viewer,
        )

        # Create terrain
        if mesh_type in ['heightfield', 'trimesh']:
            raise NotImplementedError("Heightfield and trimesh terrains are not yet supported")
        if mesh_type=='plane':
            if hasattr(self, 'LEGGED_GYM_ROOT_DIR'):
                envs_asset_path = f"{self.LEGGED_GYM_ROOT_DIR}/resources/envs/plane/plane.urdf"
            else:
                envs_asset_path = "assets/plane.urdf"
            self.terrain = self.scene.add_entity(gs.morphs.URDF(file=envs_asset_path, fixed=True))
        elif mesh_type=='heightfield':
            pass
        elif mesh_type=='trimesh':
            pass
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
         
        # Create warp environment if enabled
        if self.enable_warp:
            self._create_warp_env()
        
        self._create_envs()


    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )

        self.dof_vel_limits = torch.zeros_like(self.dof_pos_limits)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.asset.max_joint_velocity:
                if dof_name in name:
                    self.dof_vel_limits[i] = self.cfg.asset.max_joint_velocity[dof_name]
                    found = True
            if not found:
                self.dof_vel_limits[i] = 200.

        for i in range(self.dof_vel_limits.shape[0]):
            # soft limits
            m = (self.dof_vel_limits[i, 0] + self.dof_vel_limits[i, 1]) / 2
            r = self.dof_vel_limits[i, 1] - self.dof_vel_limits[i, 0]
            self.dof_vel_limits[i, 0] = (
                m - 0.5 * r * self.cfg.rewards.soft_dof_vel_limit
            )
            self.dof_vel_limits[i, 1] = (
                m + 0.5 * r * self.cfg.rewards.soft_dof_vel_limit
            )

        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            #self.measured_heights = self._get_heights()
            raise NotImplementedError("Measuring heights not yet implemented")
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            raise NotImplementedError("Pushing robots not yet implemented")

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    def _apply_action(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            target_dof_pos = actions_scaled + self.default_dof_pos
            self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
            
        elif control_type=="V":
            raise NotImplementedError("Velocity control not yet implemented")
        elif control_type=="T":
            raise NotImplementedError("Torque control not yet implemented")
        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos #
        self.dof_vel[env_ids] = 0.

        self.robot.set_dofs_position(
            position=self.dof_pos[env_ids],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=env_ids,
        )
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """

        self.base_pos[env_ids] = self.base_init_pos

        self.base_quat[env_ids] = self.base_init_quat.reshape(1, -1)



        self.robot.set_pos(
            self.base_pos[ env_ids], zero_velocity=False,  envs_idx= env_ids
        )
        self.robot.set_quat(
            self.base_quat[ env_ids], zero_velocity=False,  envs_idx= env_ids
        )
        self.robot.zero_all_dofs_velocity( env_ids)





    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """

        # self.gym.refresh_net_contact_force_tensor(self.sim)
        self.proprio_obs = torch.zeros(
            (self.num_envs, 123), device=self.device, dtype=gs.tc_float
        )
        self.policy_obs = torch.zeros(
            (self.num_envs, 771), device=self.device, dtype=gs.tc_float
        )
        self.proprio_obs_buf = torch.zeros(
            (self.num_envs, 9, 123), device=self.device, dtype=gs.tc_float
        )
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float).repeat(
            self.num_envs, 1
        )
        self.contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )

        self.dof_pos = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=gs.tc_float
        )
        self.dof_vel = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=gs.tc_float
        )
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        



        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)

        self.root_states = torch.cat((self.base_pos[:], self.base_quat[:], self.base_lin_vel[:], self.base_ang_vel[:]), dim=-1)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)


        self.root_vel = self.root_states[:, 7:13]
        self.last_root_vel = self.root_states[:, 7:13]
        self._process_dof_props()
        
        if self.cfg.terrain.measure_heights:
            raise NotImplementedError("Measuring heights not yet implemented")
        self.measured_heights = 0



    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = "assets/g1_new/g1_29dof_fakehand.urdf"


        self.base_init_pos = to_torch(self.cfg.init_state.pos,device=self.device, requires_grad=False)
        self.base_init_quat = to_torch(self.cfg.init_state.rot, device=self.device, requires_grad=False)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=asset_path,
                links_to_keep=self.cfg.asset.links_to_keep,
                merge_fixed_links=True,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )


        self.scene.build(n_envs=self.num_envs)
        #self.dof_names_auto=[]
        if self.cfg.init_state.dof_names is not None:
            self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.cfg.init_state.dof_names]
        else:
            raise ValueError("DOF names not defined in the config file")

        #print("####Dof names auto 2: ", self.dof_names_auto)
        self.num_bodies = self.robot.n_links
        self.dof_names=[]
        self.joints_obj = self.robot.joints
        for joint in self.joints_obj:
            if joint.name in self.cfg.init_state.dof_names:
                self.dof_names.append(joint.name)
        if len(self.dof_names)!=len(self.motor_dofs):
            raise ValueError("Number of DOF names and motor DOFs do not match!")
        self.num_dofs = len(self.dof_names) #minux base joint 
        self.num_dof = self.num_dofs
        
        
        
        body_names=[]
        self.links_obj = self.robot.links
        self.body_names = [link.name for link in self.links_obj]
        print("####Body names: ", self.body_names)
        print("####Dof names : ", self.dof_names)
        # PD control parameters

        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.cfg.init_state.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
                    print(f"PD gains of joint {name} set to {self.p_gains[i]} / {self.d_gains[i]}")
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)
        
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.feet_indices_world_frame = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        index = 0
        for name in feet_names:
            self.feet_indices[index] = self.robot.get_link(name).idx_local
            self.feet_indices_world_frame[index] = self.robot.get_link(name).idx
            index=index+1
        assert len(self.feet_indices) > 0

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.penalised_contact_indices_world_frame = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        index = 0
        for name in penalized_contact_names:
            self.penalised_contact_indices[index] = self.robot.get_link(name).idx_local
            self.penalised_contact_indices_world_frame[index] = self.robot.get_link(name).idx
            index = index + 1


        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.termination_contact_indices_world_frame = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        index = 0
        for name in termination_contact_names:
            self.termination_contact_indices[index] = self.robot.get_link(name).idx_local
            self.termination_contact_indices_world_frame[index] = self.robot.get_link(name).idx
            index = index + 1

    

    def find_link_indices(self,names):
        link_indices = list()
        for link in self.robot.links:
            flag = False
            for name in names:
                if name in link.name:
                    flag = True
            if flag:
                link_indices.append(link.idx - self.robot.link_start)

        return link_indices

    # Sensor and Warp related methods
    def _draw_debug_lidar(self):
        pixels = wp.to_torch(self.pixels).clone().detach() # shape: (num_envs, 1, 18, 36, 3)
        pixels = pixels.view(self.num_envs,-1,3)
        pixels_num = pixels.shape[1]
        poss = []  

        for i in range(1):  # Only process the first environment
           for j in range(pixels_num):
                    x = pixels[i, j,0]
                    y = pixels[i, j,1]
                    z = pixels[i, j,2]
                    pos = [x, y, z]
                    poss.append(pos)  

        poss = torch.tensor(poss, device=self.device)
        poss = poss.view(-1, 3)
        self.scene.clear_debug_objects()
        self.scene.draw_debug_spheres(poss=poss, radius=0.03, color=(1.0, 0.0, 0.0, 0.5))

    def _init_sensor_buffer(self):
        self.root_states_all = torch.zeros(self.num_envs, 1+self.cfg.asset.object_num, 13, device=self.device, dtype=gs.tc_float)
        self.object_states_all_list = []
        for object_handles in self.object_handles:
            object_pos = object_handles.get_pos()
            object_quat = object_handles.get_quat()
            object_lin_vel = torch.zeros(1, 3, device=self.device)
            object_ang_vel = torch.zeros(1, 3, device=self.device)
            object_states = torch.cat((object_pos[:], object_quat[:], object_lin_vel[:], object_ang_vel[:]), dim=-1)
            self.object_states_all_list.append(object_states)
        self.obstacle_states_all = torch.stack(self.object_states_all_list, dim=1)
        self.root_states_all[:,0,:] = self.root_states
        self.root_states_all[:,1:,:] = self.obstacle_states_all

        self.vec_root_tensor = self.root_states_all.view(
            self.num_envs,1+self.cfg.asset.object_num, -1
        )
        obstacle_num = self.cfg.asset.object_num
        self.obstacle_root_states = self.vec_root_tensor[:,1:obstacle_num+1,:]
        self.obstacle_states_order = self.obstacle_root_states.clone().reshape(self.num_envs*obstacle_num, -1)
        self.init_obstacle_root_states = self.obstacle_states_order.clone()
        self.init_states_translation = self.init_obstacle_root_states[:, :3]
        self.expanded_init__translation = self.init_states_translation.repeat_interleave(self.single_num_vertices, dim=0)
        self.sensor_t = 0
        self.graph = None
        
        self.random__obstacles_offsets = (torch.rand(self.cfg.asset.object_num, 3, device=self.device) - 0.5) * 6.28
        
        self.sensor_init()

    def _create_warp_env(self):
        """ Create Warp environment
        """
        wp.init()
        #create sensors
        self.object_handles = []
        self.warp_meshes_trasnformation =[]
        self.warp_meshes_list = []
        
        self.global_sim_dict = {}
        self.global_tensor_dict={}

        self.warp_mesh_per_env = []
        self.warp_mesh_id_list = []
        per_env_obstacle_transformations = []
        per_env_obstacle_meshes =[]        
        if True:
            obstacle_path_mesh = f"assets/humanoid/meshes/Group1.obj"
            obstacle_path_asset = f"assets/humanoid/humanoid.urdf"
            x_range = (-8.0, 8.0)  # x coordinate range
            y_range = (-8.0, 8.0)  # y coordinate range
            z_height = 0

            #for i in range(self.num_envs):
            for _obj in range(self.cfg.asset.object_num):
                x = random.uniform(*x_range)
                y = random.uniform(*y_range)
                pos = np.array([x, y, z_height])
                quat = np.array([1.0, 0.0, 0.0, 0.0])
                object_handle = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=obstacle_path_asset,
                        fixed=True,
                        pos=pos,
                        quat=quat,
                    ),
                )

                per_env_obstacle_transformations.append(pos)
                asset_path = obstacle_path_mesh
                per_env_obstacle_meshes.append(asset_path)

                self.object_handles.append(object_handle)  

                self.warp_meshes_list.append(per_env_obstacle_meshes)
                self.warp_meshes_trasnformation.append(per_env_obstacle_transformations)     
    
        # Warp
        self.create_obstacles_warp_mesh(self.warp_meshes_list,self.warp_meshes_trasnformation)
        
        self.warp_sensor_config = None
        if self.cfg.sensor_config.enable_camera:
            self.warp_sensor_config = self.cfg.sensor_config.camera_config
        elif self.cfg.sensor_config.enable_lidar:
            self.warp_sensor_config = self.cfg.sensor_config.lidar_config

        if self.warp_sensor_config is not None:
            # prepare the tensors for simulation before preparing the tensors for the sensors
            image_tensor_dims = 3 * (self.warp_sensor_config.return_pointcloud == True)
            if self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] is None:
                print(
                    "Warp camera is enabled but there is nothing in the environment. No rendering will take place and the camera tensor will not be populated."
                )
            else:
                if image_tensor_dims == 0:
                    self.image_tensor = torch.zeros(
                        (
                            self.num_envs,
                            self.warp_sensor_config.num_sensors,
                            self.warp_sensor_config.height,
                            self.warp_sensor_config.width,
                        ),
                        device=self.device,
                        requires_grad=False,
                    )
                else:
                    self.image_tensor = torch.zeros(
                        (
                            self.num_envs,  #4
                            self.warp_sensor_config.num_sensors, #1
                            self.warp_sensor_config.height, #128
                            self.warp_sensor_config.width, #512
                            image_tensor_dims, #3
                        ),
                        device=self.device,
                        requires_grad=False,
                    )
                    self.sensor_dist_tensor = torch.zeros(
                            (
                                self.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                            ),
                            device=self.device,
                            requires_grad=False,
                        )
                self.global_tensor_dict["depth_range_pixels"] = self.image_tensor
                self.global_tensor_dict["dist_pixels"] = self.sensor_dist_tensor

    def sensor_update(self):
        if self.graph is None:
            if self.cfg.sensor_config.enable_lidar:
                if self.cfg.sensor_config.lidar_config.return_pointcloud:
                    self.create_render_lidar_graph_pointcloud()
                else:
                    NotImplementedError
        if self.graph is not None:
            wp.capture_launch(self.graph)
        return wp.to_torch(self.pixels)

    def create_render_lidar_graph_pointcloud(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.synchronize() 
            wp.capture_begin(device=self.device)
            wp.launch(
                kernel=LidarWarpKernels.draw_optimized_kernel_pointcloud,
                dim=(
                    self.num_envs,
                    self.num_sensors,
                    self.num_scan_lines,
                    self.num_points_per_line,
                ),
                inputs=[
                    self.mesh_ids_array,
                    self.lidar_position_array,
                    self.lidar_quat_array,
                    self.ray_vectors,
                    self.far_plane,
                    self.pixels,
                    self.local_dist,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)
            wp.synchronize() 

    def create_obstacles_warp_mesh(self,obstacle_meshes,obstacle_transformations):
        self.warp_mesh_per_env =[]
        self.warp_mesh_id_list = []
        self.global_tensor_dict = {}
        self.obstacle_mesh_per_env =[]
        self.obstacle_vertex_indices = [] 
        self.obstacle_indices_per_vertex_list = []
        num_obstacles = self.cfg.asset.object_num

        self.single_num_vertices_list = []
        self.all_obstacles_points = []
        for i in range(self.num_envs):
            for j in range(num_obstacles):
                mesh_path = obstacle_meshes[i][j]
                obstacle_mesh = trimesh.load(mesh_path)

                angle_x_degrees = 90
                angle_x_radians = np.radians(angle_x_degrees)
                rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x_radians, [1, 0, 0])
                obstacle_mesh.apply_transform(rotation_matrix_x)

                translation = trimesh.transformations.translation_matrix(obstacle_transformations[i][j])
                
                obstacle_mesh.apply_transform(translation)
                self.obstacle_mesh_per_env.append(obstacle_mesh)
                obstacle_points = obstacle_mesh.vertices
                self.all_obstacles_points.append(obstacle_points)
                
                self.single_num_vertices_list.append(len(obstacle_mesh.vertices))

        terrain_mesh = trimesh.load("assets/plane100.obj")
        transform = np.zeros((3,))
        transform[0] = -25
        transform[1] = -25 # TODO
        transform[2] = 0.0
        translation = trimesh.transformations.translation_matrix(transform)
        terrain_mesh.apply_transform(translation)
        
        combined_mesh = trimesh.util.concatenate(self.obstacle_mesh_per_env+[terrain_mesh])
        output_filename = "assets/combined_terrain.stl"
        combined_mesh.export(output_filename)

        print(f"Combined mesh saved to {output_filename}")
        vertices = combined_mesh.vertices
        triangles = combined_mesh.faces

        vertex_tensor = torch.tensor(
                vertices,
                device=self.device,
                requires_grad=False,
                dtype=torch.float32,
            )           
        vertex_vec3_array = wp.from_torch(
            vertex_tensor,
            dtype=wp.vec3,
        )
        faces_wp_int32_array = wp.from_numpy(triangles.flatten(), dtype=wp.int32,device=self.device)
                
        self.wp_mesh =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
        
        old_points = wp.to_torch(self.wp_mesh.points)
        self.init_points = old_points.clone()

        self.warp_mesh_per_env.append(self.wp_mesh)
        self.warp_mesh_id_list.append(self.wp_mesh.id)
        
        self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = self.warp_mesh_id_list
        self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = self.warp_mesh_per_env
        
        self.single_num_vertices = torch.tensor(self.single_num_vertices_list, device=self.device)
        self.all_obstacle_num_vertices = torch.sum(self.single_num_vertices)

    def sensor_init(self):
        self.mesh_ids_array = wp.array(self.warp_mesh_id_list, dtype=wp.uint64)
        
        if self.cfg.sensor_config.enable_lidar:
            self.initialize_lidar_parameters()
            self.initialize_ray_vectors()

    def initialize_ray_vectors(self):
        ray_vectors = torch.zeros(
            (self.num_scan_lines, self.num_points_per_line, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_scan_lines):
            for j in range(self.num_points_per_line):
                azimuth_angle = self.horizontal_fov_max - (
                    self.horizontal_fov_max - self.horizontal_fov_min
                ) * (j / (self.num_points_per_line - 1))
                elevation_angle = self.vertical_fov_max - (
                    self.vertical_fov_max - self.vertical_fov_min
                ) * (i / (self.num_scan_lines - 1))
                ray_vectors[i, j, 0] = math.cos(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 1] = math.sin(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 2] = math.sin(elevation_angle)
        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
        self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)

    def initialize_lidar_parameters(self):
        pixels = self.global_tensor_dict["depth_range_pixels"]
        dist_tensor = self.global_tensor_dict["dist_pixels"]
        self.pixels = wp.from_torch(pixels, dtype=wp.vec3)
        
        self.local_dist = wp.from_torch(dist_tensor, dtype=wp.float32)
        self.pointcloud_in_world_frame = self.cfg.sensor_config.lidar_config.pointcloud_in_world_frame
        self.num_sensors = self.cfg.sensor_config.lidar_config.num_sensors
        self.num_scan_lines = self.cfg.sensor_config.lidar_config.height
        self.num_points_per_line = self.cfg.sensor_config.lidar_config.width
        self.horizontal_fov_min = math.radians(self.cfg.sensor_config.lidar_config.horizontal_fov_deg_min)
        self.horizontal_fov_max = math.radians(self.cfg.sensor_config.lidar_config.horizontal_fov_deg_max)
        self.horizontal_fov = self.horizontal_fov_max - self.horizontal_fov_min
        self.horizontal_fov_mean = (self.horizontal_fov_max + self.horizontal_fov_min) / 2
        if self.horizontal_fov > 2 * math.pi:
            raise ValueError("Horizontal FOV must be less than 2pi")

        self.vertical_fov_min = math.radians(self.cfg.sensor_config.lidar_config.vertical_fov_deg_min)
        self.vertical_fov_max = math.radians(self.cfg.sensor_config.lidar_config.vertical_fov_deg_max)
        self.vertical_fov = self.vertical_fov_max - self.vertical_fov_min
        self.vertical_fov_mean = (self.vertical_fov_max + self.vertical_fov_min) / 2
        
        if self.vertical_fov > math.pi:
            raise ValueError("Vertical FOV must be less than pi")
        self.far_plane = self.cfg.sensor_config.lidar_config.max_range

        if self.cfg.sensor_config.use_local_z_axis:
            self.lidar_pos_tensor = torch.cat((self.root_states[:,:2].view(self.num_envs,1,2), torch.ones(self.num_envs,1,1,device=self.device)*1.20,), dim=-1)
            self.lidar_position_array = wp.from_torch(self.lidar_pos_tensor,dtype=wp.vec3)
            self.lidar_quat_tensor = torch.cat([self.root_states[:,4:7],self.root_states[:,3:4]],dim=-1).reshape(self.num_envs,1,4)
            self.lidar_quat_array = wp.from_torch(self.lidar_quat_tensor, dtype=wp.quat)

    def update_warp_mesh(self):
        self.sensor_t += self.dt
        old_points = wp.to_torch(self.wp_mesh.points)
        
        # Get new transforms [total_obstacles, 3]
        self.obstacle_root_states = self.vec_root_tensor[:,1:self.num_envs*self.cfg.asset.object_num+1,:]
        self.obstacle_states_order = self.obstacle_root_states.clone().reshape(self.num_envs*self.cfg.asset.object_num, -1)
        
        new_transforms = self.obstacle_states_order[:, :3]
        expanded_current_tensor = new_transforms.repeat_interleave(self.single_num_vertices, dim=0)

        # update warp mesh
        trans =expanded_current_tensor-self.expanded_init__translation
        old_points[:self.all_obstacle_num_vertices,:3] = self.init_points[:self.all_obstacle_num_vertices,:3] + trans[:,:3]

        self.wp_mesh.points = wp.from_torch(old_points, dtype=wp.vec3)
        self.wp_mesh.refit()
        
        # Move obstacles, delay one step
        self.object_states_all_list=[]
        obstacle_index = 0
        
        pos_param = torch.full((self.cfg.asset.object_num, 3), self.sensor_t*self.cfg.asset.obstacle_k, device=self.device)
        pos_param += self.random__obstacles_offsets
        diff_x =  torch.sin(pos_param[:,0]).view(-1,1) #or other func
        diff_y =  torch.cos(pos_param[:,1]).view(-1,1)
        diff_z = torch.zeros_like(diff_x)
        pos_diff = torch.cat((diff_x,diff_y,diff_z),dim=-1)
        for object_handles in self.object_handles:
            object_pos = self.init_obstacle_root_states[obstacle_index,:3].view(-1,3)+pos_diff[obstacle_index,:3]
            obstacle_index += 1
            object_handles.set_pos(object_pos)
            object_quat = object_handles.get_quat()
            object_lin_vel = torch.zeros(1, 3, device=self.device)
            object_ang_vel = torch.zeros(1, 3, device=self.device)
            object_states = torch.cat((object_pos[:], object_quat[:], object_lin_vel[:], object_ang_vel[:]), dim=-1)
            self.object_states_all_list.append(object_states)
        self.obstacle_states_all = torch.stack(self.object_states_all_list, dim=1)
        self.root_states_all[:,0,:] = self.root_states
        self.root_states_all[:,1:,:] = self.obstacle_states_all

        self.vec_root_tensor = self.root_states_all.view(
            self.num_envs,1+self.cfg.asset.object_num, -1
        )

    def downsample_spherical_points_vectorized(self, sphere_points, num_theta_bins=10, num_phi_bins=10):
        """Downsample points in spherical coordinates by binning theta and phi values."""
        num_envs = sphere_points.shape[0]
        num_points = sphere_points.shape[1]
        device = sphere_points.device
        num_bins = num_theta_bins * num_phi_bins
        
        # Define bin ranges
        theta_min, theta_max = -3.14, 3.14
        phi_min, phi_max = -0.12, 0.907
        
        # Extract r, theta, phi for all environments
        r = sphere_points[:, :, 0]
        theta = sphere_points[:, :, 1]
        phi = sphere_points[:, :, 2]
        
        # Compute bin indices
        theta_bin = ((theta - theta_min) / (theta_max - theta_min) * num_theta_bins).long()
        phi_bin = ((phi - phi_min) / (phi_max - phi_min) * num_phi_bins).long()
        
        theta_bin = torch.clamp(theta_bin, 0, num_theta_bins - 1)
        phi_bin = torch.clamp(phi_bin, 0, num_phi_bins - 1)
        
        bin_indices = theta_bin * num_phi_bins + phi_bin
        
        # Prepare tensors for scatter operations
        r_sum = torch.zeros(num_envs, num_bins, device=device)
        bin_count = torch.zeros(num_envs, num_bins, device=device)
        
        r_sum.scatter_add_(1, bin_indices, r)
        ones = torch.ones_like(r)
        bin_count.scatter_add_(1, bin_indices, ones)
        
        bin_count = torch.clamp(bin_count, min=1.0)
        avg_r = r_sum / bin_count
        
        # Create bin centers
        theta_centers = torch.linspace(
            theta_min + (theta_max - theta_min) / (2 * num_theta_bins),
            theta_max - (theta_max - theta_min) / (2 * num_theta_bins),
            num_theta_bins, device=device
        )
        
        phi_centers = torch.linspace(
            phi_min + (phi_max - phi_min) / (2 * num_phi_bins),
            phi_max - (phi_max - phi_min) / (2 * num_phi_bins),
            num_phi_bins, device=device
        )
        
        theta_grid, phi_grid = torch.meshgrid(theta_centers, phi_centers, indexing='ij')
        theta_centers_flat = theta_grid.reshape(-1)
        phi_centers_flat = phi_grid.reshape(-1)
        
        downsampled = torch.zeros(num_envs, num_bins, 3, device=device)
        downsampled[:, :, 0] = avg_r
        downsampled[:, :, 1] = theta_centers_flat.unsqueeze(0) + 3.14
        downsampled[:, :, 2] = phi_centers_flat.unsqueeze(0)
        
        return downsampled   

    def downsample_theta_bins_vectorized(self, sphere_points, num_theta_bins: int = 36):
        """Downsample points by binning theta values and averaging r across all phi values per theta bin"""
        num_envs = sphere_points.shape[0]
        device = sphere_points.device
        
        r = sphere_points[..., 0]
        theta = sphere_points[..., 1]
        
        theta_min = -torch.pi
        theta_max = torch.pi
        theta_bin = ((theta - theta_min) / (theta_max - theta_min) * num_theta_bins).long()
        theta_bin = torch.clamp(theta_bin, 0, num_theta_bins-1)
        
        r_sum = torch.zeros(num_envs, num_theta_bins, device=device)
        count = torch.zeros(num_envs, num_theta_bins, device=device)
        
        ones = torch.ones_like(r)
        r_sum.scatter_add_(1, theta_bin, r)
        count.scatter_add_(1, theta_bin, ones)
        
        avg_r = r_sum / torch.clamp(count, min=1.0)
        
        theta_centers = torch.linspace(
            theta_min + (theta_max-theta_min)/(2*num_theta_bins),
            theta_max - (theta_max-theta_min)/(2*num_theta_bins),
            num_theta_bins,
            device=device
        )
        
        downsampled = torch.zeros(num_envs, num_theta_bins, 3, device=device)
        downsampled[..., 0] = avg_r
        downsampled[..., 1] = theta_centers.unsqueeze(0)
        downsampled[..., 2] = 0.0

        if KEYBOARD_AVAILABLE:
            try:
                if keyboard.is_pressed('3'):
                    plt.ion()
                    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                    ax.set_title("Theta vs Radius (r)")
                    r_vis = downsampled[..., 0].cpu().numpy()
                    theta_vis = downsampled[..., 1].cpu().numpy()
                    ax.scatter(theta_vis[0], r_vis[0], c='b', s=10, label='r vs θ')
                    ax.legend()
                    plt.show()
                    plt.pause(0.001)
            except:
                pass  # Silently ignore keyboard errors

        return downsampled

    def theta_feature(self):
        """Extract theta-based lidar features"""
        if not hasattr(self, 'pixels'):
            return torch.zeros(self.num_envs, 36, device=self.device)
            
        pixels = wp.to_torch(self.pixels).clone().detach() 
        hit_vec_w = pixels.view(self.num_envs,-1,3)

        # Handle invalid points
        hit_vec_w[torch.isinf(hit_vec_w)] = 0.0
        hit_vec_w[torch.isnan(hit_vec_w)] = 0.0
        valid_mask = (torch.abs(hit_vec_w[..., 0]) <= 50) & (torch.abs(hit_vec_w[..., 1]) <= 50)
        hit_vec_w = hit_vec_w[valid_mask].view(self.num_envs, -1, 3)

        base_pos_w = self.base_pos.clone().detach()
        base_quat_w = self.base_quat.clone().detach()

        pos=(0.0002835, 0.00003, 0.41818)
        rot=(0.0, 1.0, 0.0, 0.0)
        sensor_offset_pos = torch.tensor(pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        sensor_pos_w = base_pos_w + quat_apply(base_quat_w, sensor_offset_pos)        
        sensor_offset_quat = torch.tensor(rot, device=base_quat_w.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        yaw_quat = extract_yaw_quaternion(base_quat_w)
        sensor_quat_w = quat_mul(yaw_quat, sensor_offset_quat)

        # Convert points to sensor frame
        hit_vec_s = hit_vec_w - sensor_pos_w.unsqueeze(1)
        hit_vec_s = quat_rotate_inverse(
            sensor_quat_w.unsqueeze(1).repeat(1, hit_vec_s.shape[1], 1).view(-1, 4),
            hit_vec_s.view(-1, 3)
        ).view(hit_vec_s.shape)
        
        # Convert to spherical coordinates
        sphere_points = cart2sphere(hit_vec_s.view(-1, 3)).view(self.num_envs, -1, 3)
        
        downsampled = self.downsample_theta_bins_vectorized(sphere_points, num_theta_bins=36)
        d = downsampled[..., 0]
        return d

    def get_observations(self):
        """Enhanced observation method with lidar features"""
        keyboard_control = True

        if self.enable_warp and hasattr(self, 'pixels'):
            self.visual_input = self.theta_feature()
        else:
            self.visual_input = torch.zeros(self.num_envs, 36, device=self.device)

        if keyboard_control and KEYBOARD_AVAILABLE:
            try:
                self.commands = torch.zeros((self.num_envs, 4), device=self.device)
                if keyboard.is_pressed('up'):
                    self.commands[:, 0] = 0.5
                elif keyboard.is_pressed('down'):
                    self.commands[:, 0] = -0.4
                elif keyboard.is_pressed('left'):
                    self.commands[:, 1] = 0.4
                elif keyboard.is_pressed('right'):
                    self.commands[:, 1] = -0.4

                if keyboard.is_pressed('1'):
                    self.commands[:, 2] = 0.5
                elif keyboard.is_pressed('2'):
                    self.commands[:, 2] = -0.5
            except:
                #print("Keyboard input error")
                pass  # Silently ignore keyboard errors

        self.commands[:, 0] = 0.5

        self.curr_policy_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos-self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.visual_input  
        ), dim=-1)
        self.obs_buf = self.curr_policy_buf

        return self.obs_buf

    def _parse_cfg(self, cfg):
        # print("self.sim_params.sim: ", self.sim_params["sim"])
        self.dt = self.cfg.control.decimation * self.sim_params["dt"]
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


