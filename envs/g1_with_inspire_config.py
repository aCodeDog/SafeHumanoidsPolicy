from envs.legged_robot_config import LeggedRobotCfg
from sensors.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig
class G1InspireCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.77] # x,y,z [m]
        rot = [1.0, 0.0, 0.0, 0.0] # w,x,y,z [quat]

        dof_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 
                     'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 
                     'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 
                     'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
                     'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 
                     'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
                     'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 
                     'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
   





        dof_indices = [
            0,  # LeftHipPitch
            3,  # LeftHipRoll
            6,  # LeftHipYaw
            9,  # LeftKnee
            13, # LeftAnklePitch
            17, # LeftAnkleRoll
            1,  # RightHipPitch
            4,  # RightHipRoll
            7,  # RightHipYaw
            10, # RightKnee
            14, # RightAnklePitch
            18, # RightAnkleRoll
            2,  # WaistYaw
            5,  # WaistRoll
            8,  # WaistPitch
            11, # LeftShoulderPitch
            15, # LeftShoulderRoll
            19, # LeftShoulderYaw
            21, # LeftElbow
            23, # LeftWristRoll
            25, # LeftWristPitch
            27, # LeftWristYaw
            12, # RightShoulderPitch
            16, # RightShoulderRoll
            20, # RightShoulderYaw
            22, # RightElbow
            24, # RightWristRoll
            26, # RightWristPitch
            28  # RightWristYaw
        ]

        

        default_joint_angles = { # = target angles [rad] when action = 0.0
            "left_hip_pitch_joint": -0.20,
            "right_hip_pitch_joint": -0.20,
            "waist_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "waist_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "waist_pitch_joint": 0.1,
            "left_knee_joint": 0.42,
            "right_knee_joint": 0.42,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_pitch_joint": 0.35,
            "left_ankle_pitch_joint": -0.23,
            "right_ankle_pitch_joint": -0.23,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,   
            "left_ankle_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.87,
            "right_elbow_joint": 0.87,
            "left_wrist_roll_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            
        }

    class env(LeggedRobotCfg.env):
        # num_observations = 123+459
        num_observations = 3*4+3*29+225
        num_actions = 29
        num_envs = 1
        episode_length_s = 100
        log_joint_limit_warnings = True
      
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            heading = [0.0,0.0]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {
            "left_hip_pitch_joint": 200.0,
            "right_hip_pitch_joint": 200.0,
            "waist_yaw_joint": 200.0,
            "left_hip_roll_joint": 150.0,
            "right_hip_roll_joint": 150.0,
            "waist_roll_joint": 200.0,
            "left_hip_yaw_joint": 150.0,
            "right_hip_yaw_joint": 150.0,
            "waist_pitch_joint": 200.0,
            "left_knee_joint": 200.0,
            "right_knee_joint": 200.0,
            "left_shoulder_pitch_joint": 40.0,
            "right_shoulder_pitch_joint": 40.0,
            "left_ankle_pitch_joint": 20.0,
            "right_ankle_pitch_joint": 20.0,
            "left_shoulder_roll_joint": 40.0,
            "right_shoulder_roll_joint": 40.0,
            "left_ankle_roll_joint": 20.0,
            "right_ankle_roll_joint": 20.0,
            "left_shoulder_yaw_joint": 40.0,
            "right_shoulder_yaw_joint": 40.0,
            "left_elbow_joint": 40.0,
            "right_elbow_joint": 40.0,
            "left_wrist_roll_joint": 40.0,
            "right_wrist_roll_joint": 40.0,
            "left_wrist_pitch_joint": 40.0,
            "right_wrist_pitch_joint": 40.0,
            "left_wrist_yaw_joint": 40.0,
            "right_wrist_yaw_joint": 40.0,
        }
        
        damping = {
            "left_hip_pitch_joint": 5.0,
            "right_hip_pitch_joint": 5.0,
            "waist_yaw_joint": 5.0,
            "left_hip_roll_joint": 5.0,
            "right_hip_roll_joint": 5.0,
            "waist_roll_joint": 5.0,
            "left_hip_yaw_joint": 5.0,
            "right_hip_yaw_joint": 5.0,
            "waist_pitch_joint": 5.0,
            "left_knee_joint": 5.0,
            "right_knee_joint": 5.0,
            "left_shoulder_pitch_joint": 10.0,
            "right_shoulder_pitch_joint": 10.0,
            "left_ankle_pitch_joint": 2.0,
            "right_ankle_pitch_joint": 2.0,
            "left_shoulder_roll_joint": 10.0,
            "right_shoulder_roll_joint": 10.0,
            "left_ankle_roll_joint": 2.0,
            "right_ankle_roll_joint": 2.0,
            "left_shoulder_yaw_joint": 10.0,
            "right_shoulder_yaw_joint": 10.0,
            "left_elbow_joint": 10.0,
            "right_elbow_joint": 10.0,
            "left_wrist_roll_joint": 10.0,
            "right_wrist_roll_joint": 10.0,
            "left_wrist_pitch_joint": 10.0,
            "right_wrist_pitch_joint": 10.0,
            "left_wrist_yaw_joint": 10.0,
            "right_wrist_yaw_joint": 10.0,
        }  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '/home/xunyang/Desktop/Projects/Genesis_Legged_Gym/resources/robots/g1_new/g1_29dof_fakehand.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        object_num = 60
        object_move_velocity = 2.0
        obstacle_k = 0.62 * object_move_velocity # 2*object_move_velocity/pi
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.728
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.01
            orientation = -2.0
            base_height = -10
            dof_acc = -2.5e-7
            feet_air_time = 3.0
            collision = 0.0
            action_rate = -0.01
            # torques = -0.0001
            dof_pos_limits = -0.1
            joint_pos = -0.1
            
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
            ray2d = 1.0
        clip_observations = 100.
        clip_actions = 100.
    class sensor_config:
        
        use_warp = True
        use_local_z_axis = True
        
        warp_update_freq = 1
        enable_camera = False

        enable_lidar = True
        lidar_config = BaseLidarConfig  # OSDome_64_Config

        enable_imu = False

    class terrain(LeggedRobotCfg.terrain):
        border_size = 25 # [m]

    class sensors:
        class ray2d:
            enable = True
            log2 = True
            min_dist = 0.1
            max_dist = 6.0
            theta_start = - 1
            theta_end = 1
            theta_step = 1/16
            x_0 = -0.05
            y_0 = 0.0
            front_rear = False
            illusion = True  # add illusion when there is noise
            raycolor = (0,0.5,0.5)
        
