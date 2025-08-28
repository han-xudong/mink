from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink
from phone_subscriber.phone import PhoneSubscriber
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)
    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=10.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        ),
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=10.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1),
        com_task := mink.ComTask(cost=200.0),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    # Enable collision avoidance between the following geoms.
    # left hand - table, right hand - table
    # left hand - left thigh, right hand - right thigh
    collision_pairs = [
        # (["left_hand_collision", "right_hand_collision"], ["table"]),
        (["left_hand_collision"], ["left_thigh"]),
        (["right_hand_collision"], ["right_thigh"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    limits = [
        mink.ConfigurationLimit(model),
        collision_avoidance_limit,
    ]

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    solver = "daqp"
    
    left_phone_subscriber = PhoneSubscriber(host="192.168.31.208", port=8000)
    right_phone_subscriber = PhoneSubscriber(host="192.168.31.194", port=8000)
    # x-right, y-up, z-back to x-forward, y-left, z-up
    phone_pose_conv_mat = np.array(
        [
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    def convert_pose(pose: np.ndarray, conversion_matrix: np.ndarray) -> np.ndarray:
        """
        Convert a pose by applying a conversion matrix.

        Args:
            pose (np.ndarray): The pose to convert, expected to be a 7-element array (x, y, z, qx, qy, qz, qw).
            conversion_matrix (np.ndarray): The conversion matrix to apply.

        Returns:
            np.ndarray: The converted pose, which is also a 7-element array (x, y, z, qx, qy, qz, qw).
        """

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        mat[:3, 3] = pose[:3]

        converted_mat = conversion_matrix @ mat
        converted_pose = np.zeros_like(pose, dtype=np.float32)
        converted_pose[:3] = converted_mat[:3, 3]
        converted_pose[3:] = R.from_matrix(converted_mat[:3, :3]).as_quat()
        return converted_pose

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("stand")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]
        
        init_com_pos = data.mocap_pos[com_mid].copy()
        init_left_hand_pos = data.mocap_pos[hands_mid[1]].copy()
        init_left_hand_quat = data.mocap_quat[hands_mid[1]].copy()
        init_right_hand_pos = data.mocap_pos[hands_mid[0]].copy()
        init_right_hand_quat = data.mocap_quat[hands_mid[0]].copy()

        print("Initial COM position:", init_com_pos)
        print("Initial left hand position:", init_left_hand_pos)
        print("Initial right hand position:", init_right_hand_pos)
        print("Initial left hand orientation:", init_left_hand_quat)
        print("Initial right hand orientation:", init_right_hand_quat)
        
        viewer.sync()
        
        while True:
            try:
                _, _, _, _, _, _, left_global_pose = left_phone_subscriber.subscribeMessage()
                left_global_pose = convert_pose(np.array(left_global_pose, dtype=np.float32), phone_pose_conv_mat)
                break
            except Exception as e:
                print("Waiting for left phone pose:", e)
        while True:
            try:
                _, _, _, _, _, _, right_global_pose = right_phone_subscriber.subscribeMessage()
                right_global_pose = convert_pose(np.array(right_global_pose, dtype=np.float32), phone_pose_conv_mat)
                break
            except Exception as e:
                print("Waiting for right phone pose:", e)
        try:
            _, _, _, _, _, _, left_global_pose = left_phone_subscriber.subscribeMessage()
            left_global_pose = convert_pose(np.array(left_global_pose, dtype=np.float32), phone_pose_conv_mat)
            # init_left_phone_pos = np.array(left_global_pose[:3])
            init_left_phone_pose = np.array(left_global_pose)
            _, _, _, _, _, _, right_global_pose = right_phone_subscriber.subscribeMessage()
            right_global_pose = convert_pose(np.array(right_global_pose, dtype=np.float32), phone_pose_conv_mat)
            # init_right_phone_pos = np.array(right_global_pose[:3])
            init_right_phone_pose = np.array(right_global_pose)
        except Exception as e:
            print("Failed to get initial phone pose:", e)
            init_left_phone_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            init_right_phone_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        print("Initial left phone pose:", init_left_phone_pose)
        print("Initial right phone pose:", init_right_phone_pose)

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            com_task.set_target(data.mocap_pos[com_mid])
            
            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
            #     hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            try:
                _, _, _, _, _, _, left_global_pose = left_phone_subscriber.subscribeMessage()
                left_global_pose = convert_pose(np.array(left_global_pose, dtype=np.float32), phone_pose_conv_mat)
                # left_phone_pos = left_global_pose[:3] - init_left_phone_pos
                left_phone_pose = left_global_pose - init_left_phone_pose

                _, _, _, _, _, _, right_global_pose = right_phone_subscriber.subscribeMessage()
                right_global_pose = convert_pose(np.array(right_global_pose, dtype=np.float32), phone_pose_conv_mat)
                # right_phone_pos = right_global_pose[:3] - init_right_phone_pos
                right_phone_pose = right_global_pose - init_right_phone_pose
            except Exception as e:
                print("Failed to get phone pose:", e)
                left_phone_pose = init_left_phone_pose
                right_phone_pose = init_right_phone_pose

            left_hand_target_translation = init_left_hand_pos + left_phone_pose[:3]
            right_hand_target_translation = init_right_hand_pos + right_phone_pose[:3]
            # xyzw to wxyz
            left_hand_target_rotation = init_left_hand_quat + np.array([left_phone_pose[6], left_phone_pose[3], left_phone_pose[4], left_phone_pose[5]])
            right_hand_target_rotation = init_right_hand_quat + np.array([right_phone_pose[6], right_phone_pose[3], right_phone_pose[4], right_phone_pose[5]])
            
            for i, hand_task in enumerate(hand_tasks):
                hand_task.set_target(mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=left_hand_target_rotation if i == 1 else right_hand_target_rotation),
                    translation=left_hand_target_translation if i == 1 else right_hand_target_translation,
                ))
            
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            # mujoco.mj_fwdPosition(model, data)
            # mujoco.mj_sensorPos(model, data)
            
            for hand in hands:
                mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
                
            for foot in feet:
                mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
