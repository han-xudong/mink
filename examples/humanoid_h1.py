from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
from matplotlib import pyplot as plt
import mink
from phone_subscriber.phone import PhoneSubscriber
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_h1" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    feet = ["right_foot", "left_foot"]
    hands = ["right_wrist", "left_wrist"]

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(model, cost=1.0),
        com_task := mink.ComTask(cost=200.0),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
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
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    phone_subscriber = PhoneSubscriber(host="192.168.31.208", port=8000)
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
        model=model, data=data, #show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("stand")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]
        
        init_com_pos = data.mocap_pos[com_mid].copy()
        init_left_foot_pos = data.mocap_pos[feet_mid[1]].copy()
        init_right_foot_pos = data.mocap_pos[feet_mid[0]].copy()
        init_left_hand_pos = data.mocap_pos[hands_mid[1]].copy()
        init_right_hand_pos = data.mocap_pos[hands_mid[0]].copy()
        print("Initial COM position:", init_com_pos)
        print("Initial left foot position:", init_left_foot_pos)
        print("Initial right foot position:", init_right_foot_pos)
        print("Initial left hand position:", init_left_hand_pos)
        print("Initial right hand position:", init_right_hand_pos)
        
        _, _, _, _, _, _, global_pose = phone_subscriber.subscribeMessage()
        global_pose = convert_pose(np.array(global_pose, dtype=np.float32), phone_pose_conv_mat)
        init_phone_pos = np.array(global_pose[:3])

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            com_task.set_target(data.mocap_pos[com_mid])
            
            _, _, _, _, _, _, global_pose = phone_subscriber.subscribeMessage()
            global_pose = convert_pose(np.array(global_pose, dtype=np.float32), phone_pose_conv_mat)
            phone_pos = global_pose[:3] - init_phone_pos
            
            left_foot_target = init_left_foot_pos + np.array([0.0, 0.0, 0.0])
            right_foot_target = init_right_foot_pos + np.array([0.0, 0.0, 0.0])
            left_hand_target = init_left_hand_pos + np.array([0.0, 0.0, 0.0])
            right_hand_target = init_right_hand_pos + phone_pos
            
            # for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                # foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                # hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))
            for i, (foot_task, hand_task) in enumerate(zip(feet_tasks, hand_tasks)):
                foot_task.set_target(mink.SE3.from_translation(left_foot_target if i == 1 else right_foot_target))
                hand_task.set_target(mink.SE3.from_translation(left_hand_target if i == 1 else right_hand_target))

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)
            
            for hand, foot in zip(hands, feet):
                mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
                mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")

            
            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
