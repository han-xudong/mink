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
_XML = _HERE / "ballbot" / "scene.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    hands = ["right_wrist", "left_wrist"]

    tasks = [
        base_orientation_task := mink.FrameTask(
            frame_name="base_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(model, cost=10.0),
        com_task := mink.ComTask(cost=200.0),
    ]

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=0.0,
            lm_damping=2.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)
    
    head_tasks = []
    for head in ["head_link"]:
        task = mink.FrameTask(
            frame_name=head,
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
            lm_damping=5.0,
        )
        head_tasks.append(task)
    tasks.extend(head_tasks)

    com_mid = model.body("com_target").mocapid[0]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]
    head_mid = model.body("head_target").mocapid[0]

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
        model=model, data=data, #show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        # configuration.update_from_keyframe("stand")
        configuration.update_from_keyframe("fix-base")
        posture_task.set_target_from_configuration(configuration)
        base_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for hand in hands:
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        mink.move_mocap_to_frame(model, data, "head_target", "head_link", "body")
        data.mocap_pos[com_mid] = data.subtree_com[1]
        
        init_com_pos = data.mocap_pos[com_mid].copy()
        init_left_hand_pos = data.mocap_pos[hands_mid[1]].copy()
        init_left_hand_quat = data.mocap_quat[hands_mid[1]].copy()
        init_right_hand_pos = data.mocap_pos[hands_mid[0]].copy()
        init_right_hand_quat = data.mocap_quat[hands_mid[0]].copy()
        init_head_pos = data.mocap_pos[head_mid].copy()
        init_head_quat = data.mocap_quat[head_mid].copy()

        print("Initial COM position:", init_com_pos)
        print("Initial left hand position:", init_left_hand_pos)
        print("Initial right hand position:", init_right_hand_pos)
        print("Initial left hand orientation:", init_left_hand_quat)
        print("Initial right hand orientation:", init_right_hand_quat)
        print("Initial head position:", init_head_pos)
        print("Initial head orientation:", init_head_quat)
        
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

            head_pitch = 0.0
            head_yaw = 0.0
            
            for i, hand_task in enumerate(hand_tasks):
                hand_task.set_target(mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=left_hand_target_rotation if i == 1 else right_hand_target_rotation),
                    translation=left_hand_target_translation if i == 1 else right_hand_target_translation,
                ))

            head_tasks[0].set_target(mink.SE3.from_rotation(mink.SO3.from_rpy_radians(roll=0.0, pitch=head_pitch, yaw=head_yaw)))
            
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            for hand in hands:
                mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")

            # print(vel[:10])
            
            # Visualize at fixed FPS..
            viewer.sync()
            rate.sleep()
