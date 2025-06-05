# SO100 Real Robot
####python3 GR00T_clinet/GR00T_client.py --use_policy
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm

#################################################################################


class SO100Robot:
    def __init__(self, calibrate=False, enable_camera=False, camera_indices=(9,10,11)):  # 修改：接受一个摄像头索引的元组
        self.config = So100RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.camera_indices = camera_indices  # 保存摄像头索引
        self.top_camera_index = camera_indices[0]
        self.third_camera_index = camera_indices[1]
        self.wrist_camera_index = camera_indices[2]
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {
                "top": OpenCVCameraConfig(self.top_camera_index, 30, 640, 480, "bgr"),  # 创建顶置摄像头配置
                "third": OpenCVCameraConfig(self.third_camera_index, 30, 640, 480, "bgr"),  # 创建third摄像头配置
                "wrist": OpenCVCameraConfig(self.wrist_camera_index, 30, 640, 480, "bgr"),  # 创建腕部摄像头配置
            }
        self.config.leader_arms = {}

        # remove the .cache/calibration/so100 folder
        if self.calibrate:
            import os
            import shutil

            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]
        self.top_camera = None
        self.third_camera = None
        self.wrist_camera = None

    @contextmanager
    def activate(self):
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        self.motor_bus.connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("robot present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        # Connect the cameras
        if self.enable_camera:
            self.top_camera = self.robot.cameras["top"]
            self.third_camera = self.robot.cameras["third"]
            self.wrist_camera = self.robot.cameras["wrist"]
            self.top_camera.connect()
            self.third_camera.connect()
            self.wrist_camera.connect()

        print("================> SO100 Robot is fully connected =================")

    def set_so100_robot_preset(self):
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.motor_bus.write("P_Coefficient", 16)
        self.motor_bus.write("P_Coefficient", 10)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 0)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.motor_bus.write("Maximum_Acceleration", 254)
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        current_state = self.robot.capture_observation()["observation.state"]
        print("current_state", current_state)
        # print all keys of the observation
        print("observation keys:", self.robot.capture_observation().keys())

        # current_state[0] = 90
        # current_state[2] = 90
        # current_state[3] = 90
        # self.robot.send_action(current_state)
        time.sleep(2)

        # current_state[4] = -70
        # current_state[5] = 30
        # current_state[1] = 90
        # self.robot.send_action(current_state)
        time.sleep(2)

        print("----------------> SO100 Robot moved to initial pose")

    def go_home(self):
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("----------------> SO100 Robot moved to home pose")
        # home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        # self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self, camera_name="top"):  # 修改：添加 camera_name 参数
        if camera_name not in ["top", "third", "wrist"]:
            raise ValueError("Invalid camera name.  Must be 'top', 'third', or 'wrist'.")

        camera_key = f"observation.images.{camera_name}"
        img = self.get_observation()[camera_key].data.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        if self.enable_camera:
            self.top_camera.disconnect()
            self.third_camera.disconnect()
            self.wrist_camera.disconnect()
        print("================> SO100 Robot disconnected")

    def __del__(self):
        self.disconnect()


#################################################################################


class Gr00tRobotInferenceClient:  # 修改 Gr00tRobotInferenceClient 类
    def __init__(
        self,
        host="localhost",
        port=6006,
        language_instruction="Grasp red duck and place it in the box.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, top_img, third_img, wrist_img, state):  # 修改：添加 top_img, third_img, wrist_img 参数
        obs_dict = {
            "video.top": top_img[np.newaxis, :, :, :],  # 使用顶置摄像头图像
            "video.third": third_img[np.newaxis, :, :, :],  # 使用third摄像头图像
            "video.wrist": wrist_img[np.newaxis, :, :, :],  # 使用腕部摄像头图像
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        start_time = time.time()
        res = self.policy.get_action(obs_dict)
        print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):  # 修改 sample_action 方法
        obs_dict = {
            "video.top": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),  # 修改：顶置摄像头图像
            "video.third": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),  # 修改：third摄像头图像
            "video.wrist": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),  # 修改：腕部摄像头图像
            "state.single_arm": np.zeros((1, 5)),
            "state.gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

#################################################################################

def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame

#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("~/datasets/so100_strawberry_grape")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    parser.add_argument("--camera_index_top", type=int, default=4)
    parser.add_argument("--camera_index_third", type=int, default=14)
    parser.add_argument("--camera_index_wrist", type=int, default=6)
    args = parser.parse_args()

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["single_arm", "gripper"]

    if USE_POLICY:
        client = Gr00tRobotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction="Grasp red duck and place it in the box.",
        )

        robot = SO100Robot(calibrate=False, enable_camera=True, camera_indices=(args.camera_index_top, args.camera_index_third, args.camera_index_wrist))  # 修改：传入摄像头索引
        with robot.activate():
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
                top_img = robot.get_current_img(camera_name="top")
                third_img = robot.get_current_img(camera_name="third")
                wrist_img = robot.get_current_img(camera_name="wrist")
                view_img(top_img)  # 或者同时显示所有三个图像
                state = robot.get_current_state()
                action = client.get_action(top_img, third_img, wrist_img, state)  # 修改：传递三个图像
                start_time = time.time()
                for i in range(ACTION_HORIZON):
                    concat_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    assert concat_action.shape == (6,), concat_action.shape
                    robot.set_target_state(torch.from_numpy(concat_action))
                    time.sleep(0.01)

                    # get the realtime image
                    top_img = robot.get_current_img(camera_name="top")
                    view_img(top_img)

                    # 0.05*16 = 0.8 seconds
                    print("executing action", i, "time taken", time.time() - start_time)
                print("Action chunk execution time taken", time.time() - start_time)
    else:
        # Test Dataset Source https://huggingface.co/datasets/youliangtan/so100_strawberry_grape
        dataset = LeRobotDataset(
            repo_id="youliangtan/so100_strawberry_grape",
            root=args.dataset_path,
        )

        robot = SO100Robot(calibrate=False, enable_camera=True, camera_indices=(args.camera_index_top, args.camera_index_third, args.camera_index_wrist))  # 修改：传入摄像头索引
        with robot.activate():
            actions = []
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Loading actions"):
                action = dataset[i]["action"]
                img = dataset[i]["observation.images.webcam"].data.numpy()
                # original shape (3, 480, 640) for image data
                realtime_img = robot.get_current_img()

                img = img.transpose(1, 2, 0)
                view_img(img, realtime_img)
                actions.append(action)

            # plot the actions
            plt.plot(actions)
            plt.show()

            print("Done initial pose")

            # Use tqdm to create a progress bar
            for action in tqdm(actions, desc="Executing actions"):
                img = robot.get_current_img()
                view_img(img)

                robot.set_target_state(action)
                time.sleep(0.05)

            print("Done all actions")
            robot.go_home()
            print("Done home")