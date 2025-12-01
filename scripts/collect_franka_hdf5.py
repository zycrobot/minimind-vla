"""
使用 PyBullet 构建 Franka Panda 抓取仿真并录制 HDF5 数据。

录制内容：
    - 文本指令（UTF-8 字符串）
    - 每个时间步的 RGB 图像（uint8，H×W×3）
    - 机械臂关节动作（7 维关节角 + 1 维夹爪开合量）
"""
import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pybullet as pb
import pybullet_data


@dataclass
class Episode:
    instruction: str
    images: List[np.ndarray]
    actions: List[np.ndarray]
    robot_states: List[np.ndarray]
    meta: Dict


class FrankaPyBulletEnv:
    def __init__(
        self,
        gui: bool = False,
        workspace_center: Tuple[float, float, float] = (0.6, 0.0, 0.0),
    ):
        self.gui = gui
        self.client = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.resetSimulation(physicsClientId=self.client)
        pb.setTimeStep(1 / 60.0, physicsClientId=self.client)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.robot_id = None
        self.object_ids: Dict[str, int] = {}
        self.workspace_center = np.array(workspace_center)
        self.ee_link_index = 11  # Panda hand
        # 使用单个夹爪关节索引，只使用第一个夹爪关节来控制
        self.gripper_joint_index = 9
        self.arm_joint_indices = list(range(7))
        self.grasp_constraint = None

        self._setup_scene()
        self.home_joint_positions = [0.0, -0.4, 0.0, -2.5, 0.0, 2.2, 0.8]
        self.reset_robot()

    def _setup_scene(self):
        pb.loadURDF("plane.urdf")
        table_pos = [self.workspace_center[0], self.workspace_center[1], -0.625]
        pb.loadURDF("table/table.urdf", table_pos, useFixedBase=True)

        start_pos = [0, 0, 0]
        start_orn = pb.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = pb.loadURDF(
            "franka_panda/panda.urdf",
            start_pos,
            start_orn,
            useFixedBase=True,
            flags=pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )
        self.open_gripper()

    def reset_robot(self):
        """
        将机械臂和夹爪重置到预定义的家位置，并清理动量。
        """
        self.release_grasp()
        for joint_index, joint_value in zip(self.arm_joint_indices, self.home_joint_positions):
            pb.resetJointState(self.robot_id, joint_index, joint_value)
        # 重置夹爪，设置为打开状态
        pb.resetJointState(self.robot_id, self.gripper_joint_index, 0.04)
        # 为了兼容性，同时设置另一个夹爪关节
        pb.resetJointState(self.robot_id, 10, 0.04)
        self.open_gripper()
        pb.stepSimulation()

    def reset_objects(self):
        self.release_grasp()
        for bid in self.object_ids.values():
            pb.removeBody(bid)
        self.object_ids.clear()

    def spawn_objects(self, num_objects: int = 4):
        shapes = ["cube", "cylinder", "sphere"]
        colors = [
            ("红色", [0.9, 0.1, 0.1, 1]),
            ("蓝色", [0.1, 0.1, 0.9, 1]),
            ("绿色", [0.1, 0.75, 0.1, 1]),
            ("黄色", [0.9, 0.8, 0.1, 1]),
        ]
        random.shuffle(colors)
        for idx in range(num_objects):
            shape = random.choice(shapes)
            color_name, rgba = colors[idx % len(colors)]
            name = f"{color_name}{shape}"
            x = self.workspace_center[0] + random.uniform(-0.15, 0.15)
            y = self.workspace_center[1] + random.uniform(-0.15, 0.15)
            z = 0.0
            if shape == "cube":
                visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.02] * 3, rgbaColor=rgba)
                collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.02] * 3)
            elif shape == "cylinder":
                visual = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.02, length=0.04, rgbaColor=rgba)
                collision = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=0.02, height=0.04)
            else:
                visual = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.02, rgbaColor=rgba)
                collision = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.02)

            body_id = pb.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, 0.02],
            )
            self.object_ids[name] = body_id

    def open_gripper(self):
        # 使用单个值控制夹爪打开
        pb.setJointMotorControl2(
            self.robot_id,
            self.gripper_joint_index,
            pb.POSITION_CONTROL,
            targetPosition=0.04,
            force=100,
        )
        # 为了确保两个夹爪都打开，同时控制第二个夹爪关节
        pb.setJointMotorControl2(
            self.robot_id,
            10,
            pb.POSITION_CONTROL,
            targetPosition=0.04,
            force=100,
        )

    def close_gripper(self):
        # 使用单个值控制夹爪关闭
        pb.setJointMotorControl2(
            self.robot_id,
            self.gripper_joint_index,
            pb.POSITION_CONTROL,
            targetPosition=0.0,
            force=100,
        )
        # 为了确保两个夹爪都关闭，同时控制第二个夹爪关节
        pb.setJointMotorControl2(
            self.robot_id,
            10,
            pb.POSITION_CONTROL,
            targetPosition=0.0,
            force=100,
        )

    def release_grasp(self):
        if self.grasp_constraint is not None:
            pb.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None

    def attach_object(self, object_id: int):
        if self.grasp_constraint is not None:
            pb.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
        link_state = pb.getLinkState(self.robot_id, self.ee_link_index)
        link_pos, link_orn = link_state[0], link_state[1]
        obj_pos, obj_orn = pb.getBasePositionAndOrientation(object_id)
        parent_inv = pb.invertTransform(link_pos, link_orn)
        local_pos, local_orn = pb.multiplyTransforms(parent_inv[0], parent_inv[1], obj_pos, obj_orn)
        self.grasp_constraint = pb.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=local_pos,
            parentFrameOrientation=local_orn,
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0, 1],
        )

    def move_to(self, target_pos, target_orn, steps: int = 60):
        joint_positions = pb.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_pos,
            target_orn,
            maxNumIterations=200,
        )
        for step in range(steps):
            for joint_i, joint_index in enumerate(self.arm_joint_indices):
                pb.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    pb.POSITION_CONTROL,
                    targetPosition=joint_positions[joint_i],
                    force=200,
                    positionGain=0.03,
                    velocityGain=1.0,
                )
            pb.stepSimulation()

    def apply_joint_positions(self, joint_positions: np.ndarray, steps: int = 30, position_gain: float = 0.03):
        """
        将机械臂关节驱动到指定角度序列的一步。
        
        Args:
            joint_positions: 关节位置目标值
            steps: 执行步数
            position_gain: 位置控制增益，默认为0.03（比原来的0.05更小，可减少抖动）
        """
        for joint_i, joint_index in enumerate(self.arm_joint_indices):
            if joint_i >= len(joint_positions):
                break
            pb.setJointMotorControl2(
                self.robot_id,
                joint_index,
                pb.POSITION_CONTROL,
                targetPosition=float(joint_positions[joint_i]),
                force=200,
                positionGain=position_gain,
                velocityGain=1.0,
            )
        for _ in range(steps):
            pb.stepSimulation()

    def render(self, width: int = 640, height: int = 480) -> np.ndarray:
        cam_target = self.workspace_center + np.array([0, 0, 0.05])
        cam_distance = 0.7
        yaw, pitch, roll = (90, -35, 0)
        view = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target,
            distance=cam_distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2,
        )
        proj = pb.computeProjectionMatrixFOV(
            fov=60, aspect=width / height, nearVal=0.01, farVal=3.0
        )
        _, _, rgb, _, _ = pb.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=pb.ER_TINY_RENDERER,
        )
        rgb = np.reshape(rgb, (height, width, 4))[:, :, :3]
        return rgb.astype(np.uint8)

    def record_joint_state(self) -> np.ndarray:
        # 获取机械臂关节状态
        arm_states = pb.getJointStates(self.robot_id, self.arm_joint_indices)
        arm_q = np.array([state[0] for state in arm_states], dtype=np.float32)
        
        # 获取夹爪关节状态（只取一个值）
        gripper_state = pb.getJointState(self.robot_id, self.gripper_joint_index)
        gripper_q = np.array([gripper_state[0]], dtype=np.float32)
        
        # 组合为8维数组：7个机械臂关节 + 1个夹爪值
        q = np.concatenate([arm_q, gripper_q])
        return q
    
    def get_robot_state(self) -> np.ndarray:
        """
        获取机器人状态信息
        返回：[7个机械臂关节角度, 1个夹爪关节角度]
        与动作维度保持一致，都是8维数据
        """
        # 获取机械臂关节状态
        arm_states = pb.getJointStates(self.robot_id, self.arm_joint_indices)
        arm_q = np.array([state[0] for state in arm_states], dtype=np.float32)
        
        # 获取夹爪关节状态（只取一个值）
        gripper_state = pb.getJointState(self.robot_id, self.gripper_joint_index)
        gripper_q = np.array([gripper_state[0]], dtype=np.float32)
        
        # 组合为8维数组：7个机械臂关节 + 1个夹爪值
        robot_state = np.concatenate([arm_q, gripper_q])
        return robot_state

    def scripted_pick(self, object_name: str) -> Episode:
        object_id = self.object_ids[object_name]
        obj_pos, _ = pb.getBasePositionAndOrientation(object_id)
        hover_height = 0.30
        grasp_height = obj_pos[2] + 0.02
        waypoints = [
            (np.array([obj_pos[0], obj_pos[1], hover_height]), pb.getQuaternionFromEuler([0, np.pi, 0])),
            (np.array([obj_pos[0], obj_pos[1], grasp_height]), pb.getQuaternionFromEuler([0, np.pi, 0])),
        ]

        images: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        robot_states: List[np.ndarray] = []

        self.open_gripper()
        for pos, orn in waypoints:
            self.move_to(pos, orn)
            images.append(self.render())
            actions.append(self.record_joint_state())
            robot_states.append(self.get_robot_state())

        # 缓慢逼近确保夹爪接触
        fine_steps = 30
        approach_pos = np.array([obj_pos[0], obj_pos[1], grasp_height])
        for i in range(fine_steps):
            offset = np.array([0, 0, -0.002 * i])
            self.move_to(approach_pos + offset, pb.getQuaternionFromEuler([0, np.pi, 0]), steps=1)
        self.close_gripper()
        for _ in range(30):
            pb.stepSimulation()
        self.attach_object(object_id)
        images.append(self.render())
        actions.append(self.record_joint_state())
        robot_states.append(self.get_robot_state())

        lift_pos = np.array([obj_pos[0], obj_pos[1], 0.30])
        self.move_to(lift_pos, pb.getQuaternionFromEuler([0, np.pi, 0]))
        images.append(self.render())
        actions.append(self.record_joint_state())
        robot_states.append(self.get_robot_state())

        instruction = f"请抓取{object_name}"
        return Episode(
            instruction=instruction,
            images=images,
            actions=actions,
            robot_states=robot_states,
            meta={"target": object_name},
        )


def save_episodes_to_hdf5(episodes: List[Episode], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    utf8_str = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("data")
        for idx, ep in enumerate(episodes):
            ep_grp = grp.create_group(f"episode_{idx:05d}")
            ep_grp.create_dataset("rgb", data=np.stack(ep.images, axis=0), compression="gzip")
            ep_grp.create_dataset("action", data=np.stack(ep.actions, axis=0), compression="gzip")
            ep_grp.create_dataset("robot_state", data=np.stack(ep.robot_states, axis=0), compression="gzip")
            ep_grp.create_dataset("text", data=ep.instruction, dtype=utf8_str)
            for key, value in ep.meta.items():
                ep_grp.attrs[key] = value

        # 兼容旧 dataloader：保存单帧与动作
        final_images = np.stack([ep.images[-1] for ep in episodes], axis=0)
        final_actions = np.stack([ep.actions[-1] for ep in episodes], axis=0)
        final_robot_states = np.stack([ep.robot_states[-1] for ep in episodes], axis=0)
        texts = [ep.instruction for ep in episodes]
        f.create_dataset("image", data=final_images, compression="gzip")
        f.create_dataset("action", data=final_actions, compression="gzip")
        f.create_dataset("robot_state", data=final_robot_states, compression="gzip")
        f.create_dataset("text", data=texts, dtype=utf8_str)

    print(f"保存 {len(episodes)} 条演示到 {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PyBullet Franka 抓取数据采集")
    parser.add_argument("--episodes", type=int, default=10, help="录制的任务数量")
    parser.add_argument("--output", type=str, default="dataset/franka_pick_dataset.hdf5")
    parser.add_argument("--gui", action="store_true", help="是否启用 PyBullet GUI")
    args = parser.parse_args()

    env = FrankaPyBulletEnv(gui=args.gui)
    episodes: List[Episode] = []

    for i in range(args.episodes):
        env.reset_robot()
        env.reset_objects()
        env.spawn_objects()
        target_name = random.choice(list(env.object_ids.keys()))
        episode = env.scripted_pick(target_name)
        episodes.append(episode)
        print(f"[{i+1}/{args.episodes}] 完成 {target_name} 的抓取演示")

    save_episodes_to_hdf5(episodes, args.output)
    pb.disconnect(env.client)


if __name__ == "__main__":
    main()

