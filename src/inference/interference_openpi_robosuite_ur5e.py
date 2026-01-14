import os
import time
import datetime
import pathlib
import math
import dataclasses
import logging

import imageio
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from openpi_client import image_tools
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from typing import Tuple
import tyro


def setup_logging(args: "Args") -> None:
    level = getattr(logging, str(args.logging).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=os.path.join(args.video_out_dir, 'ur5e_robosuite_openpi.log'),
        force=True
    )


def make_env(args: "Args", controller_config):
    logging.info(f"Creating robosuite environment: {args.env_name} with UR5e robot")
    logging.info(f"Cameras: {args.camera_names}")
    env = suite.make(
        args.env_name,
        robots=["UR5e"],
        gripper_types="default",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(args.camera_names),
        camera_heights=args.cam_h,
        camera_widths=args.cam_w,
        control_freq=20,
        horizon=args.horizon,
        reward_shaping=False,
    )
    logging.info("Environment created successfully")
    return env


def load_policy_from_config(args: "Args"):
    logging.info(f"Loading policy configuration: {args.policy_config}")
    config = _config.get_config(args.policy_config)
    if args.checkpoint_dir is None:
        logging.info("No checkpoint directory specified, using default checkpoint path")
        checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
    else:
        checkpoint_dir = args.checkpoint_dir
    logging.info(f"Loading policy from checkpoint: {checkpoint_dir}")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    logging.info("Policy loaded successfully")
    return policy


def setup_writers(args: "Args", ts: str) -> dict:
    writers: dict[str, imageio.core.Format.Writer] = {}
    if args.save_video:
        logging.info(f"Video recording enabled, saving to: {args.video_out_dir}")
        for cam in args.camera_names:
            out_path = os.path.join(args.video_out_dir, f"ur5e_{args.env_name}_{cam}_{ts}.mp4")
            writers[cam] = imageio.get_writer(out_path, fps=20)
            logging.info(f"Video writer created for {cam}: {out_path}")
    return writers


def get_images(obs: dict, camera_names: tuple[str, ...], resize: int):
    base_cam = camera_names[0] if camera_names else "agentview"
    base_img = obs.get(f"{base_cam}_image")
    if base_img is None:
        base_cam = "frontview"
        base_img = obs[f"{base_cam}_image"]
    wrist_cam = "robot0_eye_in_hand"
    wrist_img = obs.get(f"{wrist_cam}_image")
    if wrist_img is None:
        wrist_img = np.zeros_like(base_img)
    # base_img = np.ascontiguousarray(base_img)
    base_img = np.rot90(base_img, 2)  # Align birdview by rotating 180°
    base_img = np.ascontiguousarray(base_img)
    wrist_img = np.ascontiguousarray(wrist_img[::-1])
    base_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(base_img, resize, resize)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize, resize)
    )
    
    return base_cam, wrist_cam, base_img, wrist_img


def build_state(obs: dict) -> Tuple[np.ndarray, np.ndarray]:
    joint_pos = obs["robot0_joint_pos"]
    eef_quat = obs["robot0_eef_quat"]
    eef_axisangle = quat2axisangle(eef_quat)
    gripper_pos = obs["robot0_gripper_qpos"] # Change from quat to position
    
    
    logging.debug(f"Robot Joint position: [{joint_pos[0]:.4f}, {joint_pos[1]:.4f}, {joint_pos[2]:.4f}, {joint_pos[3]:.4f}, {joint_pos[4]:.4f}, {joint_pos[5]:.4f}]")
    logging.debug(f"Robot EEF quaternion: [{eef_quat[0]:.4f}, {eef_quat[1]:.4f}, {eef_quat[2]:.4f}, {eef_quat[3]:.4f}]")
    logging.debug(f"Robot EEF axis-angle: [{eef_axisangle[0]:.4f}, {eef_axisangle[1]:.4f}, {eef_axisangle[2]:.4f}]")
    logging.debug(f"Robot gripper position: [{gripper_pos[0]:.4f}, {gripper_pos[1]:.4f}]")
    
    # return np.concatenate((joint_pos, eef_axisangle, gripper_qpos))
    return joint_pos, gripper_pos


def run_rollout(env, policy, args: "Args", writers: dict[str, imageio.core.Format.Writer]) -> None:
    logging.info("Starting rollout")
    action_plan: list[np.ndarray] = []
    t = 0
    done = False
    obs = env.reset()
    logging.info("Environment reset complete")
    try:
        while t < args.horizon and not done:
            logging.debug(f"Step {t}/{args.horizon}")
            base_cam, wrist_cam, base_img, wrist_img = get_images(obs, args.camera_names, args.resize_size)
            if writers:
                writers.get(base_cam, None) and writers[base_cam].append_data(base_img)
                writers.get(wrist_cam, None) and writers[wrist_cam].append_data(wrist_img)
            joints, gripper = build_state(obs) # TODO: factor out 
            if not action_plan:
                logging.debug("Action plan empty, querying policy for new actions")
                element = {
                    "observation/joints": joints,
                    "observation/gripper": gripper,
                    "observation/base_rgb": base_img,
                    "observation/wrist_rgb": wrist_img,
                    "observation/prompt": "Grasp the red cube on the table.",
                }
                
                # Infer actions from policy via OpenPI client
                result = policy.infer(element)
                actions = result["actions"]
                logging.debug(f"Policy returned {len(actions)} actions")
                assert len(actions) >= args.replan_steps, (
                    f"Policy predicted {len(actions)} steps, expected at least {args.replan_steps}"
                )
                
                action_plan.extend(actions[: args.replan_steps])
                logging.debug(f"Added {args.replan_steps} actions to plan")
                
                
            # Pop next action
            action = np.asarray(action_plan.pop(0))
            
            # ----- DELTA CHECK -----
            prev_joints = joints.copy()
            obs, reward, done, info = env.step(action.tolist())
            new_joints, new_gripper = build_state(obs)
            
            joint_delta = new_joints - prev_joints
            gripper_delta = new_gripper[0] - gripper[0]
            logging.debug(
                f"Action executed: [{', '.join(f'{a:.4f}' for a in action)}]"
            )
            logging.debug(
                f"Joint change:   [{', '.join(f'{d:.4f}' for d in joint_delta)}], "
                f"Gripper change: {gripper_delta:.4f}"
            )
            # ----------------------
            
            logging.debug(f"Reward: {reward:.4f}, Done: {done}")
            if done:
                logging.info(f"Task completed successfully at step {t}")
            t += 1
    # except Exception as e:
    #     logging.error(f"An error occurred during rollout: {e}")
    finally:
        logging.info(f"Rollout finished after {t} steps")

@dataclasses.dataclass
class Args:
    # OpenPI policy configuration
    policy_config: str = "pi0_ur5"
    checkpoint_dir: str | None = None  # If None, downloads default checkpoint
    # Set the size to which camera images are resized before being passed to the policy
    resize_size: int = 256
    replan_steps: int = 35

    # Robosuite task
    env_name: str = "Lift"
    horizon: int = 300

    # Cameras
    camera_names: tuple[str, ...] = ("birdview", "robot0_eye_in_hand")
    cam_w: int = 256
    cam_h: int = 256

    # Output video
    save_video: bool = True
    video_out_dir: str = "videos/openpi_ur5e"

    seed: int = 7
    
    logging: str = "DEBUG"  # DEBUG, INFO


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle vector."""
    q = np.array(quat, dtype=np.float64).copy()
    # clip w
    q[3] = max(min(q[3], 1.0), -1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def main(args: Args) -> None:
    pathlib.Path(args.video_out_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(args)
    
    logging.info("Starting UR5e robosuite + OpenPI integration")
    logging.info(f"Environment: {args.env_name}")
    logging.info(f"Policy config: {args.policy_config}")
    logging.info(f"Checkpoint dir: {args.checkpoint_dir or 'default (auto-download)'}")
    logging.info(f"Horizon: {args.horizon} steps")
    logging.info(f"Replan every: {args.replan_steps} steps")
    
    np.random.seed(args.seed)
    
    logging.info("Loading BASIC composite controller configuration")
    controller_config = load_composite_controller_config(controller="OSC_POSITION")    
    
    env = make_env(args, controller_config)
    policy = load_policy_from_config(args)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writers = setup_writers(args, ts)
    
    try:
        run_rollout(env, policy, args, writers)
    finally:
        if writers:
            logging.info("Closing video writers")
            for cam, w in writers.items():
                w.close()
                logging.info(f"Video saved for {cam}")
        env.close()
        logging.info("Environment closed")


if __name__ == "__main__":
    tyro.cli(main)
