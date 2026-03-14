import logging
import os
import pickle
from typing import Dict

import imageio

from .config import Config


class RolloutRecorder:
    def __init__(self, cfg: Config, save_dir: str, trial_idx: int):
        self.cfg = cfg
        self.save_dir = save_dir
        self.trial_idx = trial_idx
        self.writers: Dict[str, imageio.core.Format.Writer] = {}
        self.temp_paths: Dict[str, str] = {}
        self.active = cfg.save_video

        if self.active:
            self._setup_writers()

    def _setup_writers(self):
        """Initialize video writers for configured cameras."""
        os.makedirs(self.save_dir, exist_ok=True)
        for cam in self.cfg.camera_names:
            # Create temporary path
            out_path = os.path.join(self.save_dir, f"temp_trial{self.trial_idx}_{cam}.mp4")
            self.temp_paths[cam] = out_path
            self.writers[cam] = imageio.get_writer(out_path, fps=20)

    def record_frame(self, cam_name: str, frame):
        if self.active and cam_name in self.writers:
            self.writers[cam_name].append_data(frame)

    def close(self, success: bool, metadata: Dict):
        """Closes writers, renames files based on success, saves metadata."""
        if not self.active:
            return

        # Close writers
        for w in self.writers.values():
            w.close()

        # Rename files and save metadata
        final_paths = {}
        for cam, temp_path in self.temp_paths.items():
            if os.path.exists(temp_path):
                base_name = f"ur5e_{self.cfg.task.env_name}_{cam}"
                new_name = f"{base_name}--trial{self.trial_idx}--succ{int(success)}.mp4"
                new_path = os.path.join(self.save_dir, new_name)
                os.rename(temp_path, new_path)
                final_paths[cam] = new_path
                logging.info(f"Saved video: {new_path}")

        # Save metadata pickle
        self._save_metadata(metadata, success, final_paths)

    def _save_metadata(self, run_info: Dict, success: bool, video_paths: Dict):
        meta_path = os.path.join(
            self.save_dir, f"ur5e_{self.cfg.task.env_name}--trial{self.trial_idx}--succ{int(success)}.pkl"
        )

        # Create save dict with the specified structure
        save_dict = {
            "task_suite_name": getattr(self.cfg, "save_name", "pi0-ur5"),
            "task_id": self.cfg.task.env_name,
            "task_description": self.cfg.task.prompt,
            "episode_idx": self.trial_idx,
            "episode_success": success,
            "mp4_path": video_paths.get(self.cfg.camera_names[0], "") if video_paths else "",
            "wrist_mp4_path": video_paths.get("robot0_eye_in_hand", "") if len(video_paths) > 1 else "",
            "model_infer_times": run_info.get("model_infer_times", []),
            "replan_steps": getattr(self.cfg, "replan_steps", None),
            "num_steps_wait": getattr(self.cfg, "num_steps_wait", None),
            "end_step": run_info.get("steps", 0),
            "step_done": run_info.get("step_done", run_info.get("success_step", [])),
        }

        with open(meta_path, "wb") as f:
            pickle.dump(save_dict, f)
