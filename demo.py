import dotenv

dotenv.load_dotenv()

import logging
import os
import signal
import sys
from copy import deepcopy
from pathlib import Path

# Suppress verbose httpx debug logging
logging.getLogger("httpx").setLevel(logging.WARNING)

import cv2
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import observations_to_image
from hydra import main as hydra_main
from omegaconf import DictConfig
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from glmap import GLMap, save_gl_map, update_gl_map

IS_PRESET = "--preset" in sys.argv
if IS_PRESET:
    sys.argv.remove("--preset")

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"

ACTION_NAMES = {
    HabitatSimActions.turn_left: "LEFT",
    HabitatSimActions.turn_right: "RIGHT",
    HabitatSimActions.move_forward: "FORWARD",
    HabitatSimActions.stop: "STOP",
}
AUTO_ACTIONS = [HabitatSimActions.turn_left] * 12 + [HabitatSimActions.stop]


def signal_handler(sig, frame):
    print("Ctrl+C detected! Shutting down...")
    os._exit(0)


def _find_episode(env, target_scene: str, target_episode: str):
    """Advance env iterator to target scene and episode."""
    scene_name = Path(env.current_episode.scene_id).name.split(".")[0]
    episode_id = env.current_episode.episode_id
    while True:
        if scene_name == target_scene and episode_id == target_episode:
            return scene_name, episode_id
        try:
            env.current_episode = next(env.episode_iterator)
            scene_name = Path(env.current_episode.scene_id).name.split(".")[0]
            episode_id = env.current_episode.episode_id
        except StopIteration:
            print(f"Target episode {target_scene}-{target_episode} not found!")
            env.close()
            sys.exit(1)


def _keyboard_actions():
    """Yield actions from keyboard input until STOP."""
    print("\nManual controls:")
    print(f"  {FORWARD_KEY} - Move forward  |  {LEFT_KEY} - Turn left")
    print(f"  {RIGHT_KEY} - Turn right  |  {FINISH} - Stop")
    print("Note: Focus the 'Observations' window before pressing keys.\n")

    key_map = {
        ord(FORWARD_KEY): HabitatSimActions.move_forward,
        ord(LEFT_KEY): HabitatSimActions.turn_left,
        ord(RIGHT_KEY): HabitatSimActions.turn_right,
        ord(FINISH): HabitatSimActions.stop,
    }
    while True:
        keystroke = cv2.waitKey(0)
        action = key_map.get(keystroke)
        if action is not None:
            yield action


def _export_and_render(glmap: GLMap, save_dir: Path):
    """Save map, export PLYs, render all instances and groups."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_gl_map(glmap, str(save_dir / "glmap.pkl"))

    ply_dir = save_dir / "gaussians"
    ply_dir.mkdir(parents=True, exist_ok=True)
    glmap.export_gaussians(str(ply_dir))

    render_dir = save_dir / "rendered"
    render_dir.mkdir(parents=True, exist_ok=True)

    renderable_ids = [
        iid
        for iid, inst in glmap.instances.items()
        if inst.gaussian_data is not None and inst.gaussian_data.num_gaussians > 0
    ]

    for iid in renderable_ids:
        inst = glmap.instances[iid]
        img = glmap.render_instance(iid)
        out_path = render_dir / f"instance_{iid}_{inst.category}.png"
        Image.fromarray(img).save(str(out_path))
        print(f"Rendered instance {iid} ({inst.category}) -> {out_path}")

    if len(renderable_ids) > 1:
        img = glmap.render_group(renderable_ids)
        out_path = render_dir / "group_all.png"
        Image.fromarray(img).save(str(out_path))
        print(f"Rendered group ({len(renderable_ids)} instances) -> {out_path}")


@hydra_main(config_path="config", config_name="habitat_eval_hm3dv1", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = patch_config(cfg)

    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(cfg)
    scene_name, episode_id = _find_episode(env, "Dd4bFSTQ8gi", "0")
    print(f"Found target episode: scene={scene_name}, episode={episode_id}")

    observations = env.reset()
    env._episode_from_iter_on_reset = False
    glmap: GLMap = GLMap()
    save_dir = Path(
        f"outputs/{scene_name}_{episode_id}{'_preset' if IS_PRESET else ''}"
    )

    # Select action source
    if IS_PRESET:
        action_iter = iter(AUTO_ACTIONS)
        print(f"Preset mode: {len(AUTO_ACTIONS)} actions")
    else:
        action_iter = _keyboard_actions()

    # Show initial frame (GUI only)
    if not IS_PRESET:
        obs_display = deepcopy(observations)
        obs_display["rgb"] = obs_display["rgb"][:, :, [2, 1, 0]]
        frame = observations_to_image(obs_display, env.get_metrics())
        cv2.namedWindow("Observations", cv2.WINDOW_NORMAL)
        cv2.imshow("Observations", frame)

    # Main loop — identical for both modes
    for step, action in enumerate(action_iter):
        print(f"\nStep {step}: {ACTION_NAMES.get(action, action)}")
        observations = env.step(action)
        if action == HabitatSimActions.stop:
            break

        output_dir = save_dir / str(step)
        output_dir.mkdir(parents=True, exist_ok=True)
        update_gl_map(
            gl_map=glmap,
            habitat_observations=observations,
            main_config=cfg,
            output_dir=output_dir,
        )

        if not IS_PRESET:
            obs_display = deepcopy(observations)
            obs_display["rgb"] = obs_display["rgb"][:, :, [2, 1, 0]]
            frame = observations_to_image(obs_display, env.get_metrics())
            cv2.imshow("Observations", frame)

    env.close()
    _export_and_render(glmap, save_dir)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    try:
        main()
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        os._exit(1)
