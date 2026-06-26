"""Visualize a random timestep or render a full episode as video."""

import argparse
import io
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import zarr
from PIL import Image


ZARR_PATH = "/home/madhavan/lerobot/data/ppp/dataset.zarr_downsampled.zarr"

CAMERAS = [
    "observation.images.cam_azure_kinect_left.color",
    "observation.images.cam_azure_kinect_front.color",
    "observation.images.cam_wrist",
]

CAMERA_LABELS = ["Left (Azure Kinect)", "Front (Azure Kinect)", "Wrist"]

# Keys written by EvalRecorder for pre-normalisation crops
DEBUG_PREFIX = "debug."


def decode_image(bytes_arr, length):
    return np.array(Image.open(io.BytesIO(bytes(bytes_arr[:length]))))


def _debug_camera_keys(z):
    """Return sorted list of debug.* image keys present in the zarr."""
    return sorted(k for k in z.keys() if k.startswith(DEBUG_PREFIX) and "bytes" in z[k])


def render_frame(z, idx, axes, sa_ylim, eef_ylim, debug_axes=None):
    """Render one timestep into pre-created axes."""
    state = z["observation.state"][idx]
    action = z["action"][idx]
    eef_obs = z["observation.right_eef_pose"][idx]
    eef_act = z["action.right_eef_pose"][idx]
    x = np.arange(len(state))
    x_eef = np.arange(len(eef_obs))
    w = 0.35

    ax_imgs, ax_sa, ax_eef, ax_txt = axes

    # raw cameras
    for ax, cam_key in zip(ax_imgs, CAMERAS):
        ax.cla()
        length = int(z[cam_key]["lengths"][idx])
        img = decode_image(z[cam_key]["bytes"][idx], length)
        ax.imshow(img)
        ax.axis("off")

    # debug crops (what the model actually saw)
    if debug_axes:
        for ax, dk in zip(debug_axes, _debug_camera_keys(z)):
            ax.cla()
            length = int(z[dk]["lengths"][idx])
            img = decode_image(z[dk]["bytes"][idx], length)
            ax.imshow(img)
            ax.axis("off")

    # state vs action
    ax_sa.cla()
    ax_sa.bar(x - w / 2, state, w, label="state", color="steelblue")
    ax_sa.bar(x + w / 2, action, w, label="action", color="coral")
    ax_sa.set_ylim(*sa_ylim)
    ax_sa.set_xticks(x)
    ax_sa.set_xlabel("Joint")
    ax_sa.set_ylabel("Value")
    ax_sa.set_title("State vs Action (joints)")
    ax_sa.legend(fontsize=8)

    # eef pose
    ax_eef.cla()
    ax_eef.bar(x_eef - w / 2, eef_obs, w, label="obs eef", color="steelblue")
    ax_eef.bar(x_eef + w / 2, eef_act, w, label="action eef", color="coral")
    ax_eef.set_ylim(*eef_ylim)
    ax_eef.set_xticks(x_eef)
    ax_eef.set_xlabel("Dim")
    ax_eef.set_ylabel("Value")
    ax_eef.set_title("Right EEF Pose")
    ax_eef.legend(fontsize=8)

    # text
    ax_txt.cla()
    ax_txt.axis("off")
    lines = [
        f"episode_index : {z['episode_index'][idx]}",
        f"frame_index   : {z['frame_index'][idx]}",
        f"task_index    : {z['task_index'][idx]}",
        f"timestamp     : {z['timestamp'][idx]:.4f} s",
        "",
        "state:",
        "  " + str(np.round(state, 3)),
        "",
        "action:",
        "  " + str(np.round(action, 3)),
    ]
    ax_txt.text(0.02, 0.95, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=8, va="top", family="monospace")


def _build_figure(z, has_debug):
    """Create figure + axes layout. Returns (fig, ax_imgs, ax_sa, ax_eef, ax_txt, debug_axes)."""
    n_rows = 3 if has_debug else 2
    fig = plt.figure(figsize=(16, 5 * n_rows))

    ax_imgs = [fig.add_subplot(n_rows, 3, i + 1) for i in range(3)]
    ax_sa  = fig.add_subplot(n_rows, 3, n_rows * 3 - 2)
    ax_eef = fig.add_subplot(n_rows, 3, n_rows * 3 - 1)
    ax_txt = fig.add_subplot(n_rows, 3, n_rows * 3)

    debug_axes = None
    if has_debug:
        debug_keys = _debug_camera_keys(z)
        debug_axes = [fig.add_subplot(n_rows, 3, 3 + i + 1) for i in range(len(debug_keys))]
        for ax, dk in zip(debug_axes, debug_keys):
            label = dk.replace(DEBUG_PREFIX, "").split(".")[-2] if "." in dk else dk
            ax.set_title(f"[model input] {label}", fontsize=9, color="darkred")

    for ax, label in zip(ax_imgs, CAMERA_LABELS):
        ax.set_title(label, fontsize=10)

    return fig, ax_imgs, ax_sa, ax_eef, ax_txt, debug_axes


def show_single(z, idx):
    n = z["index"].shape[0]
    print(f"Timestep index: {idx} / {n - 1}")
    print(f"  episode_index : {z['episode_index'][idx]}")
    print(f"  frame_index   : {z['frame_index'][idx]}")
    print(f"  timestamp     : {z['timestamp'][idx]:.3f}s")
    print(f"  state         : {z['observation.state'][idx]}")
    print(f"  action        : {z['action'][idx]}")

    has_debug = bool(_debug_camera_keys(z))
    fig, ax_imgs, ax_sa, ax_eef, ax_txt, debug_axes = _build_figure(z, has_debug)
    fig.suptitle(
        f"Timestep {idx}  |  Episode {z['episode_index'][idx]}  |"
        f"  Frame {z['frame_index'][idx]}  |  t={z['timestamp'][idx]:.3f}s",
        fontsize=13,
    )
    sa_ylim = (-1.5, 1.5)
    eef_ylim = (-1.5, 1.5)
    render_frame(z, idx, (ax_imgs, ax_sa, ax_eef, ax_txt), sa_ylim, eef_ylim, debug_axes)
    plt.tight_layout()
    plt.show()


def make_episode_video(z, episode_id, out_path, fps=10):
    import cv2
    import matplotlib
    matplotlib.use("Agg")

    ep_idx = z["episode_index"][:]
    indices = np.where(ep_idx == episode_id)[0]
    if len(indices) == 0:
        print(f"Episode {episode_id} not found.")
        sys.exit(1)

    print(f"Episode {episode_id}: {len(indices)} frames → {out_path}")

    # pre-compute y-limits from episode data for stable axes
    states  = z["observation.state"][indices[0]:indices[-1] + 1]
    actions = z["action"][indices[0]:indices[-1] + 1]
    eef_obs = z["observation.right_eef_pose"][indices[0]:indices[-1] + 1]
    eef_act = z["action.right_eef_pose"][indices[0]:indices[-1] + 1]
    sa_all  = np.concatenate([states, actions])
    sa_ylim  = (float(sa_all.min()) - 0.1, float(sa_all.max()) + 0.1)
    eef_all  = np.concatenate([eef_obs, eef_act])
    eef_ylim = (float(eef_all.min()) - 0.1, float(eef_all.max()) + 0.1)

    has_debug = bool(_debug_camera_keys(z))
    fig, ax_imgs, ax_sa, ax_eef, ax_txt, debug_axes = _build_figure(z, has_debug)
    fig.set_dpi(100)

    # probe frame size
    render_frame(z, indices[0], (ax_imgs, ax_sa, ax_eef, ax_txt), sa_ylim, eef_ylim, debug_axes)
    fig.suptitle("", fontsize=13)
    plt.tight_layout()
    fig.canvas.draw()
    frame_w, frame_h = fig.canvas.get_width_height()
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    for i, idx in enumerate(indices):
        fig.suptitle(
            f"Episode {episode_id}  |  Frame {int(z['frame_index'][idx])} / {len(indices) - 1}"
            f"  |  t={z['timestamp'][idx]:.3f}s",
            fontsize=13,
        )
        render_frame(z, idx, (ax_imgs, ax_sa, ax_eef, ax_txt), sa_ylim, eef_ylim, debug_axes)
        plt.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(frame_h, frame_w, 4)
        out.write(cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))
        if (i + 1) % 50 == 0 or i == len(indices) - 1:
            print(f"  {i + 1}/{len(indices)} frames encoded", end="\r")

    out.release()
    plt.close(fig)
    print(f"\nDone: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default=ZARR_PATH)
    parser.add_argument("--idx", type=int, default=None,
                        help="Single timestep index (random if omitted, ignored when --episode is set)")
    parser.add_argument("--episode", type=int, default=None, help="Episode ID to render as video")
    parser.add_argument("--out", type=str, default=None, help="Output video path (default: episode_<N>.mp4)")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS (default: 10)")
    args = parser.parse_args()

    z = zarr.open(args.zarr)

    if args.episode is not None:
        out = args.out or f"episode_{args.episode}.mp4"
        make_episode_video(z, args.episode, out, fps=args.fps)
    else:
        n = z["index"].shape[0]
        idx = args.idx if args.idx is not None else random.randint(0, n - 1)
        show_single(z, idx)


if __name__ == "__main__":
    main()
