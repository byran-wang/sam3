import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_points(point_coords, point_labels):
    if len(point_coords) % 2 != 0:
        raise ValueError("point_coords must be x y pairs")
    points = np.array(
        [[point_coords[i], point_coords[i + 1]] for i in range(0, len(point_coords), 2)],
        dtype=np.float32,
    )
    labels = np.array(point_labels, dtype=np.int32)
    if len(labels) != len(points):
        raise ValueError("point_labels must match number of points")
    return points, labels


def resolve_output_path(out_path, image_path):
    out_path = Path(out_path)
    if out_path.is_dir() or str(out_path).endswith(os.sep):
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path / f"{Path(image_path).stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def collect_points_with_labels(image, title=None):
    points_abs = []
    labels = []
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(
        title
        or "Left click: positive (1), right click: negative (0). Press Enter to finish."
    )
    ax.axis("off")

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            label = 1
            color = "lime"
        elif event.button == 3:
            label = 0
            color = "red"
        else:
            return
        points_abs.append([event.xdata, event.ydata])
        labels.append(label)
        ax.scatter([event.xdata], [event.ydata], c=color, s=30, marker="o")
        fig.canvas.draw_idle()

    def _on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()
    return points_abs, labels


def collect_points_with_labels_with_preview(image, predict_mask_fn, title=None):
    points_abs = []
    labels = []
    fig, ax = plt.subplots()
    ax.imshow(image)
    mask_artist = ax.imshow(np.zeros(image.shape[:2]), cmap="jet", alpha=0.0)
    contour_artist = None
    point_artists = []
    ax.set_title(
        title
        or (
            "Left click: positive (1), right click: negative (0), "
            "middle click: reset. Press Enter to finish."
        )
    )
    ax.axis("off")

    def _refresh_mask():
        nonlocal contour_artist
        if not points_abs:
            mask_artist.set_data(np.zeros(image.shape[:2]))
            if contour_artist is not None:
                if hasattr(contour_artist, "collections"):
                    for c in contour_artist.collections:
                        c.remove()
                else:
                    contour_artist.remove()
                contour_artist = None
            fig.canvas.draw_idle()
            return
        mask = predict_mask_fn(points_abs, labels)
        if mask is None:
            return
        mask_artist.set_data(mask)
        if contour_artist is not None:
            if hasattr(contour_artist, "collections"):
                for c in contour_artist.collections:
                    c.remove()
            else:
                contour_artist.remove()
        contour_artist = ax.contour(
            mask.astype(np.float32),
            levels=[0.5],
            colors="cyan",
            linewidths=1.5,
        )
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 2:
            points_abs.clear()
            labels.clear()
            for artist in point_artists:
                artist.remove()
            point_artists.clear()
            _refresh_mask()
            return
        if event.button == 1:
            label = 1
            color = "lime"
        elif event.button == 3:
            label = 0
            color = "red"
        else:
            return
        points_abs.append([event.xdata, event.ydata])
        labels.append(label)
        artist = ax.scatter([event.xdata], [event.ydata], c=color, s=30, marker="o")
        point_artists.append(artist)
        _refresh_mask()

    def _on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()
    return points_abs, labels


def main(args):
    image_path = args.image_path
    out_path = resolve_output_path(args.out_path, image_path)

    image = Image.open(image_path).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_root = os.path.join(os.path.dirname(__file__))
    checkpoint_path = os.path.join(sam3_root, "sam3", "model", "checkpoints", "sam3.pt")

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model, device=device)
    inference_state = processor.set_image(image)

    if args.point_coords and args.point_labels:
        point_coords, point_labels = parse_points(args.point_coords, args.point_labels)
    else:
        def _predict_mask(points_abs, labels):
            point_coords = np.array(points_abs, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)
            masks, scores, _ = model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=args.multimask_output,
            )
            if masks is None or len(masks) == 0:
                return None
            best_idx = int(np.argmax(scores))
            return masks[best_idx].astype(np.uint8)

        points_abs, labels = collect_points_with_labels_with_preview(
            np.array(image),
            _predict_mask,
        )
        if not points_abs:
            raise RuntimeError("No points selected.")
        point_coords = np.array(points_abs, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)
    masks, scores, _ = model.predict_inst(
        inference_state,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=args.multimask_output,
    )
    if masks is None or len(masks) == 0:
        raise RuntimeError("No masks returned from SAM3.")

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)
    print(f"Saved mask to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--multimask_output", type=int, default=1)
    parser.add_argument("--point_coords", type=float, nargs="*", default=None)
    parser.add_argument("--point_labels", type=int, nargs="*", default=None)
    args = parser.parse_args()
    main(args)
