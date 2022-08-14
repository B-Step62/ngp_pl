import argparse
import imageio
import os
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

from datasets import dataset_dict
from datasets.ray_utils import get_rays
from models.networks import NGP
from models.rendering import render
from train import depth2img
from utils import load_ckpt


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--scene", required=True)
  parser.add_argument("--load_model", required=True)
  parser.add_argument("--dataset_name")
  parser.add_argument("--plot_path", action="store_true")

  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()

  dataset_name = args.dataset_name
  scene = args.scene
  dataset_path = f"/content/datasets/{dataset_name}/Synthetic_NeRF/{scene}"
  dataset = dataset_dict[dataset_name](dataset_path, split="test", downsample=1.0)
  w, h = dataset.img_wh

  model = NGP(scale=0.5).cuda()
  load_ckpt(model, args.load_model)

  result_dir = f"results/{dataset_name}/{scene}/"
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  frame_dir = os.path.join(result_dir, "frames")
  if not os.path.exists(frame_dir):
    os.mkdir(frame_dir)

  imgs = []
  depths = []
  times = []
  camera_origins = []
  for img_idx in tqdm(range(len(dataset)), desc="Rendering frames"):
    t = time.time()
    rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]["pose"].cuda())
    
    if args.plot_path:
      camera_origins.append(rays_o[0].cpu().numpy()) # camera origin is same for all points

    results = render(model, rays_o, rays_d,
                     test_time=True, T_threshold=1e-2, exp_step_factor=1/256)
    torch.cuda.synchronize()
    times.append(time.time() - t)

    pred = results["rgb"].reshape(h, w, 3).cpu().numpy()
    pred = (pred*255).astype(np.uint8)
    imgs.append(pred)

    depth = results["depth"].reshape(h, w).cpu().numpy()
    depth_img = depth2img(depth)
    depths.append(depth_img)

    imageio.imwrite(os.path.join(frame_dir, f"{img_idx:03d}.png"), pred)
    imageio.imwrite(os.path.join(frame_dir, f"{img_idx:03d}_d.png"), depth_img)

  print(f"mean time: {np.mean(times):.4f} s, FPS: {1/np.mean(times):.2f}")

  # write video
  imageio.mimsave(os.path.join(result_dir, "rgb.mp4"), imgs, fps=15)
  imageio.mimsave(os.path.join(result_dir, "depth.mp4"), depths, fps=15)
  print(f"RGB video is saved to {result_dir}/rgb.mp4")
  print(f"RGB video is saved to {result_dir}/depth.mp4")

  # Ploting camera path
  if args.plot_path:
    print("Plotting camera path...")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    camera_origins = np.array(camera_origins)
    xs = camera_origins[:, 0]
    ys = camera_origins[:, 1]
    zs = camera_origins[:, 2]
    ax.scatter(xs, ys, zs)
    plt.savefig(f"{result_dir}/camera_path.png")
    print("Camera path visualization is saved to {result_dir}/camera_path.png")
