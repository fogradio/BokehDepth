import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# args

def parse_block_ids(arg):
    result = []
    for part in arg.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result

def compress_block_ids(block_ids):
    if not block_ids:
        return ""
    block_ids = sorted(set(block_ids))
    ranges = []
    start = prev = block_ids[0]
    for num in block_ids[1:]:
        if num == prev + 1:
            prev = num
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = num
    # Add the final range
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")
    return "_".join(ranges)

# image utils

def save_gif_from_images(images, gif_path, duration=500, loop=0):
    images = [img.convert('RGB') for img in images]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        format='GIF'
    )

def save_video_from_images(images, video_path, fps=2):
    frames = [np.array(img.convert('RGB')) for img in images]
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()

def side_by_side_collage(images):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    collage = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in images:
        collage.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return collage

# attention utils

def attnmaps2images(net_attn_maps, upscale=False, average=False, target_size=(64, 64)):
    if len(list(net_attn_maps.values())) == 0:
        return None

    if upscale:
        for name, attn_map in net_attn_maps.items():
            net_attn_maps[name] = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)[0, 0]

    if average:
        assert upscale, "Average attn map requires upscale to be True"
        all_maps = torch.stack(list(net_attn_maps.values()))
        attn_map = all_maps.mean(dim=0)
        attn_map = attn_map.detach().cpu().float().numpy()
        min_val = np.min(attn_map)
        max_val = np.max(attn_map)
        if np.isclose(min_val, max_val):
            normalized_attn_map = np.full_like(attn_map, 128, dtype=np.uint8)
        else:
            normalized_attn_map = ((attn_map - min_val) / (max_val - min_val) * 255).clip(0, 255)
            normalized_attn_map = normalized_attn_map.astype(np.uint8)
        image = Image.fromarray(normalized_attn_map)
        return image

    images = {}
    for name, attn_map in net_attn_maps.items():
        attn_map = attn_map.detach().cpu().numpy()
        min_val = np.min(attn_map)
        max_val = np.max(attn_map)
        if np.isclose(min_val, max_val):
            normalized_attn_map = np.full_like(attn_map, 128, dtype=np.uint8)
        else:
            normalized_attn_map = ((attn_map - min_val) / (max_val - min_val) * 255).clip(0, 255)
            normalized_attn_map = normalized_attn_map.astype(np.uint8)
        image = Image.fromarray(normalized_attn_map)
        images[name] = image
    return images
