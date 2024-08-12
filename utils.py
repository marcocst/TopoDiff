import torch
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import os
import numpy as np
import cv2

def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap

def compute_location_loss(attn_maps_mid, attn_maps_up, locations, object_positions, regression):
    loss = 0
    object_number = len(locations)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()

    left_centric = np.zeros((32, 32), dtype=np.float32)
    draw_msra_gaussian(left_centric, (32 * 0.25, 32 * 0.5), 32 / 8)
    
    middle_centric = np.zeros((32, 32), dtype=np.float32)
    draw_msra_gaussian(middle_centric, (32 * 0.5, 32 * 0.5), 32 / 8)

    right_centric = np.zeros((32, 32), dtype=np.float32)
    draw_msra_gaussian(right_centric, (32 * 0.75, 32 * 0.5), 32 / 8)
    centrics = [torch.from_numpy(left_centric[None, None]).cuda(), torch.from_numpy(middle_centric[None, None]).cuda(), torch.from_numpy(right_centric[None, None]).cuda()]
    delta_init = [0.3 * 2 - 1, 0, 0.7 * 2 - 1]
    delta_range = [((-0.9) - delta_init[0], (+0.9) - delta_init[0]), ((-0.5) - delta_init[0], (+0.5) - delta_init[0]), ((-0.9) - delta_init[1], (0.9) - delta_init[1])]

    loss = 0
    featuremaps = attn_maps_up[0] + attn_maps_mid

    global_features = []
    for attn_map_integrated, features in featuremaps:
        features = features.max(0)[0].T
        # xy       = regression(features)
        global_features.append(features)

    global_xys = regression(torch.stack(global_features).max(0)[0])
    for attn_map_integrated, features in featuremaps:
        # features = features.max(0)[0].T
        attn_map = attn_map_integrated.chunk(2)[1]
        
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        grid   = torch.stack([xs, ys], dim=-1).cuda()
        # left_mask = np.zeros((H, W), dtype=np.float32)
        # draw_msra_gaussian(left_mask, (W * 0.25, H * 0.5), W / 8)

        # right_mask = np.zeros((H, W), dtype=np.float32)
        # draw_msra_gaussian(right_mask, (W * 0.75, H * 0.5), W / 8)
        # draw_msra_gaussian(right_mask, (W * 0.75, H * 0.75), W / 16)
        # t_loss = ((right.x - left.x) - 0.25) ** 2   # pull loss, push loss
        #          right.x - left.x < 0.25   有loss
        #          right.x - left.x > 0.25   无loss 
        #          (min（right.x - left.x, 0.25） - 0.25) ** 2
        offsets = []
        for i, items in enumerate(object_positions):
            ntoken = len(items)
            delta = ((global_xys[items].sum(0) / 200).sigmoid() * 2 - 1).clamp(*delta_range[i])
            
            # delta = 0
            offset = grid - delta
            offsets.append(delta)
            # print(delta.cpu().data.numpy().tolist())
            # target = generate_target(x, y, centrics[i], W, H, grid)
            target = torch.nn.functional.grid_sample(centrics[i], offset[None], "bilinear", "zeros", True)[0, 0]
            activation = attn_map[:, :, items].reshape(b, H, W, ntoken).max(0)[0]
            # loss += ((activation - target.unsqueeze(2)) ** 2).sum() / (W * H * ntoken)
            
            activation_value = (activation * target.unsqueeze(2)).reshape(b, -1).sum(dim=-1) / activation.reshape(b, -1).sum(dim=-1)
            # obj_loss += torch.mean((ca_map_obj - mask) ** 2)
            # obj_loss += -(mask * torch.log(ca_map_obj) + (1 - mask) * torch.log(1 - ca_map_obj)).mean()
            # obj_loss += _neg_loss(ca_map_obj, mask)

            loss += torch.mean((1 - activation_value) ** 2) * 0.5 #@ * 0.3

        left_offsetx  = offsets[0][0] + delta_init[0]
        middle_offsetx = offsets[1][0] + delta_init[1]
        right_offsetx = offsets[2][0] + delta_init[2]
        # print(f"{(left_offsetx).item()}, {right_offsetx.item()}")
        loss += abs(min(right_offsetx - left_offsetx, 1) - 1) * 0.2  # pull loss, push loss
        loss += abs(min(middle_offsetx - left_offsetx, 0.2) - 0.2) * 0.2  # pull loss, push loss
        loss += abs(min(right_offsetx - middle_offsetx, 0.2) - 0.2) * 0.2  # pull loss, push loss

    loss = loss / len(featuremaps)
    return loss * 0.5

def Pharse2idx(prompt, phrases):
    phrases = [x.strip() for x in phrases.split('|')]
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions

def draw_box(pil_img, bboxes, phrases, save_path):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype('./FreeMono.ttf', 25)
    phrases = [x.strip() for x in phrases.split(';')]
    for obj_bboxes, phrase in zip(bboxes, phrases):
        for obj_bbox in obj_bboxes:
            x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            draw.rectangle([int(x_0 * 512), int(y_0 * 512), int(x_1 * 512), int(y_1 * 512)], outline='red', width=5)
            draw.text((int(x_0 * 512) + 5, int(y_0 * 512) + 5), phrase, font=font, fill=(255, 0, 0))
    pil_img.save(save_path)



def setup_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(os.path.join(save_path, f"{logger_name}.log"))
    file_handler.setLevel(logging.INFO)

    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger