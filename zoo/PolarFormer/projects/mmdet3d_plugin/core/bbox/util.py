import torch 
import math

def normalize_bbox(bboxes, pc_range):
    
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]
    # change to polar coord
    theta_center = torch.atan2(cx, cy)
    # radius_center = torch.sqrt(cx**2 + cy**2)
    delta_rot = rot - theta_center
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        v_abs = torch.sqrt(vx**2 + vy**2)
        v_dir = torch.atan2(vx, vy)
        delta_vel = v_dir - theta_center
        v_theta = v_abs * delta_vel.sin()
        v_r = v_abs * delta_vel.cos()
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, delta_rot.sin(), delta_rot.cos(), v_theta, v_r), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, delta_rot.sin(), delta_rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # change rot and vel to polar coordinate
    # rotation 
    delta_rot_sine = normalized_bboxes[..., 6:7]

    delta_rot_cosine = normalized_bboxes[..., 7:8]
    delta_rot = torch.atan2(delta_rot_sine, delta_rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    theta_center = torch.atan2(cx, cy)
    rot = theta_center + delta_rot  # 
    rot[rot<-math.pi] += 2 * math.pi
    rot[rot>math.pi] -= 2* math.pi
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
        # velocity 
        v_theta = normalized_bboxes[:, 8:9]
        v_r = normalized_bboxes[:, 9:10]
        v_abs = torch.sqrt(v_theta**2 + v_r**2)
        delta_vel = torch.atan2(v_theta, v_r)
        v_dir = delta_vel + theta_center
        vx = v_abs * v_dir.sin()
        vy = v_abs * v_dir.cos()
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes