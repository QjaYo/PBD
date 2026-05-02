import math
import numpy as np

FOV_DEG = 45.0  # taichi GGUI 기본값


def _camera_basis(cam_pos, cam_lookat, cam_up):
    cam_pos    = np.asarray(cam_pos,    dtype=np.float32)
    cam_lookat = np.asarray(cam_lookat, dtype=np.float32)
    cam_up_in  = np.asarray(cam_up,     dtype=np.float32)
    forward = cam_lookat - cam_pos
    forward /= (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, cam_up_in)
    right /= (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, forward)
    return cam_pos, forward, right, up


def mouse_ray(cursor, aspect, cam_pos, cam_lookat, cam_up, fov_deg=FOV_DEG):
    """cursor: (x,y) ∈ [0,1] (taichi GGUI: y=0 아래, y=1 위). returns (origin, dir)."""
    origin, forward, right, up = _camera_basis(cam_pos, cam_lookat, cam_up)
    ndc_x = 2.0 * cursor[0] - 1.0
    ndc_y = 2.0 * cursor[1] - 1.0
    t_h = math.tan(math.radians(fov_deg) * 0.5)
    direction = forward + (ndc_x * aspect * t_h) * right + (ndc_y * t_h) * up
    direction /= (np.linalg.norm(direction) + 1e-12)
    return origin.astype(np.float32), direction.astype(np.float32)


def pick(particles_x, cursor, aspect, cam_pos, cam_lookat, cam_up, floor_y,
         screen_radius=0.04, fov_deg=FOV_DEG):
    """
    스크린 공간 picking: 입자는 화면거리 임계 내에서 가장 앞쪽(min depth) 선택,
    바닥은 ray-plane intersection. 더 가까운 쪽 채택.
    반환: ('bunny', (idx, depth)) | ('floor', t) | ('none', None)
    """
    origin, forward, right, up = _camera_basis(cam_pos, cam_lookat, cam_up)
    ndc_x = 2.0 * cursor[0] - 1.0
    ndc_y = 2.0 * cursor[1] - 1.0
    t_h = math.tan(math.radians(fov_deg) * 0.5)

    # 1) 입자 스크린 좌표
    rel = particles_x - origin                       # (N,3)
    z   = rel @ forward                              # depth
    valid = z > 1e-4
    sx = (rel @ right) / (z * t_h * aspect + 1e-12)
    sy = (rel @ up)    / (z * t_h + 1e-12)
    dist2 = (sx - ndc_x) ** 2 + (sy - ndc_y) ** 2
    in_radius = (dist2 < screen_radius * screen_radius) & valid
    best_idx, best_depth = -1, float('inf')
    if np.any(in_radius):
        depths = np.where(in_radius, z, np.inf)
        idx = int(np.argmin(depths))
        best_idx, best_depth = idx, float(depths[idx])

    # 2) 바닥 평면 교차
    direction = forward + (ndc_x * aspect * t_h) * right + (ndc_y * t_h) * up
    direction /= (np.linalg.norm(direction) + 1e-12)
    floor_t = float('inf')
    if abs(direction[1]) > 1e-6:
        cand = (floor_y - origin[1]) / direction[1]
        if cand > 1e-4:
            floor_t = cand

    if best_idx == -1 and not np.isfinite(floor_t):
        return 'none', None
    if floor_t < best_depth:
        return 'floor', floor_t
    return 'bunny', (best_idx, best_depth)
