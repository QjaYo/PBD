import taichi as ti

EPSILON = 1e-8


@ti.kernel
def apply_floor_friction(
    particles: ti.template(),
    mu_s: ti.f32,
    mu_k: ti.f32,
    dt: ti.f32
):
    n = ti.math.vec3(0.0, 1.0, 0.0)

    for i in particles.x:
        if particles.w[i] == 0.0:
            continue

        dx_coll = particles.dx_coll[i]
        if dx_coll.norm() < EPSILON:
            continue

        v_n = particles.v[i].dot(n) * n
        v_t = particles.v[i] - v_n
        v_t_norm = v_t.norm()

        if particles.was_moving[i] == 0:
            if v_t_norm < mu_s * dx_coll.norm() / dt:
                particles.v[i] = v_n
            else:
                particles.was_moving[i] = 1
                dv_k_norm = mu_k * dx_coll.norm() / dt
                reduction = ti.min(dv_k_norm, v_t_norm)
                particles.v[i] -= reduction * v_t.normalized()
        else:
            if v_t_norm < EPSILON:
                particles.v[i] = v_n
                particles.was_moving[i] = 0
            else:
                dv_k_norm = mu_k * dx_coll.norm() / dt
                reduction = ti.min(dv_k_norm, v_t_norm)
                particles.v[i] -= reduction * v_t.normalized()
                if (particles.v[i] - v_n).norm() < EPSILON:
                    particles.was_moving[i] = 0



'''
        v_n = ti.abs(particles.v[i].y)
        v_t = ti.math.vec3(particles.v[i].x, 0.0, particles.v[i].z)
        v_t_norm = v_t.norm()

        if v_t_norm < EPSILON:
            continue

        if v_t_norm <= mu_s * v_n:
            # 정지마찰: 접선 속도 제거
            particles.v[i].x = 0.0
            particles.v[i].z = 0.0
        else:
            # 운동마찰: 접선 속도를 mu_k * v_n 만큼 감소
            scale = ti.max(0.0, 1.0 - mu_k * v_n / v_t_norm)
            particles.v[i].x *= scale
            particles.v[i].z *= scale
'''