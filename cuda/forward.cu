/*
 * ==============================================================================
 * 3DGS FORWARD PROJECTION & SPHERICAL HARMONICS CUDA KERNEL
 * ==============================================================================
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

struct Splat2D {
    float2 pos;
    float3 cov2d;
    float4 color;
    float depth;
};

__device__ float3 EvaluateSH(
    int deg,
    const float3& dir,
    const float* sh_coeffs
) {
    // Degree 0 (Base Ambient Color)
    float3 result = make_float3(
        0.28209479f * sh_coeffs[0] + 0.5f,
        0.28209479f * sh_coeffs[1] + 0.5f,
        0.28209479f * sh_coeffs[2] + 0.5f
    );

    if (deg >= 1) {
        float x = dir.x, y = dir.y, z = dir.z;
        float c1 = 0.48860251f;
        result.x += c1 * (-y * sh_coeffs[3] + z * sh_coeffs[6] - x * sh_coeffs[9]);
        result.y += c1 * (-y * sh_coeffs[4] + z * sh_coeffs[7] - x * sh_coeffs[10]);
        result.z += c1 * (-y * sh_coeffs[5] + z * sh_coeffs[8] - x * sh_coeffs[11]);
    }

    result.x = fmaxf(0.0f, result.x);
    result.y = fmaxf(0.0f, result.y);
    result.z = fmaxf(0.0f, result.z);
    return result;
}

__global__ void ProjectGaussiansCUDA(
    int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float* __restrict__ sh_coeffs,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ projmatrix,
    float fx, float fy, float cx, float cy,
    int width, int height,
    Splat2D* __restrict__ out_splats,
    uint2* __restrict__ out_tiles_touched
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float3 p_world = means3d[idx];

    // Transform to Camera Coordinates: p_cam = R * p_world + T
    float p_cam_x = viewmatrix[0] * p_world.x + viewmatrix[4] * p_world.y + viewmatrix[8]  * p_world.z + viewmatrix[12];
    float p_cam_y = viewmatrix[1] * p_world.x + viewmatrix[5] * p_world.y + viewmatrix[9]  * p_world.z + viewmatrix[13];
    float p_cam_z = viewmatrix[2] * p_world.x + viewmatrix[6] * p_world.y + viewmatrix[10] * p_world.z + viewmatrix[14];

    if (p_cam_z < 0.2f) return;

    // Screen Space Center (u, v)
    float u = (p_cam_x * fx / p_cam_z) + cx;
    float v = (p_cam_y * fy / p_cam_z) + cy;

    if (u < -32.0f || u >= (float)width + 32.0f || v < -32.0f || v >= (float)height + 32.0f) return;

    // 3D Scale & Rotation Matrix
    float3 s = scales[idx];
    float4 q = rotations[idx];
    float q_norm = rsqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w + 1e-7f);
    q.x *= q_norm; q.y *= q_norm; q.z *= q_norm; q.w *= q_norm;

    // EWA Jacobian Matrix J
    float J00 = fx / p_cam_z;
    float J02 = -(fx * p_cam_x) / (p_cam_z * p_cam_z);
    float J11 = fy / p_cam_z;
    float J12 = -(fy * p_cam_y) / (p_cam_z * p_cam_z);

    // Approximate 2D Covariance
    float max_s = fmaxf(s.x, fmaxf(s.y, s.z));
    float cov_xx = (max_s * J00) * (max_s * J00) + 0.3f;
    float cov_yy = (max_s * J11) * (max_s * J11) + 0.3f;
    float cov_xy = (max_s * J00) * (max_s * J12) * 0.1f;

    float det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det <= 0.0f) return;

    float inv_det = 1.0f / det;
    float3 inv_cov2d = make_float3(cov_yy * inv_det, -cov_xy * inv_det, cov_xx * inv_det);

    // Color & Opacity
    float3 view_dir = make_float3(p_cam_x, p_cam_y, p_cam_z);
    float dir_norm = rsqrtf(view_dir.x * view_dir.x + view_dir.y * view_dir.y + view_dir.z * view_dir.z + 1e-7f);
    view_dir.x *= dir_norm; view_dir.y *= dir_norm; view_dir.z *= dir_norm;

    float3 color = EvaluateSH(1, view_dir, &sh_coeffs[idx * 12]);
    float opacity = 1.0f / (1.0f + expf(-opacities[idx]));

    out_splats[idx].pos = make_float2(u, v);
    out_splats[idx].cov2d = inv_cov2d;
    out_splats[idx].color = make_float4(color.x, color.y, color.z, opacity);
    out_splats[idx].depth = p_cam_z;
}
