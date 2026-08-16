/*
 * ==============================================================================
 * 3DGS BACKWARD GRADIENT & ADAPTIVE DENSIFICATION CUDA KERNEL
 * ==============================================================================
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void ComputePositionalGradientsCUDA(
    int num_splats,
    const float2* __restrict__ dL_dmeans2d,
    float* __restrict__ out_grad_accum,
    float* __restrict__ out_denom
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_splats) return;

    float2 g = dL_dmeans2d[idx];
    float norm = sqrtf(g.x * g.x + g.y * g.y);

    if (norm > 0.0f) {
        atomicAdd(&out_grad_accum[idx], norm);
        atomicAdd(&out_denom[idx], 1.0f);
    }
}
