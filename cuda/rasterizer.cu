/*
 * ==============================================================================
 * 3D GAUSSIAN SPLATTING (3DGS) CUDA TILE-BASED RASTERIZER
 * High-Performance C++/CUDA Implementation for NVIDIA RTX 3090 (Ampere sm_86)
 * ==============================================================================
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define NUM_TILES_X(w) (((w) + BLOCK_X - 1) / BLOCK_X)
#define NUM_TILES_Y(h) (((h) + BLOCK_Y - 1) / BLOCK_Y)

// Spherical Harmonics constants
__constant__ float C0 = 0.28209479177387814f;
__constant__ float C1 = 0.4886025119029199f;
__constant__ float C2[5] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};

struct __align__(16) Splat2D {
    float2 pos;
    float3 cov2d;   // a, b, c (inverse covariance)
    float4 color;   // rgb + opacity
    float depth;
};

// ------------------------------------------------------------------------------
// CUDA KERNEL: Forward Tile Rasterization with Volumetric Transmittance
// ------------------------------------------------------------------------------
__global__ void RasterizeSplatsCUDA(
    const Splat2D* __restrict__ splats,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ splat_indices,
    float4* __restrict__ out_pixels,
    int width,
    int height
) {
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tile_idx = tile_y * gridDim.x + tile_x;

    int pixel_x = tile_x * BLOCK_X + threadIdx.x;
    int pixel_y = tile_y * BLOCK_Y + threadIdx.y;
    int pixel_idx = pixel_y * width + pixel_x;

    bool valid_pixel = (pixel_x < width && pixel_y < height);
    float2 p = make_float2((float)pixel_x + 0.5f, (float)pixel_y + 0.5f);

    float3 color_accum = make_float3(0.0f, 0.0f, 0.0f);
    float T = 1.0f; // Transmittance (starts at 100%)

    int start = tile_offsets[tile_idx];
    int end = tile_offsets[tile_idx + 1];

    // Shared memory cache for tile splats
    __shared__ Splat2D shared_splats[BLOCK_X * BLOCK_Y];

    for (int i = start; i < end; i += BLOCK_X * BLOCK_Y) {
        int load_idx = i + threadIdx.y * BLOCK_X + threadIdx.x;
        if (load_idx < end) {
            int s_idx = splat_indices[load_idx];
            shared_splats[threadIdx.y * BLOCK_X + threadIdx.x] = splats[s_idx];
        }
        __syncthreads();

        int batch_size = min(BLOCK_X * BLOCK_Y, end - i);
        for (int j = 0; j < batch_size; j++) {
            if (T < 0.0001f) break; // Early ray termination

            if (valid_pixel) {
                const Splat2D& s = shared_splats[j];
                float2 d = make_float2(p.x - s.pos.x, p.y - s.pos.y);

                // Compute exponent: -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
                float power = -0.5f * (s.cov2d.x * d.x * d.x + 2.0f * s.cov2d.y * d.x * d.y + s.cov2d.z * d.y * d.y);
                if (power > 0.0f || power < -4.0f) continue;

                float alpha = min(0.99f, s.color.w * expf(power));
                if (alpha < 1.0f / 255.0f) continue;

                float weight = alpha * T;
                color_accum.x += s.color.x * weight;
                color_accum.y += s.color.y * weight;
                color_accum.z += s.color.z * weight;
                T *= (1.0f - alpha);
            }
        }
        __syncthreads();
    }

    if (valid_pixel) {
        // Final pixel write with background blending
        float bg = 0.0f;
        out_pixels[pixel_idx] = make_float4(
            color_accum.x + T * bg,
            color_accum.y + T * bg,
            color_accum.z + T * bg,
            1.0f - T
        );
    }
}

// C++ Host API
extern "C" void LaunchRasterizerCUDA(
    const Splat2D* d_splats,
    const int* d_tile_offsets,
    const int* d_splat_indices,
    float4* d_out_pixels,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 grid(NUM_TILES_X(width), NUM_TILES_Y(height));
    dim3 block(BLOCK_X, BLOCK_Y);

    RasterizeSplatsCUDA<<<grid, block, 0, stream>>>(
        d_splats,
        d_tile_offsets,
        d_splat_indices,
        d_out_pixels,
        width,
        height
    );
}
