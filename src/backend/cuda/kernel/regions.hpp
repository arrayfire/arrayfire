/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <debug_cuda.hpp>
#include <stdio.h>
#include <memory.hpp>

#include <thrust/system/cuda/detail/par.h>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>

#if __CUDACC__

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

// This flag is used to track convergence (i.e. it is set, whenever
// any label equivalency changes.  When no labels changed, the
// algorithm is finished and the kernel ends.
__device__ static int continue_flag = 1;

// Wrapper function for texture fetch
template<typename T>
__device__ __inline__
static T fetch(const int n,
               cuda::Param<T> equiv_map,
               cudaTextureObject_t tex)
{
// FIXME: Enable capability >= 3.0
//#if (__CUDA_ARCH__ >= 300)
#if 0
    // Kepler bindless texture objects
    return tex1Dfetch<T>(tex, n);
#else
    return equiv_map.ptr[n];
#endif
}

// The initial label kernel distinguishes between valid (nonzero)
// pixels and "background" (zero) pixels.
template<typename T, int n_per_thread>
__global__
static void initial_label(cuda::Param<T> equiv_map, cuda::CParam<char> bin)
{
    const int base_x = (blockIdx.x * blockDim.x * n_per_thread) + threadIdx.x;
    const int base_y = (blockIdx.y * blockDim.y * n_per_thread) + threadIdx.y;

    // If in bounds and a valid pixel, set the initial label.
    #pragma unroll
    for (int xb = 0; xb < n_per_thread; ++xb) {
        #pragma unroll
        for (int yb = 0; yb < n_per_thread; ++yb) {
            const int x = base_x + (xb * blockDim.x);
            const int y = base_y + (yb * blockDim.y);
            const int n = y * bin.dims[0] + x;
            if (x < bin.dims[0] && y < bin.dims[1]) {
                equiv_map.ptr[n] = (bin.ptr[n] > (char)0) ? n + 1 : 0;
            }
        }
    }
}

template<typename T, int n_per_thread>
__global__
static void final_relabel(cuda::Param<T> equiv_map, cuda::CParam<char> bin, const T* d_tmp)
{
    const int base_x = (blockIdx.x * blockDim.x * n_per_thread) + threadIdx.x;
    const int base_y = (blockIdx.y * blockDim.y * n_per_thread) + threadIdx.y;

    // If in bounds and a valid pixel, set the initial label.
    #pragma unroll
    for (int xb = 0; xb < n_per_thread; ++xb) {
        #pragma unroll
        for (int yb = 0; yb < n_per_thread; ++yb) {
            const int x = base_x + (xb * blockDim.x);
            const int y = base_y + (yb * blockDim.y);
            const int n = y * bin.dims[0] + x;
            if (x < bin.dims[0] && y < bin.dims[1]) {
                equiv_map.ptr[n] = (bin.ptr[n] > (char)0) ? d_tmp[(int)equiv_map.ptr[n]] : (T)0;
            }
        }
    }
}

// When two labels are equivalent, choose the lower label, but
// do not choose zero, which indicates invalid.
template<typename T>
__device__ __inline__
static T relabel(const T a, const T b) {
    return min((a + (cuda::limit_max<T>() * (a == 0))),(b + (cuda::limit_max<T>() * (b == 0))));
}
__device__ __inline__
static double relabel(const double a, const double b) {
    return fmin((a + (cuda::limit_max<double>() * (a == 0))),(b + (cuda::limit_max<double>() * (b == 0))));
}
__device__ __inline__
static float relabel(const float a, const float b) {
    return fminf((a + (cuda::limit_max<float>() * (a == 0))),(b + (cuda::limit_max<float>() * (b == 0))));
}

//Calculates the number of warps at compile time
template<unsigned thread_count>
struct warp_count {
    enum { value = ((thread_count % 32) == 0 ? thread_count/32 : thread_count/32 + 1)};
};

// The following kernel updates the equivalency map.  This kernel
// must be launched with a square block configuration with
// correctly instantiated template parameters.

// The following is the preferred configuration for Kepler:
// block_dim = 16 // 256 threads per block
// num_warps = 8; // (Could compute this from block dim)
// Number of elements to handle per thread in each dimension
// int n_per_thread = 2; // 2x2 per thread = 4 total elems per thread
template <typename T, int block_dim, int n_per_thread, bool full_conn>
__global__
static void update_equiv(cuda::Param<T> equiv_map, const cudaTextureObject_t tex)
{

    typedef warp_count<block_dim*block_dim> num_warps;
#if (__CUDA_ARCH__ >= 120) // This function uses warp ballot instructions
    // Basic coordinates
    const int base_x = (blockIdx.x * blockDim.x * n_per_thread) + threadIdx.x;
    const int base_y = (blockIdx.y * blockDim.y * n_per_thread) + threadIdx.y;

    const int width  = equiv_map.dims[0];
    const int height = equiv_map.dims[1];

    bool tid_changed = false;

    // Per element write flags and label, initially 0
    char write[n_per_thread * n_per_thread];
    T    best_label[n_per_thread * n_per_thread];

    #pragma unroll
    for (int i = 0; i < n_per_thread * n_per_thread; ++i) {
        write[i]      = (char)0;
        best_label[i] = (T)0;
    }

    // Cached tile of the equivalency map
    __shared__ T s_tile[n_per_thread*block_dim][(n_per_thread*block_dim)];

    // Space to track ballot funcs to track convergence
    __shared__ T s_changed[num_warps::value];

    const int tn = (threadIdx.y * blockDim.x) + threadIdx.x;

    const int warpIdx = tn / warpSize;
    s_changed[warpIdx] = (T)0;
    __syncthreads();

#if (__CUDA_ARCH__ >= 130)
    #pragma unroll
#endif
    for (int xb = 0; xb < n_per_thread; ++xb) {
#if (__CUDA_ARCH__ >= 130)
        #pragma unroll
#endif
        for (int yb = 0; yb < n_per_thread; ++yb) {

            // Indexing variables
            const int x = base_x + (xb * blockDim.x);
            const int y = base_y + (yb * blockDim.y);
            const int tx = threadIdx.x + (xb * blockDim.x);
            const int ty = threadIdx.y + (yb * blockDim.y);
            const int tid_i = xb * n_per_thread + yb;
            const int n = y * width + x;

            // Get the label for this pixel if we're  in bounds
            const T orig_label = (x < width && y < height) ?
                fetch<T>(n, equiv_map, tex) : (T)0;
            s_tile[ty][tx] = orig_label;

            // Find the lowest label of the nearest valid pixel
            // So far, all we know is that this pixel is valid.
            best_label[tid_i] = orig_label;

            if (orig_label != (T)0) {
                const int south_y = min(y, height-2) + 1;
                const int north_y = max(y, 1) - 1;
                const int east_x = min(x, width-2) + 1;
                const int west_x = max(x, 1) - 1;

                // Check bottom
                best_label[tid_i] = relabel(best_label[tid_i],
                        fetch((south_y) * width + x, equiv_map, tex));

                // Check right neighbor
                best_label[tid_i] = relabel(best_label[tid_i],
                        fetch(y * width + east_x, equiv_map, tex));

                // Check left neighbor
                best_label[tid_i] = relabel(best_label[tid_i],
                        fetch(y * width + west_x, equiv_map, tex));

                // Check top neighbor
                best_label[tid_i] = relabel(best_label[tid_i],
                        fetch((north_y) * width + x, equiv_map, tex));

                if (full_conn) {
                    // Check NW corner
                    best_label[tid_i] = relabel(best_label[tid_i],
                            fetch((north_y) * width + west_x, equiv_map, tex));

                    // Check NE corner
                    best_label[tid_i] = relabel(best_label[tid_i],
                            fetch((north_y) * width + east_x, equiv_map, tex));

                    // Check SW corner
                    best_label[tid_i] = relabel(best_label[tid_i],
                        fetch((south_y) * width + west_x, equiv_map, tex));

                    // Check SE corner
                    best_label[tid_i] = relabel(best_label[tid_i],
                            fetch((south_y) * width + east_x, equiv_map, tex));
                } // if connectivity == 8
            } // if orig_label != 0

            // Process the equivalency list.
            T last_label = orig_label;
            T new_label  = best_label[tid_i];

            while (best_label[tid_i] != (T)0 && new_label < last_label) {
                last_label = new_label;
                new_label = fetch(new_label - (T)1, equiv_map, tex);
            }

            if (orig_label != new_label) {
                tid_changed = true;
                s_tile[ty][tx] = new_label;
                write[tid_i] = (char)1;
            }
            best_label[tid_i] = new_label;
        }
    }
    __syncthreads();

    // Determine if any pixel changed
    bool continue_iter = false;
    s_changed[warpIdx] = __any((int)tid_changed);
    __syncthreads();

#if (__CUDA_ARCH__ >= 130)
    #pragma unroll
#endif
    for (int i = 0; i < num_warps::value; i++)
        continue_iter = continue_iter || (s_changed[i] != 0);

    // Iterate until no pixel in the tile changes
    while (continue_iter) {

        // Reset whether or not this thread's pixels have changed.
        tid_changed = false;

#if (__CUDA_ARCH__ >= 130)
        #pragma unroll
#endif
        for (int xb = 0; xb < n_per_thread; ++xb) {
#if (__CUDA_ARCH__ >= 130)
            #pragma unroll
#endif
            for (int yb = 0; yb < n_per_thread; ++yb) {

                // Indexing
                const int tx = threadIdx.x + (xb * blockDim.x);
                const int ty = threadIdx.y + (yb * blockDim.y);
                const int tid_i = xb * n_per_thread + yb;

                T last_label = best_label[tid_i];

                if (best_label[tid_i] != 0) {

                    const int north_y   = max(ty, 1) -1;
                    const int south_y   = min(ty, n_per_thread*block_dim - 2) +1;
                    const int east_x    = min(tx, n_per_thread*block_dim - 2) +1;
                    const int west_x    = max(tx, 1) -1;

                    // Check bottom
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[south_y][tx]);

                    // Check right neighbor
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[ty][east_x]);

                    // Check left neighbor
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[ty][west_x]);

                    // Check top neighbor
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[north_y][tx]);

                    if (full_conn) {
                        // Check NW corner
                        best_label[tid_i] = relabel(best_label[tid_i],
                                                    s_tile[north_y][west_x]);

                        // Check NE corner
                        best_label[tid_i] = relabel(best_label[tid_i],
                                                    s_tile[north_y][east_x]);

                        // Check SW corner
                        best_label[tid_i] = relabel(best_label[tid_i],
                                                    s_tile[south_y][west_x]);

                        // Check SE corner
                        best_label[tid_i] = relabel(best_label[tid_i],
                                                    s_tile[south_y][east_x]);
                    } // if connectivity == 8

                    // This thread's value changed during this iteration if the
                    // best label is not the same as the last label.
                    const bool changed = best_label[tid_i] != last_label;
                    write[tid_i] = write[tid_i] || changed;
                    tid_changed  =  tid_changed || changed;
                }
            }
        }
        // Done looking at neighbors for this iteration
        __syncthreads();

        // Decide if we need to continue iterating
        s_changed[warpIdx] = __any((int)tid_changed);
        __syncthreads();
        continue_iter = false;
#if (__CUDA_ARCH__ >= 130)
        #pragma unroll
#endif
        for (int i = 0; i < num_warps::value; i++)
            continue_iter = continue_iter | (s_changed[i] != 0);

        // If we have to continue iterating, update the tile of the
        // equiv map in shared memory
        if (continue_iter) {
#if (__CUDA_ARCH__ >= 130)
            #pragma unroll
#endif
            for (int xb = 0; xb < n_per_thread; ++xb) {
#if (__CUDA_ARCH__ >= 130)
                #pragma unroll
#endif
                for (int yb = 0; yb < n_per_thread; ++yb) {
                    const int tx = threadIdx.x + (xb * blockDim.x);
                    const int ty = threadIdx.y + (yb * blockDim.y);
                    const int tid_i = xb * n_per_thread + yb;
                    // Update tile in shared memory
                    s_tile[ty][tx] = best_label[tid_i];
                }
            }
            __syncthreads();
        }
    } // while (continue_iter)

    // Write out equiv_map
#if (__CUDA_ARCH__ >= 130)
    #pragma unroll
#endif
    for (int xb = 0; xb < n_per_thread; ++xb) {
#if (__CUDA_ARCH__ >= 130)
        #pragma unroll
#endif
        for (int yb = 0; yb < n_per_thread; ++yb) {
            const int x = base_x + (xb * blockDim.x);
            const int y = base_y + (yb * blockDim.y);
            const int n = y * width + x;
            const int tid_i = xb * n_per_thread + yb;
            if (x < width && y < height && write[tid_i]) {
                equiv_map.ptr[n]  = best_label[tid_i];
                continue_flag = 1;
            }
        }
    }
#endif // __CUDA_ARCH__ >= 120
}

template<typename T>
struct clamp_to_one : public thrust::unary_function<T,T>
{
    __host__ __device__ T operator()(const T& in) const
    {
        return (in >= (T)1) ? (T)1 : in;
    }
};

template<typename T, bool full_conn, int n_per_thread>
void regions(cuda::Param<T> out, cuda::CParam<char> in, cudaTextureObject_t tex)
{
    const dim3 threads(THREADS_X, THREADS_Y);

    const int blk_x = divup(in.dims[0], threads.x*2);
    const int blk_y = divup(in.dims[1], threads.y*2);

    const dim3 blocks(blk_x, blk_y);

    CUDA_LAUNCH((initial_label<T,n_per_thread>), blocks, threads, out, in);

    POST_LAUNCH_CHECK();

    int h_continue = 1;

    while (h_continue) {
        h_continue = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(continue_flag, &h_continue, sizeof(int),
                                      0, cudaMemcpyHostToDevice));

        CUDA_LAUNCH((update_equiv<T, 16, n_per_thread, full_conn>), blocks, threads, out, tex);

        POST_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpyFromSymbol(&h_continue, continue_flag, sizeof(int),
                                        0, cudaMemcpyDeviceToHost));
    }

    // Now, perform the final relabeling.  This converts the equivalency
    // map from having unique labels based on the lowest pixel in the
    // component to being sequentially numbered components starting at
    // 1.
    int size = in.dims[0] * in.dims[1];
    T* tmp = cuda::memAlloc<T>(size);
    CUDA_CHECK(cudaMemcpyAsync(tmp, out.ptr, size * sizeof(T),
                          cudaMemcpyDeviceToDevice,
                          cuda::getStream(cuda::getActiveDeviceId())));

    // Wrap raw device ptr
    thrust::device_ptr<T> wrapped_tmp = thrust::device_pointer_cast(tmp);

    // Sort the copy
    THRUST_SELECT(thrust::sort, wrapped_tmp, wrapped_tmp + size);

    // Take the max element, this is the number of label assignments to
    // compute.
    int num_bins = wrapped_tmp[size - 1] + 1;

    thrust::device_vector<T> labels(num_bins);

    // Find the end of each section of values
    thrust::counting_iterator<T> search_begin(0);
    THRUST_SELECT(thrust::upper_bound, wrapped_tmp,  wrapped_tmp  + size,
                        search_begin, search_begin + num_bins,
                        labels.begin());
    THRUST_SELECT(thrust::adjacent_difference, labels.begin(), labels.end(), labels.begin());

    // Operators for the scan
    clamp_to_one<T> clamp;
    thrust::plus<T> add;

    // Perform the scan -- this can computes the correct labels for each
    // component
    THRUST_SELECT(thrust::transform_exclusive_scan,
                                     labels.begin(),
                                     labels.end(),
                                     labels.begin(),
                                     clamp,
                                     0,
                                     add);

    // Apply the correct labels to the equivalency map
    CUDA_LAUNCH((final_relabel<T,n_per_thread>), blocks,threads,
            out, in, thrust::raw_pointer_cast(&labels[0]));

    POST_LAUNCH_CHECK();

    cuda::memFree(tmp);
}

#endif // __CUDACC__
