#include <af/defines.h>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <stdio.h>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>

#if defined(__clang__) && (__clang_major__ >= 5) && (__clang_minor__ >= 1)

#pragma clang diagnostic ignored "-Wunused-const-variable"
#endif

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

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
template<typename T>
__global__
static void initial_label(cuda::Param<T> equiv_map, cuda::CParam<cuda::uchar> bin)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = y * bin.dims[0] + x;

    // If in bounds and a valid pixel, set the initial label.
    if (x < bin.dims[0] && y < bin.dims[1])
        equiv_map.ptr[n] = bin.ptr[n] ? n + 1 : 0;
}
                          
template<typename T>
__global__
static void final_relabel(cuda::Param<T> equiv_map, cuda::CParam<cuda::uchar> bin, const T* d_tmp)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = y * bin.dims[0] + x;

    // If in bounds and a valid pixel, set the initial label.
    if (x < bin.dims[0] && y < bin.dims[1])
        equiv_map.ptr[n] = (bin.ptr[n] == (cuda::uchar)1) ? d_tmp[(int)equiv_map.ptr[n]] : (T)0;
}

// When two labels are equivalent, choose the lower label, but
// do not choose zero, which indicates invalid.
template<typename T>
__device__ __inline__
static T relabel(const T a, const T b) {
    if (a == (T)0) return b;
    if (b == (T)0) return a;
    else           return (T)fminf((float)a,(float)b);
}

// The following kernel updates the equivalency map.  This kernel
// must be launched with a square block configuration with
// correctly instantiated template parameters.

// The following is the preferred configuration for Kepler:
// block_dim = 16 // 256 threads per block
// num_warps = 8; // (Could compute this from block dim)
// Number of elements to handle per thread in each dimension
// int n_per_thread = 2; // 2x2 per thread = 4 total elems per thread
template <typename T, int block_dim, int num_warps, int n_per_thread, bool full_conn>
__global__
static void update_equiv(cuda::Param<T> equiv_map, const cudaTextureObject_t tex)
{

#if (__CUDA_ARCH__ >= 120) // This function uses warp ballot instructions
    // Basic coordinates
    const int base_x = (blockIdx.x * blockDim.x * n_per_thread) + threadIdx.x;
    const int base_y = (blockIdx.y * blockDim.y * n_per_thread) + threadIdx.y;

    const int width  = equiv_map.dims[0];
    const int height = equiv_map.dims[1];

    bool tid_changed = false;

    // Per element write flags and label, initially 0
    cuda::uchar      write[n_per_thread * n_per_thread];
    T     best_label[n_per_thread * n_per_thread];

    #pragma unroll
    for (int i = 0; i < n_per_thread * n_per_thread; ++i) {
        write[i]      = (cuda::uchar)0;
        best_label[i] = (T)0;
    }

    // Cached tile of the equivalency map
    __shared__ T s_tile[n_per_thread*block_dim][(n_per_thread*block_dim)+1];

    // Space to track ballot funcs to track convergence
    __shared__ T s_changed[num_warps];

    const int tn = (threadIdx.y * blockDim.x) + threadIdx.x;

    const int warpIdx = tn / warpSize;
    s_changed[warpIdx] = (T)0;
    __syncthreads();

    #pragma unroll
    for (int xb = 0; xb < n_per_thread; ++xb) {
        #pragma unroll
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

                // Check bottom
                if (y < height - 1)
                    best_label[tid_i] = relabel(best_label[tid_i],
                            fetch((y+1) * width + x, equiv_map, tex));

                // Check right neighbor
                if (x < width - 1)
                    best_label[tid_i] = relabel(best_label[tid_i],
                            fetch(y * width + x + 1, equiv_map, tex));

                // Check left neighbor
                if (x > 0)
                    best_label[tid_i] = relabel(best_label[tid_i],
                            fetch(y * width + x - 1, equiv_map, tex));

                // Check top neighbor
                if (y > 0)
                    best_label[tid_i] = relabel(best_label[tid_i],
                            fetch((y-1) * width + x, equiv_map, tex));

                if (full_conn) {
                    // Check NW corner
                    if (x > 0 && y > 0)
                        best_label[tid_i] = relabel(best_label[tid_i],
                                fetch((y-1) * width + x - 1, equiv_map, tex));

                    // Check NE corner
                    if (x < width - 1 && y > 0)
                        best_label[tid_i] = relabel(best_label[tid_i],
                                fetch((y-1) * width + x + 1, equiv_map, tex));

                    // Check SW corner
                    if (x > 0 && y < height - 1)
                        best_label[tid_i] = relabel(best_label[tid_i],
                            fetch((y+1) * width + x - 1, equiv_map, tex));

                    // Check SE corner
                    if (x < width - 1 && y < height - 1)
                        best_label[tid_i] = relabel(best_label[tid_i],
                                fetch((y+1) * width + x+1, equiv_map, tex));
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
                write[tid_i] = (cuda::uchar)1;
            }
            best_label[tid_i] = new_label;
        }
    }
    __syncthreads();

    // Determine if any pixel changed
    bool continue_iter = false;
    s_changed[warpIdx] = __any((int)tid_changed);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < num_warps; i++)
        continue_iter = continue_iter || (s_changed[i] != 0);

    // Iterate until no pixel in the tile changes
    while (continue_iter) {

        // Reset whether or not this thread's pixels have changed.
        tid_changed = false;

        #pragma unroll
        for (int xb = 0; xb < n_per_thread; ++xb) {
            #pragma unroll
            for (int yb = 0; yb < n_per_thread; ++yb) {

                // Indexing
                const int tx = threadIdx.x + (xb * blockDim.x);
                const int ty = threadIdx.y + (yb * blockDim.y);
                const int tid_i = xb * n_per_thread + yb;

                T last_label = best_label[tid_i];

                if (best_label[tid_i] != 0) {

                    // Check bottom
                    if (ty < n_per_thread*block_dim - 1)
                        best_label[tid_i] = relabel(best_label[tid_i],
                                                    s_tile[ty+1][tx]);

                    // Check right neighbor
                    if (tx < n_per_thread*block_dim - 1)
                        best_label[tid_i] = relabel(best_label[tid_i],
                                                    s_tile[ty][tx+1]);

                    // Check left neighbor
                    if (tx > 0)
                        best_label[tid_i] = relabel(best_label[tid_i],
                                                    s_tile[ty][tx-1]);

                    // Check top neighbor
                    if (ty > 0)
                        best_label[tid_i] = relabel(best_label[tid_i],
                                                    s_tile[ty-1][tx]);

                    if (full_conn) {
                        // Check NW corner
                        if (tx > 0 && ty > 0)
                            best_label[tid_i] = relabel(best_label[tid_i],
                                                        s_tile[ty-1][tx-1]);

                        // Check NE corner
                        if (tx < n_per_thread*block_dim - 1 && ty > 0)
                            best_label[tid_i] = relabel(best_label[tid_i],
                                                        s_tile[ty-1][tx+1]);

                        // Check SW corner
                        if (tx > 0 && ty < n_per_thread*block_dim - 1)
                            best_label[tid_i] = relabel(best_label[tid_i],
                                                        s_tile[ty+1][tx-1]);

                        // Check SE corner
                        if (tx < n_per_thread*block_dim - 1 &&
                            ty < n_per_thread*block_dim - 1)
                            best_label[tid_i] = relabel(best_label[tid_i],
                                                        s_tile[ty+1][tx+1]);
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
        #pragma unroll
        for (int i = 0; i < num_warps; i++)
            continue_iter = continue_iter || (s_changed[i] != 0);

        // If we have to continue iterating, update the tile of the
        // equiv map in shared memory
        if (continue_iter) {
            #pragma unroll
            for (int xb = 0; xb < n_per_thread; ++xb) {
                #pragma unroll
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
    #pragma unroll
    for (int xb = 0; xb < n_per_thread; ++xb) {
        #pragma unroll
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

template<typename T, bool full_conn>
void regions(cuda::Param<T> out, cuda::CParam<cuda::uchar> in, cudaTextureObject_t tex)
{
    const dim3 threads(THREADS_X, THREADS_Y);

    const dim_type blk_x = divup(in.dims[0], threads.x);
    const dim_type blk_y = divup(in.dims[1], threads.y);

    const dim3 blocks(blk_x, blk_y);

    (initial_label<T>)<<<blocks, threads>>>(out, in);

    POST_LAUNCH_CHECK();

    int h_continue = 1;

    while (h_continue) {
        h_continue = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(continue_flag, &h_continue, sizeof(int),
                                      0, cudaMemcpyHostToDevice));
        (update_equiv<T, 16, 8, 1, full_conn>)<<<blocks, threads>>>
            (out, tex);
        CUDA_CHECK(cudaMemcpyFromSymbol(&h_continue, continue_flag, sizeof(int),
                                        0, cudaMemcpyDeviceToHost));
    }

    // Now, perform the final relabeling.  This converts the equivalency
    // map from having unique labels based on the lowest pixel in the
    // component to being sequentially numbered components starting at
    // 1.
    int size = in.dims[0] * in.dims[1];
    T* tmp;
    CUDA_CHECK(cudaMalloc((void **)&tmp,  size * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(tmp, out.ptr, size * sizeof(T),
            cudaMemcpyDeviceToDevice));

    // Wrap raw device ptr
    thrust::device_ptr<T> wrapped_tmp = thrust::device_pointer_cast(tmp);

    // Sort the copy
    thrust::sort(wrapped_tmp, wrapped_tmp + size);

    // Take the max element, this is the number of label assignments to
    // compute.
    int num_bins = wrapped_tmp[size - 1] + 1;

    thrust::device_vector<T> labels(num_bins);

    // Find the end of each section of values
    thrust::counting_iterator<T> search_begin(0);
    thrust::upper_bound(wrapped_tmp,  wrapped_tmp  + size,
                        search_begin, search_begin + num_bins,
                        labels.begin());
    thrust::adjacent_difference(labels.begin(), labels.end(), labels.begin());

    // Operators for the scan
    clamp_to_one<T> clamp;
    thrust::plus<T> add;

    // Perform the scan -- this can computes the correct labels for each
    // component
    thrust::transform_exclusive_scan(labels.begin(),
                                     labels.end(),
                                     labels.begin(),
                                     clamp,
                                     0,
                                     add);

    // Apply the correct labels to the equivalency map
    (final_relabel<T>)<<<blocks,threads>>>(out,
                                           in,
                                           thrust::raw_pointer_cast(&labels[0]));

    CUDA_CHECK(cudaFree(tmp));
}
