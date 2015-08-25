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
#include <debug_cuda.hpp>
#include <memory.hpp>

#include <convolve_common.hpp>
#include "convolve.hpp"
#include "orb_patch.hpp"
#include "sort_index.hpp"

#include <boost/scoped_ptr.hpp>

using std::vector;
using boost::scoped_ptr;

namespace cuda
{

namespace kernel
{

static const int THREADS   = 256;
static const int THREADS_X = 16;
static const int THREADS_Y = 16;

static const float PI_VAL = 3.14159265358979323846f;

template<typename T>
void gaussian1D(T* out, const int dim, double sigma=0.0)
{
    if(!(sigma>0)) sigma = 0.25*dim;

    T sum = (T)0;
    for(int i=0;i<dim;i++)
    {
        int x = i-(dim-1)/2;
        T el = 1. / sqrt(2 * PI_VAL * sigma*sigma) * exp(-((x*x)/(2*(sigma*sigma))));
        out[i] = el;
        sum   += el;
    }

    for(int k=0;k<dim;k++)
        out[k] /= sum;
}

inline __device__ float block_reduce_sum(float val)
{
    __shared__ float data[THREADS_X*THREADS_Y];

    unsigned idx = threadIdx.x * blockDim.x + threadIdx.y;

    data[idx] = val;
    __syncthreads();

    for (unsigned i = blockDim.y / 2; i > 0; i >>= 1)
    {
        if (threadIdx.y < i)
        {
            data[idx] += data[idx + i];
        }

        __syncthreads();
    }

    return data[threadIdx.x * blockDim.x];
}

template<typename T>
__global__ void keep_features(
    float* x_out,
    float* y_out,
    float* score_out,
    float* size_out,
    const float* x_in,
    const float* y_in,
    const float* score_in,
    const unsigned* score_idx,
    const float* size_in,
    const unsigned n_feat)
{
    unsigned f = blockDim.x * blockIdx.x + threadIdx.x;

    // Keep only the first n_feat features
    if (f < n_feat) {
        x_out[f] = x_in[score_idx[f]];
        y_out[f] = y_in[score_idx[f]];
        score_out[f] = score_in[f];
        if (size_in != NULL && size_out != NULL)
            size_out[f] = size_in[score_idx[f]];
    }
}

template<typename T, bool use_scl>
__global__ void harris_response(
        float* score_out,
        float* size_out,
        const float* x_in,
        const float* y_in,
        const float* scl_in,
        const unsigned total_feat,
        CParam<T> image,
        const unsigned block_size,
        const float k_thr,
        const unsigned patch_size)
{
    unsigned f = blockDim.x * blockIdx.x + threadIdx.x;

    float ixx = 0.f, iyy = 0.f, ixy = 0.f;
    float size = 0.f;

    if (f < total_feat) {
        unsigned x, y;
        float scl = 1.f;
        if (use_scl) {
            // Update x and y coordinates according to scale
            scl = scl_in[f];
            x = (unsigned)round(x_in[f] * scl);
            y = (unsigned)round(y_in[f] * scl);
        }
        else {
            x = (unsigned)round(x_in[f]);
            y = (unsigned)round(y_in[f]);
        }

        // Round feature size to nearest odd integer
        size = 2.f * floor((patch_size * scl) / 2.f) + 1.f;

        // Avoid keeping features that might be too wide and might not fit on
        // the image, sqrt(2.f) is the radius when angle is 45 degrees and
        // represents widest case possible
        unsigned patch_r = ceil(size * sqrt(2.f) / 2.f);
        if (x < patch_r || y < patch_r || x >= image.dims[1] - patch_r || y >= image.dims[0] - patch_r)
            return;

        unsigned r = block_size / 2;

        unsigned block_size_sq = block_size * block_size;
        for (unsigned k = threadIdx.y; k < block_size_sq; k += blockDim.y) {
            int i = k / block_size - r;
            int j = k % block_size - r;

            // Calculate local x and y derivatives
            float ix = image.ptr[(x+i+1) * image.dims[0] + y+j] - image.ptr[(x+i-1) * image.dims[0] + y+j];
            float iy = image.ptr[(x+i) * image.dims[0] + y+j+1] - image.ptr[(x+i) * image.dims[0] + y+j-1];

            // Accumulate second order derivatives
            ixx += ix*ix;
            iyy += iy*iy;
            ixy += ix*iy;
        }
    }
    __syncthreads();

    ixx = block_reduce_sum(ixx);
    iyy = block_reduce_sum(iyy);
    ixy = block_reduce_sum(ixy);

    if (f < total_feat && threadIdx.y == 0) {
        float tr = ixx + iyy;
        float det = ixx*iyy - ixy*ixy;

        // Calculate Harris responses
        float resp = det - k_thr * (tr*tr);

        // Scale factor
        // TODO: improve response scaling
        float rscale = 0.001f;
        rscale = rscale * rscale * rscale * rscale;

        score_out[f] = resp * rscale;
        if (use_scl)
            size_out[f] = size;
    }
}

template<typename T>
__global__ void centroid_angle(
    const float* x_in,
    const float* y_in,
    float* orientation_out,
    const unsigned total_feat,
    CParam<T> image,
    const unsigned patch_size)
{
    unsigned f = blockDim.x * blockIdx.x + threadIdx.x;

    if (f < total_feat) {
        unsigned x = (unsigned)round(x_in[f]);
        unsigned y = (unsigned)round(y_in[f]);

        unsigned r = patch_size / 2;
        if (x < r || y < r || x > image.dims[1] - r || y > image.dims[0] - r)
            return;

        T m01 = (T)0, m10 = (T)0;
        unsigned patch_size_sq = patch_size * patch_size;
        for (unsigned k = threadIdx.y; k < patch_size_sq; k += blockDim.y) {
            int i = k / patch_size - r;
            int j = k % patch_size - r;

            // Calculate first order moments
            T p = image.ptr[(x+i) * image.dims[0] + y+j];
            m01 += j * p;
            m10 += i * p;
        }

        m01 = block_reduce_sum(m01);
        m10 = block_reduce_sum(m10);

        if (threadIdx.y == 0) {
            float angle = atan2((float)m01, (float)m10);
            orientation_out[f] = angle;
        }
    }
}

template<typename T>
inline __device__ T get_pixel(
    unsigned x,
    unsigned y,
    const float ori,
    const unsigned size,
    const int dist_x,
    const int dist_y,
    CParam<T> image,
    const unsigned patch_size)
{
    float ori_sin = sin(ori);
    float ori_cos = cos(ori);
    float patch_scl = (float)size / (float)patch_size;

    // Calculate point coordinates based on orientation and size
    x += round(dist_x * patch_scl * ori_cos - dist_y * patch_scl * ori_sin);
    y += round(dist_x * patch_scl * ori_sin + dist_y * patch_scl * ori_cos);

    return image.ptr[x * image.dims[0] + y];
}

template<typename T>
__global__ void extract_orb(
    unsigned* desc_out,
    const unsigned n_feat,
    float* x_in_out,
    float* y_in_out,
    const float* ori_in,
    float* size_out,
    CParam<T> image,
    const float scl,
    const unsigned patch_size)
{
    unsigned f = blockDim.x * blockIdx.x + threadIdx.x;

    if (f < n_feat) {
        unsigned x = (unsigned)round(x_in_out[f]);
        unsigned y = (unsigned)round(y_in_out[f]);
        float ori = ori_in[f];
        unsigned size = patch_size;

        unsigned r = ceil(patch_size * sqrt(2.f) / 2.f);
        if (x < r || y < r || x >= image.dims[1] - r || y >= image.dims[0] - r)
            return;

        // Descriptor fixed at 256 bits for now
        // Storing descriptor as a vector of 8 x 32-bit unsigned numbers
        for (unsigned i = threadIdx.y; i < 16; i += blockDim.y) {
            unsigned v = 0;

            // j < 16 for 256 bits descriptor
            for (unsigned j = 0; j < 16; j++) {
                // Get position from distribution pattern and values of points p1 and p2
                int dist_x = d_ref_pat[i*16*4 + j*4];
                int dist_y = d_ref_pat[i*16*4 + j*4+1];
                T p1 = get_pixel(x, y, ori, size, dist_x, dist_y, image, patch_size);

                dist_x = d_ref_pat[i*16*4 + j*4+2];
                dist_y = d_ref_pat[i*16*4 + j*4+3];
                T p2 = get_pixel(x, y, ori, size, dist_x, dist_y, image, patch_size);

                // Calculate bit based on p1 and p2 and shifts it to correct position
                v |= (p1 < p2) << (j + 16*(i % 2));
            }

            // Store 16 bits of descriptor
            atomicAdd(&desc_out[f * 8 + i/2], v);
        }

        if (threadIdx.y == 0) {
            x_in_out[f] = round(x * scl);
            y_in_out[f] = round(y * scl);
            size_out[f] = patch_size * scl;
        }
    }
}



template<typename T, typename convAccT>
void orb(unsigned* out_feat,
         float** d_x,
         float** d_y,
         float** d_score,
         float** d_ori,
         float** d_size,
         unsigned** d_desc,
         vector<unsigned>& feat_pyr,
         vector<float*>& d_x_pyr,
         vector<float*>& d_y_pyr,
         vector<unsigned>& lvl_best,
         vector<float>& lvl_scl,
         vector<CParam<T> >& img_pyr,
         const float fast_thr,
         const unsigned max_feat,
         const float scl_fctr,
         const unsigned levels,
         const bool blur_img)
{
    unsigned patch_size = REF_PAT_SIZE;

    unsigned max_levels = feat_pyr.size();

    // In future implementations, the user will be capable of passing his
    // distribution instead of using the reference one
    //CUDA_CHECK(cudaMemcpyToSymbol(d_ref_pat, h_ref_pat, 256 * 4 * sizeof(int), 0, cudaMemcpyHostToDevice));

    vector<float*> d_score_pyr(max_levels);
    vector<float*> d_ori_pyr(max_levels);
    vector<float*> d_size_pyr(max_levels);
    vector<unsigned*> d_desc_pyr(max_levels);
    vector<unsigned*> d_idx_pyr(max_levels);

    unsigned total_feat = 0;

    // Calculate a separable Gaussian kernel
    Param<convAccT> gauss_filter;
    if (blur_img) {
        unsigned gauss_len = 9;
        scoped_ptr<convAccT> h_gauss(new convAccT[gauss_len]);
        gaussian1D(h_gauss.get(), gauss_len, 2.f);
        gauss_filter.dims[0] = gauss_len;
        gauss_filter.strides[0] = 1;

        for (int k = 1; k < 4; k++) {
            gauss_filter.dims[k] = 1;
            gauss_filter.strides[k] = gauss_filter.dims[k - 1] * gauss_filter.strides[k - 1];
        }

        int gauss_elem = gauss_filter.strides[3] * gauss_filter.dims[3];
        gauss_filter.ptr = memAlloc<convAccT>(gauss_elem);
        CUDA_CHECK(cudaMemcpy(gauss_filter.ptr, h_gauss.get(), gauss_elem * sizeof(convAccT), cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < (int)max_levels; i++) {
        if (feat_pyr[i] == 0 || lvl_best[i] == 0) {
            if (i > 0)
                memFree((T*)img_pyr[i].ptr);
            continue;
        }

        float* d_score_harris = memAlloc<float>(feat_pyr[i]);

        // Calculate Harris responses
        // Good block_size >= 7 (must be an odd number)
        dim3 threads(THREADS_X, THREADS_Y);
        dim3 blocks(divup(feat_pyr[i], threads.x), 1);
        CUDA_LAUNCH((harris_response<T,false>), blocks, threads,
               d_score_harris, NULL, d_x_pyr[i], d_y_pyr[i], NULL, feat_pyr[i], img_pyr[i], 7, 0.04f, patch_size);
        POST_LAUNCH_CHECK();

        Param<float> harris_sorted;
        Param<unsigned> harris_idx;

        harris_sorted.dims[0] = harris_idx.dims[0] = feat_pyr[i];
        harris_sorted.strides[0] = harris_idx.strides[0] = 1;

        for (int k = 1; k < 4; k++) {
            harris_sorted.dims[k] = 1;
            harris_sorted.strides[k] = harris_sorted.dims[k - 1] * harris_sorted.strides[k - 1];
            harris_idx.dims[k] = 1;
            harris_idx.strides[k] = harris_idx.dims[k - 1] * harris_idx.strides[k - 1];
        }

        int sort_elem = harris_sorted.strides[3] * harris_sorted.dims[3];
        harris_sorted.ptr = d_score_harris;
        harris_idx.ptr = memAlloc<unsigned>(sort_elem);

        // Sort features according to Harris responses
        sort0_index<float, false>(harris_sorted, harris_idx);

        feat_pyr[i] = std::min(feat_pyr[i], lvl_best[i]);

        float* d_x_lvl = memAlloc<float>(feat_pyr[i]);
        float* d_y_lvl = memAlloc<float>(feat_pyr[i]);
        float* d_score_lvl = memAlloc<float>(feat_pyr[i]);

        // Keep only features with higher Harris responses
        threads = dim3(THREADS, 1);
        blocks = dim3(divup(feat_pyr[i], threads.x), 1);
        CUDA_LAUNCH((keep_features<T>), blocks, threads,
                d_x_lvl, d_y_lvl, d_score_lvl, NULL,
                d_x_pyr[i], d_y_pyr[i], harris_sorted.ptr, harris_idx.ptr, NULL, feat_pyr[i]);
        POST_LAUNCH_CHECK();

        memFree(d_x_pyr[i]);
        memFree(d_y_pyr[i]);
        memFree(harris_sorted.ptr);
        memFree(harris_idx.ptr);

        float* d_ori_lvl = memAlloc<float>(feat_pyr[i]);

        // Compute orientation of features
        threads = dim3(THREADS_X, THREADS_Y);
        blocks  = dim3(divup(feat_pyr[i], threads.x), 1);
        CUDA_LAUNCH((centroid_angle<T>), blocks, threads,
                d_x_lvl, d_y_lvl, d_ori_lvl, feat_pyr[i], img_pyr[i], patch_size);
        POST_LAUNCH_CHECK();

        Param<T> lvl_tmp;
        Param<T> lvl_filt;

        if (blur_img) {
            for (int k = 0; k < 4; k++) {
                lvl_tmp.dims[k] = img_pyr[i].dims[k];
                lvl_tmp.strides[k] = img_pyr[i].strides[k];
                lvl_filt.dims[k] = img_pyr[i].dims[k];
                lvl_filt.strides[k] = img_pyr[i].strides[k];
            }

            int lvl_elem = img_pyr[i].strides[3] * img_pyr[i].dims[3];
            lvl_tmp.ptr = memAlloc<T>(lvl_elem);
            lvl_filt.ptr = memAlloc<T>(lvl_elem);

            // Separable Gaussian filtering to reduce noise sensitivity
            convolve2<T, convAccT, 0, false>(lvl_tmp, img_pyr[i], gauss_filter);
            convolve2<T, convAccT, 1, false>(lvl_filt, CParam<T>(lvl_tmp), gauss_filter);

            memFree(lvl_tmp.ptr);
            if (i > 0)
                memFree((T*)img_pyr[i].ptr);

            img_pyr[i].ptr = lvl_filt.ptr;
            for (int k = 0; k < 4; k++) {
                img_pyr[i].dims[k] = lvl_filt.dims[k];
                img_pyr[i].strides[k] = lvl_filt.strides[k];
            }
        }

        float* d_size_lvl = memAlloc<float>(feat_pyr[i]);

        unsigned* d_desc_lvl = memAlloc<unsigned>(feat_pyr[i] * 8);
        CUDA_CHECK(cudaMemsetAsync(d_desc_lvl, 0, feat_pyr[i] * 8 * sizeof(unsigned),
                    cuda::getStream(cuda::getActiveDeviceId())));

        // Compute ORB descriptors
        threads = dim3(THREADS_X, THREADS_Y);
        blocks  = dim3(divup(feat_pyr[i], threads.x), 1);
        CUDA_LAUNCH((extract_orb<T>), blocks, threads,
                d_desc_lvl, feat_pyr[i], d_x_lvl, d_y_lvl, d_ori_lvl, d_size_lvl,
                img_pyr[i], lvl_scl[i], patch_size);
        POST_LAUNCH_CHECK();

        if (i > 0)
            memFree((T*)img_pyr[i].ptr);

        // Store results to pyramids
        total_feat += feat_pyr[i];
        d_x_pyr[i] = d_x_lvl;
        d_y_pyr[i] = d_y_lvl;
        d_score_pyr[i] = d_score_lvl;
        d_ori_pyr[i] = d_ori_lvl;
        d_size_pyr[i] = d_size_lvl;
        d_desc_pyr[i] = d_desc_lvl;
    }

    if (blur_img)
        memFree((T*)gauss_filter.ptr);

    // If no features are found, set found features to 0 and return
    if (total_feat == 0) {
        *out_feat = 0;
        return;
    }

    // Allocate output memory
    *d_x     = memAlloc<float>(total_feat);
    *d_y     = memAlloc<float>(total_feat);
    *d_score = memAlloc<float>(total_feat);
    *d_ori   = memAlloc<float>(total_feat);
    *d_size  = memAlloc<float>(total_feat);
    *d_desc  = memAlloc<unsigned>(total_feat * 8);
    unsigned offset = 0;
    for (unsigned i = 0; i < max_levels; i++) {
        if (feat_pyr[i] == 0)
            continue;

        if (i > 0)
            offset += feat_pyr[i-1];

        CUDA_CHECK(cudaMemcpyAsync(*d_x+offset, d_x_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getStream(cuda::getActiveDeviceId())));
        CUDA_CHECK(cudaMemcpyAsync(*d_y+offset, d_y_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getStream(cuda::getActiveDeviceId())));
        CUDA_CHECK(cudaMemcpyAsync(*d_score+offset, d_score_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getStream(cuda::getActiveDeviceId())));
        CUDA_CHECK(cudaMemcpyAsync(*d_ori+offset, d_ori_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getStream(cuda::getActiveDeviceId())));
        CUDA_CHECK(cudaMemcpyAsync(*d_size+offset, d_size_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getStream(cuda::getActiveDeviceId())));
        CUDA_CHECK(cudaMemcpyAsync(*d_desc+(offset*8), d_desc_pyr[i], feat_pyr[i] * 8 * sizeof(unsigned),
                    cudaMemcpyDeviceToDevice, cuda::getStream(cuda::getActiveDeviceId())));

        memFree(d_x_pyr[i]);
        memFree(d_y_pyr[i]);
        memFree(d_score_pyr[i]);
        memFree(d_ori_pyr[i]);
        memFree(d_size_pyr[i]);
        memFree(d_desc_pyr[i]);
    }

    // Sets number of output features
    *out_feat = total_feat;
}

} // namespace kernel

} // namespace cuda
