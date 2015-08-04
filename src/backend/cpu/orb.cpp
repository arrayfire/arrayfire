/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_cpu.hpp>
#include <handle.hpp>
#include <resize.hpp>
#include <fast.hpp>
#include <sort_index.hpp>
#include <convolve.hpp>
#include <memory.hpp>
#include <cstring>

using af::dim4;

namespace cpu
{

static const float PI_VAL = 3.14159265358979323846f;

// Reference pattern, generated for a patch size of 31x31, as suggested by
// original ORB paper
#define REF_PAT_SIZE 31
#define REF_PAT_SAMPLES 256
#define REF_PAT_COORDS 4
#define REF_PAT_LENGTH (REF_PAT_SAMPLES*REF_PAT_COORDS)

// Current reference pattern was borrowed from OpenCV, to build a pattern with
// similar quality, a training process must be applied, as described in
// sections 4.2 and 4.3 of the original ORB paper.
const int ref_pat[REF_PAT_LENGTH] = {
    8,-3, 9,5,
    4,2, 7,-12,
    -11,9, -8,2,
    7,-12, 12,-13,
    2,-13, 2,12,
    1,-7, 1,6,
    -2,-10, -2,-4,
    -13,-13, -11,-8,
    -13,-3, -12,-9,
    10,4, 11,9,
    -13,-8, -8,-9,
    -11,7, -9,12,
    7,7, 12,6,
    -4,-5, -3,0,
    -13,2, -12,-3,
    -9,0, -7,5,
    12,-6, 12,-1,
    -3,6, -2,12,
    -6,-13, -4,-8,
    11,-13, 12,-8,
    4,7, 5,1,
    5,-3, 10,-3,
    3,-7, 6,12,
    -8,-7, -6,-2,
    -2,11, -1,-10,
    -13,12, -8,10,
    -7,3, -5,-3,
    -4,2, -3,7,
    -10,-12, -6,11,
    5,-12, 6,-7,
    5,-6, 7,-1,
    1,0, 4,-5,
    9,11, 11,-13,
    4,7, 4,12,
    2,-1, 4,4,
    -4,-12, -2,7,
    -8,-5, -7,-10,
    4,11, 9,12,
    0,-8, 1,-13,
    -13,-2, -8,2,
    -3,-2, -2,3,
    -6,9, -4,-9,
    8,12, 10,7,
    0,9, 1,3,
    7,-5, 11,-10,
    -13,-6, -11,0,
    10,7, 12,1,
    -6,-3, -6,12,
    10,-9, 12,-4,
    -13,8, -8,-12,
    -13,0, -8,-4,
    3,3, 7,8,
    5,7, 10,-7,
    -1,7, 1,-12,
    3,-10, 5,6,
    2,-4, 3,-10,
    -13,0, -13,5,
    -13,-7, -12,12,
    -13,3, -11,8,
    -7,12, -4,7,
    6,-10, 12,8,
    -9,-1, -7,-6,
    -2,-5, 0,12,
    -12,5, -7,5,
    3,-10, 8,-13,
    -7,-7, -4,5,
    -3,-2, -1,-7,
    2,9, 5,-11,
    -11,-13, -5,-13,
    -1,6, 0,-1,
    5,-3, 5,2,
    -4,-13, -4,12,
    -9,-6, -9,6,
    -12,-10, -8,-4,
    10,2, 12,-3,
    7,12, 12,12,
    -7,-13, -6,5,
    -4,9, -3,4,
    7,-1, 12,2,
    -7,6, -5,1,
    -13,11, -12,5,
    -3,7, -2,-6,
    7,-8, 12,-7,
    -13,-7, -11,-12,
    1,-3, 12,12,
    2,-6, 3,0,
    -4,3, -2,-13,
    -1,-13, 1,9,
    7,1, 8,-6,
    1,-1, 3,12,
    9,1, 12,6,
    -1,-9, -1,3,
    -13,-13, -10,5,
    7,7, 10,12,
    12,-5, 12,9,
    6,3, 7,11,
    5,-13, 6,10,
    2,-12, 2,3,
    3,8, 4,-6,
    2,6, 12,-13,
    9,-12, 10,3,
    -8,4, -7,9,
    -11,12, -4,-6,
    1,12, 2,-8,
    6,-9, 7,-4,
    2,3, 3,-2,
    6,3, 11,0,
    3,-3, 8,-8,
    7,8, 9,3,
    -11,-5, -6,-4,
    -10,11, -5,10,
    -5,-8, -3,12,
    -10,5, -9,0,
    8,-1, 12,-6,
    4,-6, 6,-11,
    -10,12, -8,7,
    4,-2, 6,7,
    -2,0, -2,12,
    -5,-8, -5,2,
    7,-6, 10,12,
    -9,-13, -8,-8,
    -5,-13, -5,-2,
    8,-8, 9,-13,
    -9,-11, -9,0,
    1,-8, 1,-2,
    7,-4, 9,1,
    -2,1, -1,-4,
    11,-6, 12,-11,
    -12,-9, -6,4,
    3,7, 7,12,
    5,5, 10,8,
    0,-4, 2,8,
    -9,12, -5,-13,
    0,7, 2,12,
    -1,2, 1,7,
    5,11, 7,-9,
    3,5, 6,-8,
    -13,-4, -8,9,
    -5,9, -3,-3,
    -4,-7, -3,-12,
    6,5, 8,0,
    -7,6, -6,12,
    -13,6, -5,-2,
    1,-10, 3,10,
    4,1, 8,-4,
    -2,-2, 2,-13,
    2,-12, 12,12,
    -2,-13, 0,-6,
    4,1, 9,3,
    -6,-10, -3,-5,
    -3,-13, -1,1,
    7,5, 12,-11,
    4,-2, 5,-7,
    -13,9, -9,-5,
    7,1, 8,6,
    7,-8, 7,6,
    -7,-4, -7,1,
    -8,11, -7,-8,
    -13,6, -12,-8,
    2,4, 3,9,
    10,-5, 12,3,
    -6,-5, -6,7,
    8,-3, 9,-8,
    2,-12, 2,8,
    -11,-2, -10,3,
    -12,-13, -7,-9,
    -11,0, -10,-5,
    5,-3, 11,8,
    -2,-13, -1,12,
    -1,-8, 0,9,
    -13,-11, -12,-5,
    -10,-2, -10,11,
    -3,9, -2,-13,
    2,-3, 3,2,
    -9,-13, -4,0,
    -4,6, -3,-10,
    -4,12, -2,-7,
    -6,-11, -4,9,
    6,-3, 6,11,
    -13,11, -5,5,
    11,11, 12,6,
    7,-5, 12,-2,
    -1,12, 0,7,
    -4,-8, -3,-2,
    -7,1, -6,7,
    -13,-12, -8,-13,
    -7,-2, -6,-8,
    -8,5, -6,-9,
    -5,-1, -4,5,
    -13,7, -8,10,
    1,5, 5,-13,
    1,0, 10,-13,
    9,12, 10,-1,
    5,-8, 10,-9,
    -1,11, 1,-13,
    -9,-3, -6,2,
    -1,-10, 1,12,
    -13,1, -8,-10,
    8,-11, 10,-6,
    2,-13, 3,-6,
    7,-13, 12,-9,
    -10,-10, -5,-7,
    -10,-8, -8,-13,
    4,-6, 8,5,
    3,12, 8,-13,
    -4,2, -3,-3,
    5,-13, 10,-12,
    4,-13, 5,-1,
    -9,9, -4,3,
    0,3, 3,-9,
    -12,1, -6,1,
    3,2, 4,-8,
    -10,-10, -10,9,
    8,-13, 12,12,
    -8,-12, -6,-5,
    2,2, 3,7,
    10,6, 11,-8,
    6,8, 8,-12,
    -7,10, -6,5,
    -3,-9, -3,9,
    -1,-13, -1,5,
    -3,-7, -3,4,
    -8,-2, -8,3,
    4,2, 12,12,
    2,-5, 3,11,
    6,-9, 11,-13,
    3,-1, 7,12,
    11,-1, 12,4,
    -3,0, -3,6,
    4,-11, 4,12,
    2,-4, 2,1,
    -10,-6, -8,1,
    -13,7, -11,1,
    -13,12, -11,-13,
    6,0, 11,-13,
    0,-1, 1,4,
    -13,3, -9,-2,
    -9,8, -6,-3,
    -13,-6, -8,-2,
    5,-9, 8,10,
    2,7, 3,-9,
    -1,-6, -1,-1,
    9,5, 11,-2,
    11,-3, 12,-8,
    3,0, 3,5,
    -1,4, 0,10,
    3,-6, 4,5,
    -13,0, -10,5,
    5,8, 12,11,
    8,9, 9,-6,
    7,-4, 8,-12,
    -10,4, -10,9,
    7,3, 12,4,
    9,-7, 10,-2,
    7,0, 12,-2,
    -1,-6, 0,-11,
};

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

template<typename T>
void keep_features(
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
    // Keep only the first n_feat features
    for (unsigned f = 0; f < n_feat; f++) {
        x_out[f] = x_in[score_idx[f]];
        y_out[f] = y_in[score_idx[f]];
        score_out[f] = score_in[f];
        if (size_in != nullptr && size_out != nullptr)
            size_out[f] = size_in[score_idx[f]];
    }
}

template<typename T, bool use_scl>
void harris_response(
    float* x_out,
    float* y_out,
    float* score_out,
    float* size_out,
    const float* x_in,
    const float* y_in,
    const float* scl_in,
    const unsigned total_feat,
    unsigned* usable_feat,
    const Array<T>& image,
    const unsigned block_size,
    const float k_thr,
    const unsigned patch_size)
{
    const af::dim4 idims = image.dims();
    const T* image_ptr = image.get();
    for (unsigned f = 0; f < total_feat; f++) {
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
        float size = 2.f * floor((patch_size * scl) / 2.f) + 1.f;

        // Avoid keeping features that might be too wide and might not fit on
        // the image, sqrt(2.f) is the radius when angle is 45 degrees and
        // represents widest case possible
        unsigned patch_r = ceil(size * sqrt(2.f) / 2.f);
        if (x < patch_r || y < patch_r || x >= idims[1] - patch_r || y >= idims[0] - patch_r)
            continue;

        unsigned r = block_size / 2;

        float ixx = 0.f, iyy = 0.f, ixy = 0.f;
        unsigned block_size_sq = block_size * block_size;
        for (unsigned k = 0; k < block_size_sq; k++) {
            int i = k / block_size - r;
            int j = k % block_size - r;

            // Calculate local x and y derivatives
            float ix = image_ptr[(x+i+1) * idims[0] + y+j] - image_ptr[(x+i-1) * idims[0] + y+j];
            float iy = image_ptr[(x+i) * idims[0] + y+j+1] - image_ptr[(x+i) * idims[0] + y+j-1];

            // Accumulate second order derivatives
            ixx += ix*ix;
            iyy += iy*iy;
            ixy += ix*iy;
        }

        unsigned idx = *usable_feat;
        *usable_feat += 1;
        float tr = ixx + iyy;
        float det = ixx*iyy - ixy*ixy;

        // Calculate Harris responses
        float resp = det - k_thr * (tr*tr);

        // Scale factor
        // TODO: improve response scaling
        float rscale = 0.001f;
        rscale = rscale * rscale * rscale * rscale;

        x_out[idx] = x;
        y_out[idx] = y;
        score_out[idx] = resp * rscale;
        if (use_scl)
            size_out[idx] = size;
    }
}

template<typename T>
void centroid_angle(
    const float* x_in,
    const float* y_in,
    float* orientation_out,
    const unsigned total_feat,
    const Array<T>& image,
    const unsigned patch_size)
{
    const af::dim4 idims = image.dims();
    const T* image_ptr = image.get();
    for (unsigned f = 0; f < total_feat; f++) {
        unsigned x = (unsigned)round(x_in[f]);
        unsigned y = (unsigned)round(y_in[f]);

        unsigned r = patch_size / 2;
        if (x < r || y < r || x > idims[1] - r || y > idims[0] - r)
            continue;

        T m01 = (T)0, m10 = (T)0;
        unsigned patch_size_sq = patch_size * patch_size;
        for (unsigned k = 0; k < patch_size_sq; k++) {
            int i = k / patch_size - r;
            int j = k % patch_size - r;

            // Calculate first order moments
            T p = image_ptr[(x+i) * idims[0] + y+j];
            m01 += j * p;
            m10 += i * p;
        }

        float angle = atan2(m01, m10);
        orientation_out[f] = angle;
    }
}

template<typename T>
inline T get_pixel(
    unsigned x,
    unsigned y,
    const float ori,
    const unsigned size,
    const int dist_x,
    const int dist_y,
    const Array<T>& image,
    const unsigned patch_size)
{
    const af::dim4 idims = image.dims();
    const T* image_ptr = image.get();
    float ori_sin = sin(ori);
    float ori_cos = cos(ori);
    float patch_scl = (float)size / (float)patch_size;

    // Calculate point coordinates based on orientation and size
    x += round(dist_x * patch_scl * ori_cos - dist_y * patch_scl * ori_sin);
    y += round(dist_x * patch_scl * ori_sin + dist_y * patch_scl * ori_cos);

    return image_ptr[x * idims[0] + y];
}

template<typename T>
void extract_orb(
    unsigned* desc_out,
    const unsigned n_feat,
    float* x_in_out,
    float* y_in_out,
    const float* ori_in,
    float* size_out,
    const Array<T>& image,
    const float scl,
    const unsigned patch_size)
{
    const af::dim4 idims = image.dims();
    for (unsigned f = 0; f < n_feat; f++) {
        unsigned x = (unsigned)round(x_in_out[f]);
        unsigned y = (unsigned)round(y_in_out[f]);
        float ori = ori_in[f];
        unsigned size = patch_size;

        unsigned r = ceil(patch_size * sqrt(2.f) / 2.f);
        if (x < r || y < r || x >= idims[1] - r || y >= idims[0] - r)
            continue;

        // Descriptor fixed at 256 bits for now
        // Storing descriptor as a vector of 8 x 32-bit unsigned numbers
        for (unsigned i = 0; i < 8; i++) {
            unsigned v = 0;

            // j < 32 for 256 bits descriptor
            for (unsigned j = 0; j < 32; j++) {
                // Get position from distribution pattern and values of points p1 and p2
                int dist_x = ref_pat[i*32*4 + j*4];
                int dist_y = ref_pat[i*32*4 + j*4+1];
                T p1 = get_pixel(x, y, ori, size, dist_x, dist_y, image, patch_size);

                dist_x = ref_pat[i*32*4 + j*4+2];
                dist_y = ref_pat[i*32*4 + j*4+3];
                T p2 = get_pixel(x, y, ori, size, dist_x, dist_y, image, patch_size);

                // Calculate bit based on p1 and p2 and shifts it to correct position
                v |= (p1 < p2) << j;
            }

            // Store 32 bits of descriptor
            desc_out[f * 8 + i] += v;
        }

        x_in_out[f] = round(x * scl);
        y_in_out[f] = round(y * scl);
        size_out[f] = patch_size * scl;
    }
}



template<typename T, typename convAccT>
unsigned orb(Array<float> &x, Array<float> &y,
             Array<float> &score, Array<float> &ori,
             Array<float> &size, Array<uint> &desc,
             const Array<T>& image,
             const float fast_thr, const unsigned max_feat,
             const float scl_fctr, const unsigned levels,
             const bool blur_img)
{

    unsigned patch_size = REF_PAT_SIZE;

    const af::dim4 idims = image.dims();
    unsigned min_side = std::min(idims[0], idims[1]);
    unsigned max_levels = 0;
    float scl_sum = 0.f;

    for (unsigned i = 0; i < levels; i++) {
        min_side /= scl_fctr;

        // Minimum image side for a descriptor to be computed
        if (min_side < patch_size || max_levels == levels) break;

        max_levels++;
        scl_sum += 1.f / (float)std::pow(scl_fctr,(float)i);
    }

    std::vector<float*> h_x_pyr(max_levels);
    std::vector<float*> h_y_pyr(max_levels);
    std::vector<float*> h_score_pyr(max_levels);
    std::vector<float*> h_ori_pyr(max_levels);
    std::vector<float*> h_size_pyr(max_levels);
    std::vector<unsigned*> h_desc_pyr(max_levels);

    std::vector<unsigned> feat_pyr(max_levels);
    unsigned total_feat = 0;

    // Compute number of features to keep for each level
    std::vector<unsigned> lvl_best(max_levels);
    unsigned feat_sum = 0;
    for (unsigned i = 0; i < max_levels-1; i++) {
        float lvl_scl = (float)std::pow(scl_fctr,(float)i);
        lvl_best[i] = ceil((max_feat / scl_sum) / lvl_scl);
        feat_sum += lvl_best[i];
    }
    lvl_best[max_levels-1] = max_feat - feat_sum;

    // Maintain a reference to previous level image
    Array<T> prev_img = createEmptyArray<T>(af::dim4());
    af::dim4 prev_ldims;

    af::dim4 gauss_dims(9);
    T* h_gauss = nullptr;
    Array<T> gauss_filter = createEmptyArray<T>(af::dim4());

    for (unsigned i = 0; i < max_levels; i++) {
        af::dim4 ldims;
        const float lvl_scl = (float)std::pow(scl_fctr,(float)i);
        Array<T> lvl_img = createEmptyArray<T>(af::dim4());

        if (i == 0) {
            // First level is used in its original size
            lvl_img = image;
            ldims = image.dims();

            prev_img = image;
            prev_ldims = image.dims();
        }
        else {
            // Resize previous level image to current level dimensions
            ldims[0] = round(idims[0] / lvl_scl);
            ldims[1] = round(idims[1] / lvl_scl);

            lvl_img = resize<T>(prev_img, ldims[0], ldims[1], AF_INTERP_BILINEAR);

            prev_img = lvl_img;
            prev_ldims = lvl_img.dims();
        }


        Array<float> x_feat = createEmptyArray<float>(dim4());
        Array<float> y_feat = createEmptyArray<float>(dim4());
        Array<float> score_feat = createEmptyArray<float>(dim4());

        // Round feature size to nearest odd integer
        float size = 2.f * floor(patch_size / 2.f) + 1.f;

        // Avoid keeping features that might be too wide and might not fit on
        // the image, sqrt(2.f) is the radius when angle is 45 degrees and
        // represents widest case possible
        unsigned edge = ceil(size * sqrt(2.f) / 2.f);

        unsigned lvl_feat = fast(x_feat, y_feat, score_feat,
                                 lvl_img, fast_thr, 9, 1, 0.15f, edge);


        if (lvl_feat == 0) {
            continue;
        }

        float* h_x_feat = x_feat.get();
        float* h_y_feat = y_feat.get();

        float* h_x_harris = memAlloc<float>(lvl_feat);
        float* h_y_harris = memAlloc<float>(lvl_feat);
        float* h_score_harris = memAlloc<float>(lvl_feat);

        // Calculate Harris responses
        // Good block_size >= 7 (must be an odd number)
        unsigned usable_feat = 0;
        harris_response<T, false>(h_x_harris, h_y_harris, h_score_harris, nullptr,
                                  h_x_feat, h_y_feat, nullptr,
                                  lvl_feat, &usable_feat,
                                  lvl_img,
                                  7, 0.04f, patch_size);

        if (usable_feat == 0) {
            memFree(h_x_harris);
            memFree(h_y_harris);
            memFree(h_score_harris);

            continue;
        }

        // Sort features according to Harris responses
        af::dim4 usable_feat_dims(usable_feat);
        Array<float> score_harris = createDeviceDataArray<float>(usable_feat_dims, h_score_harris);
        Array<float> harris_sorted = createEmptyArray<float>(af::dim4());
        Array<unsigned> harris_idx = createEmptyArray<unsigned>(af::dim4());

        sort_index<float, false>(harris_sorted, harris_idx, score_harris, 0);

        usable_feat = std::min(usable_feat, lvl_best[i]);

        if (usable_feat == 0) {
            memFree(h_x_harris);
            memFree(h_y_harris);

            continue;
        }

        float* h_x_lvl = memAlloc<float>(usable_feat);
        float* h_y_lvl = memAlloc<float>(usable_feat);
        float* h_score_lvl = memAlloc<float>(usable_feat);

        // Keep only features with higher Harris responses
        keep_features<T>(h_x_lvl, h_y_lvl, h_score_lvl, nullptr,
                         h_x_harris, h_y_harris, harris_sorted.get(), harris_idx.get(),
                         nullptr, usable_feat);

        memFree(h_x_harris);
        memFree(h_y_harris);

        float* h_ori_lvl = memAlloc<float>(usable_feat);
        float* h_size_lvl = memAlloc<float>(usable_feat);

        // Compute orientation of features
        centroid_angle<T>(h_x_lvl, h_y_lvl, h_ori_lvl, usable_feat,
                          lvl_img, patch_size);

        Array<T> lvl_filt = createEmptyArray<T>(dim4());

        if (blur_img) {
            // Calculate a separable Gaussian kernel, if one is not already stored
            if (!h_gauss) {
                h_gauss = memAlloc<T>(gauss_dims[0]);
                gaussian1D(h_gauss, gauss_dims[0], 2.f);
                gauss_filter = createDeviceDataArray<T>(gauss_dims, h_gauss);
            }

            // Filter level image with Gaussian kernel to reduce noise sensitivity
            lvl_filt = convolve2<T, convAccT, false>(lvl_img, gauss_filter, gauss_filter);
        }

        // Compute ORB descriptors
        unsigned* h_desc_lvl = memAlloc<unsigned>(usable_feat * 8);
        memset(h_desc_lvl, 0, usable_feat * 8 * sizeof(unsigned));
        if (blur_img)
            extract_orb<T>(h_desc_lvl, usable_feat,
                           h_x_lvl, h_y_lvl, h_ori_lvl, h_size_lvl,
                           lvl_filt, lvl_scl, patch_size);
        else
            extract_orb<T>(h_desc_lvl, usable_feat,
                           h_x_lvl, h_y_lvl, h_ori_lvl, h_size_lvl,
                           lvl_img, lvl_scl, patch_size);

        // Store results to pyramids
        total_feat += usable_feat;
        feat_pyr[i] = usable_feat;
        h_x_pyr[i] = h_x_lvl;
        h_y_pyr[i] = h_y_lvl;
        h_score_pyr[i] = h_score_lvl;
        h_ori_pyr[i] = h_ori_lvl;
        h_size_pyr[i] = h_size_lvl;
        h_desc_pyr[i] = h_desc_lvl;

    }

    if (total_feat > 0 ) {

        // Allocate feature Arrays
        const af::dim4 total_feat_dims(total_feat);
        const af::dim4 desc_dims(8, total_feat);

        x     = createEmptyArray<float>(total_feat_dims);
        y     = createEmptyArray<float>(total_feat_dims);
        score = createEmptyArray<float>(total_feat_dims);
        ori   = createEmptyArray<float>(total_feat_dims);
        size  = createEmptyArray<float>(total_feat_dims);
        desc  = createEmptyArray<uint >(desc_dims);

        float* h_x = x.get();
        float* h_y = y.get();
        float* h_score = score.get();
        float* h_ori = ori.get();
        float* h_size = size.get();

        unsigned* h_desc = desc.get();

        unsigned offset = 0;
        for (unsigned i = 0; i < max_levels; i++) {
            if (feat_pyr[i] == 0)
                continue;

            if (i > 0)
                offset += feat_pyr[i-1];

            memcpy(h_x+offset, h_x_pyr[i], feat_pyr[i] * sizeof(float));
            memcpy(h_y+offset, h_y_pyr[i], feat_pyr[i] * sizeof(float));
            memcpy(h_score+offset, h_score_pyr[i], feat_pyr[i] * sizeof(float));
            memcpy(h_ori+offset, h_ori_pyr[i], feat_pyr[i] * sizeof(float));
            memcpy(h_size+offset, h_size_pyr[i], feat_pyr[i] * sizeof(float));

            memcpy(h_desc+(offset*8), h_desc_pyr[i], feat_pyr[i] * 8 * sizeof(unsigned));

            memFree(h_x_pyr[i]);
            memFree(h_y_pyr[i]);
            memFree(h_score_pyr[i]);
            memFree(h_ori_pyr[i]);
            memFree(h_size_pyr[i]);
            memFree(h_desc_pyr[i]);
        }
    }

    return total_feat;
}

#define INSTANTIATE(T, convAccT)                                                        \
    template unsigned orb<T, convAccT>(Array<float> &x, Array<float> &y,                \
                                       Array<float> &score, Array<float> &ori,          \
                                       Array<float> &size, Array<uint> &desc,           \
                                       const Array<T>& image,                           \
                                       const float fast_thr, const unsigned max_feat,   \
                                       const float scl_fctr, const unsigned levels,     \
                                       const bool blur_img);

INSTANTIATE(float , float )
INSTANTIATE(double, double)

}
