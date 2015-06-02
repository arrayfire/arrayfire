/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <program.hpp>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <convolve_common.hpp>
#include <kernel/convolve_separable.hpp>
#include <kernel/fast.hpp>
#include <kernel/resize.hpp>
#include <kernel/sort_index.hpp>
#include <kernel_headers/orb.hpp>
#include <memory.hpp>
#include <vector>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::EnqueueArgs;
using cl::LocalSpaceArg;
using cl::NDRange;
using std::vector;

namespace opencl
{

namespace kernel
{

static const int ORB_THREADS   = 256;
static const int ORB_THREADS_X = 16;
static const int ORB_THREADS_Y = 16;

static const float PI_VAL = 3.14159265358979323846f;

// Reference pattern, generated for a patch size of 31x31, as suggested by
// original ORB paper
#define REF_PAT_SIZE 31
#define REF_PAT_SAMPLES 256
#define REF_PAT_COORDS 4
#define REF_PAT_LENGTH (REF_PAT_SAMPLES*REF_PAT_COORDS)


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

template<typename T, typename convAccT>
void orb(unsigned* out_feat,
         Param& x_out,
         Param& y_out,
         Param& score_out,
         Param& ori_out,
         Param& size_out,
         Param& desc_out,
         Param image,
         const float fast_thr,
         const unsigned max_feat,
         const float scl_fctr,
         const unsigned levels,
         const bool blur_img)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> orbProgs;
        static std::map<int, Kernel*>  hrKernel;
        static std::map<int, Kernel*>  kfKernel;
        static std::map<int, Kernel*>  caKernel;
        static std::map<int, Kernel*>  eoKernel;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D BLOCK_SIZE=" << ORB_THREADS_X;

                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                cl::Program prog;
                buildProgram(prog, orb_cl, orb_cl_len, options.str());
                orbProgs[device] = new Program(prog);

                hrKernel[device] = new Kernel(*orbProgs[device], "harris_response");
                kfKernel[device] = new Kernel(*orbProgs[device], "keep_features");
                caKernel[device] = new Kernel(*orbProgs[device], "centroid_angle");
                eoKernel[device] = new Kernel(*orbProgs[device], "extract_orb");
            });

        unsigned patch_size = REF_PAT_SIZE;

        unsigned min_side = std::min(image.info.dims[0], image.info.dims[1]);
        unsigned max_levels = 0;
        float scl_sum = 0.f;
        for (unsigned i = 0; i < levels; i++) {
            min_side /= scl_fctr;

            // Minimum image side for a descriptor to be computed
            if (min_side < patch_size || max_levels == levels) break;

            max_levels++;
            scl_sum += 1.f / (float)pow(scl_fctr,(float)i);
        }

        vector<cl::Buffer*> d_x_pyr(max_levels);
        vector<cl::Buffer*> d_y_pyr(max_levels);
        vector<cl::Buffer*> d_score_pyr(max_levels);
        vector<cl::Buffer*> d_ori_pyr(max_levels);
        vector<cl::Buffer*> d_size_pyr(max_levels);
        vector<cl::Buffer*> d_desc_pyr(max_levels);

        vector<unsigned> feat_pyr(max_levels);
        unsigned total_feat = 0;

        // Compute number of features to keep for each level
        vector<unsigned> lvl_best(max_levels);
        unsigned feat_sum = 0;
        for (unsigned i = 0; i < max_levels-1; i++) {
            float lvl_scl = (float)pow(scl_fctr,(float)i);
            lvl_best[i] = ceil((max_feat / scl_sum) / lvl_scl);
            feat_sum += lvl_best[i];
        }
        lvl_best[max_levels-1] = max_feat - feat_sum;

        // Maintain a reference to previous level image
        Param prev_img;
        Param lvl_img;

        const unsigned gauss_len = 9;
        T* h_gauss = nullptr;
        Param gauss_filter;
        gauss_filter.data = nullptr;

        for (unsigned i = 0; i < max_levels; i++) {
            const float lvl_scl = (float)pow(scl_fctr,(float)i);

            if (i == 0) {
                // First level is used in its original size
                lvl_img = image;

                prev_img = image;
            }
            else if (i > 0) {
                // Resize previous level image to current level dimensions
                lvl_img.info.dims[0] = round(image.info.dims[0] / lvl_scl);
                lvl_img.info.dims[1] = round(image.info.dims[1] / lvl_scl);

                lvl_img.info.strides[0] = 1;
                lvl_img.info.strides[1] = lvl_img.info.dims[0];

                for (int k = 2; k < 4; k++) {
                    lvl_img.info.dims[k] = 1;
                    lvl_img.info.strides[k] = lvl_img.info.dims[k - 1] * lvl_img.info.strides[k - 1];
                }

                lvl_img.info.offset = 0;
                lvl_img.data = bufferAlloc(lvl_img.info.dims[3] * lvl_img.info.strides[3] * sizeof(T));

                resize<T, AF_INTERP_BILINEAR>(lvl_img, prev_img);

                if (i > 1)
                   bufferFree(prev_img.data);
                prev_img = lvl_img;
            }

            unsigned lvl_feat = 0;
            Param d_x_feat, d_y_feat, d_score_feat;

            // Round feature size to nearest odd integer
            float size = 2.f * floor(patch_size / 2.f) + 1.f;

            // Avoid keeping features that might be too wide and might not fit on
            // the image, sqrt(2.f) is the radius when angle is 45 degrees and
            // represents widest case possible
            unsigned edge = ceil(size * sqrt(2.f) / 2.f);

            // Detect FAST features
            fast<T, 9, true>(&lvl_feat, d_x_feat, d_y_feat, d_score_feat,
                             lvl_img, fast_thr, 0.15f, edge);

            if (lvl_feat == 0) {
                feat_pyr[i] = 0;

                if (i > 0 && i == max_levels-1)
                    bufferFree(lvl_img.data);

                continue;
            }

            bufferFree(d_score_feat.data);

            unsigned usable_feat = 0;
            cl::Buffer* d_usable_feat = bufferAlloc(sizeof(unsigned));
            getQueue().enqueueWriteBuffer(*d_usable_feat, CL_TRUE, 0, sizeof(unsigned), &usable_feat);

            cl::Buffer* d_x_harris = bufferAlloc(lvl_feat * sizeof(float));
            cl::Buffer* d_y_harris = bufferAlloc(lvl_feat * sizeof(float));
            cl::Buffer* d_score_harris = bufferAlloc(lvl_feat * sizeof(float));

            // Calculate Harris responses
            // Good block_size >= 7 (must be an odd number)
            const int blk_x = divup(lvl_feat, ORB_THREADS_X);
            const NDRange local(ORB_THREADS_X, ORB_THREADS_Y);
            const NDRange global(blk_x * ORB_THREADS_X, ORB_THREADS_Y);

            unsigned block_size = 7;
            float k_thr = 0.04f;

            auto hrOp = make_kernel<Buffer, Buffer, Buffer,
                                    Buffer, Buffer, const unsigned,
                                    Buffer, Buffer, KParam,
                                    const unsigned, const float, const unsigned> (*hrKernel[device]);

            hrOp(EnqueueArgs(getQueue(), global, local),
                 *d_x_harris, *d_y_harris, *d_score_harris,
                 *d_x_feat.data, *d_y_feat.data, lvl_feat,
                 *d_usable_feat, *lvl_img.data, lvl_img.info,
                 block_size, k_thr, patch_size);
            CL_DEBUG_FINISH(getQueue());

            getQueue().enqueueReadBuffer(*d_usable_feat, CL_TRUE, 0, sizeof(unsigned), &usable_feat);

            if (lvl_feat > 0) { //This is just to supress warnings
                bufferFree(d_x_feat.data);
                bufferFree(d_y_feat.data);
                bufferFree(d_usable_feat);
            }

            if (usable_feat == 0) {
                feat_pyr[i] = 0;

                bufferFree(d_x_harris);
                bufferFree(d_y_harris);
                bufferFree(d_score_harris);

                if (i > 0 && i == max_levels-1)
                    bufferFree(lvl_img.data);

                continue;
            }

            // Sort features according to Harris responses
            Param d_harris_sorted;
            Param d_harris_idx;

            d_harris_sorted.info.dims[0] = usable_feat;
            d_harris_idx.info.dims[0] = usable_feat;
            d_harris_sorted.info.strides[0] = 1;
            d_harris_idx.info.strides[0] = 1;

            for (int k = 1; k < 4; k++) {
                d_harris_sorted.info.dims[k] = 1;
                d_harris_idx.info.dims[k] = 1;
                d_harris_sorted.info.strides[k] = d_harris_sorted.info.dims[k - 1] * d_harris_sorted.info.strides[k - 1];
                d_harris_idx.info.strides[k] = d_harris_idx.info.dims[k - 1] * d_harris_idx.info.strides[k - 1];
            }

            d_harris_sorted.info.offset = 0;
            d_harris_idx.info.offset = 0;
            d_harris_sorted.data = d_score_harris;
            d_harris_idx.data = bufferAlloc((d_harris_idx.info.dims[0]) * sizeof(unsigned));

            sort0_index<float, false>(d_harris_sorted, d_harris_idx);

            cl::Buffer* d_x_lvl = bufferAlloc(usable_feat * sizeof(float));
            cl::Buffer* d_y_lvl = bufferAlloc(usable_feat * sizeof(float));
            cl::Buffer* d_score_lvl = bufferAlloc(usable_feat * sizeof(float));

            usable_feat = min(usable_feat, lvl_best[i]);

            // Keep only features with higher Harris responses
            const int keep_blk = divup(usable_feat, ORB_THREADS);
            const NDRange local_keep(ORB_THREADS, 1);
            const NDRange global_keep(keep_blk * ORB_THREADS, 1);

            auto kfOp = make_kernel<Buffer, Buffer, Buffer,
                                    Buffer, Buffer, Buffer, Buffer,
                                    const unsigned> (*kfKernel[device]);

            kfOp(EnqueueArgs(getQueue(), global_keep, local_keep),
                 *d_x_lvl, *d_y_lvl, *d_score_lvl,
                 *d_x_harris, *d_y_harris, *d_harris_sorted.data, *d_harris_idx.data,
                 usable_feat);
            CL_DEBUG_FINISH(getQueue());

            bufferFree(d_x_harris);
            bufferFree(d_y_harris);
            bufferFree(d_harris_sorted.data);
            bufferFree(d_harris_idx.data);

            cl::Buffer* d_ori_lvl = bufferAlloc(usable_feat * sizeof(float));
            cl::Buffer* d_size_lvl = bufferAlloc(usable_feat * sizeof(float));

            // Compute orientation of features
            const int centroid_blk_x = divup(usable_feat, ORB_THREADS_X);
            const NDRange local_centroid(ORB_THREADS_X, ORB_THREADS_Y);
            const NDRange global_centroid(centroid_blk_x * ORB_THREADS_X, ORB_THREADS_Y);

            auto caOp = make_kernel<Buffer, Buffer, Buffer,
                                    const unsigned, Buffer, KParam,
                                    const unsigned> (*caKernel[device]);

            caOp(EnqueueArgs(getQueue(), global_centroid, local_centroid),
                 *d_x_lvl, *d_y_lvl, *d_ori_lvl,
                 usable_feat, *lvl_img.data, lvl_img.info,
                 patch_size);
            CL_DEBUG_FINISH(getQueue());

            Param lvl_filt;
            Param lvl_tmp;

            if (blur_img) {
                lvl_filt = lvl_img;
                lvl_tmp = lvl_img;

                lvl_filt.data = bufferAlloc(lvl_filt.info.dims[0] * lvl_filt.info.dims[1] * sizeof(T));
                lvl_tmp.data = bufferAlloc(lvl_tmp.info.dims[0] * lvl_tmp.info.dims[1] * sizeof(T));

                // Calculate a separable Gaussian kernel
                if (h_gauss == nullptr) {
                    h_gauss = new T[gauss_len];
                    gaussian1D(h_gauss, gauss_len, 2.f);
                    gauss_filter.info.dims[0] = gauss_len;
                    gauss_filter.info.strides[0] = 1;

                    for (int k = 1; k < 4; k++) {
                        gauss_filter.info.dims[k] = 1;
                        gauss_filter.info.strides[k] = gauss_filter.info.dims[k - 1] * gauss_filter.info.strides[k - 1];
                    }

                    int gauss_elem = gauss_filter.info.strides[3] * gauss_filter.info.dims[3];
                    gauss_filter.data = bufferAlloc(gauss_elem * sizeof(T));
                    getQueue().enqueueWriteBuffer(*gauss_filter.data, CL_TRUE, 0, gauss_elem * sizeof(T), h_gauss);
                }

                // Filter level image with Gaussian kernel to reduce noise sensitivity
                convolve2<T, convAccT, 0, false, gauss_len>(lvl_tmp, lvl_img, gauss_filter);
                convolve2<T, convAccT, 1, false, gauss_len>(lvl_filt, lvl_tmp, gauss_filter);

                bufferFree(lvl_tmp.data);
            }

            // Compute ORB descriptors
            cl::Buffer* d_desc_lvl = bufferAlloc(usable_feat * 8 * sizeof(unsigned));
            {
                vector<unsigned> h_desc_lvl(usable_feat * 8);
                getQueue().enqueueWriteBuffer(*d_desc_lvl, CL_TRUE, 0, usable_feat * 8 * sizeof(unsigned), h_desc_lvl.data());
            }

            auto eoOp = make_kernel<Buffer, const unsigned,
                                    Buffer, Buffer, Buffer, Buffer,
                                    Buffer, KParam,
                                    const float, const unsigned> (*eoKernel[device]);

            if (blur_img) {
                eoOp(EnqueueArgs(getQueue(), global_centroid, local_centroid),
                     *d_desc_lvl, usable_feat,
                     *d_x_lvl, *d_y_lvl, *d_ori_lvl, *d_size_lvl,
                     *lvl_filt.data, lvl_filt.info,
                     lvl_scl, patch_size);
                CL_DEBUG_FINISH(getQueue());

                bufferFree(lvl_filt.data);
            }
            else {
                eoOp(EnqueueArgs(getQueue(), global_centroid, local_centroid),
                     *d_desc_lvl, usable_feat,
                     *d_x_lvl, *d_y_lvl, *d_ori_lvl, *d_size_lvl,
                     *lvl_img.data, lvl_img.info,
                     lvl_scl, patch_size);
                CL_DEBUG_FINISH(getQueue());
            }

            // Store results to pyramids
            total_feat += usable_feat;
            feat_pyr[i] = usable_feat;
            d_x_pyr[i] = d_x_lvl;
            d_y_pyr[i] = d_y_lvl;
            d_score_pyr[i] = d_score_lvl;
            d_ori_pyr[i] = d_ori_lvl;
            d_size_pyr[i] = d_size_lvl;
            d_desc_pyr[i] = d_desc_lvl;

            if (i > 0 && i == max_levels-1)
                bufferFree(lvl_img.data);
        }

        if (gauss_filter.data != nullptr)
            bufferFree(gauss_filter.data);
        if (h_gauss != nullptr)
            delete[] h_gauss;

        // If no features are found, set found features to 0 and return
        if (total_feat == 0) {
            *out_feat = 0;
            return;
        }

        // Allocate output memory
        x_out.info.dims[0] = total_feat;
        x_out.info.strides[0] = 1;
        y_out.info.dims[0] = total_feat;
        y_out.info.strides[0] = 1;
        score_out.info.dims[0] = total_feat;
        score_out.info.strides[0] = 1;
        ori_out.info.dims[0] = total_feat;
        ori_out.info.strides[0] = 1;
        size_out.info.dims[0] = total_feat;
        size_out.info.strides[0] = 1;

        desc_out.info.dims[0] = 8;
        desc_out.info.strides[0] = 1;
        desc_out.info.dims[1] = total_feat;
        desc_out.info.strides[1] = desc_out.info.dims[0];

        for (int k = 1; k < 4; k++) {
            x_out.info.dims[k] = 1;
            x_out.info.strides[k] = x_out.info.dims[k - 1] * x_out.info.strides[k - 1];
            y_out.info.dims[k] = 1;
            y_out.info.strides[k] = y_out.info.dims[k - 1] * y_out.info.strides[k - 1];
            score_out.info.dims[k] = 1;
            score_out.info.strides[k] = score_out.info.dims[k - 1] * score_out.info.strides[k - 1];
            ori_out.info.dims[k] = 1;
            ori_out.info.strides[k] = ori_out.info.dims[k - 1] * ori_out.info.strides[k - 1];
            size_out.info.dims[k] = 1;
            size_out.info.strides[k] = size_out.info.dims[k - 1] * size_out.info.strides[k - 1];
            if (k > 1) {
                desc_out.info.dims[k] = 1;
                desc_out.info.strides[k] = desc_out.info.dims[k - 1] * desc_out.info.strides[k - 1];
            }
        }

        if (total_feat > 0) {
            size_t out_sz  = total_feat * sizeof(float);
            x_out.data     = bufferAlloc(out_sz);
            y_out.data     = bufferAlloc(out_sz);
            score_out.data = bufferAlloc(out_sz);
            ori_out.data   = bufferAlloc(out_sz);
            size_out.data  = bufferAlloc(out_sz);

            size_t desc_sz = total_feat * 8 * sizeof(unsigned);
            desc_out.data  = bufferAlloc(desc_sz);
        }

        unsigned offset = 0;
        for (unsigned i = 0; i < max_levels; i++) {
            if (feat_pyr[i] == 0)
                continue;

            if (i > 0)
                offset += feat_pyr[i-1];

            getQueue().enqueueCopyBuffer(*d_x_pyr[i], *x_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));
            getQueue().enqueueCopyBuffer(*d_y_pyr[i], *y_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));
            getQueue().enqueueCopyBuffer(*d_score_pyr[i], *score_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));
            getQueue().enqueueCopyBuffer(*d_ori_pyr[i], *ori_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));
            getQueue().enqueueCopyBuffer(*d_size_pyr[i], *size_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));

            getQueue().enqueueCopyBuffer(*d_desc_pyr[i], *desc_out.data, 0, offset*8*sizeof(unsigned), feat_pyr[i] * 8 * sizeof(unsigned));

            bufferFree(d_x_pyr[i]);
            bufferFree(d_y_pyr[i]);
            bufferFree(d_score_pyr[i]);
            bufferFree(d_ori_pyr[i]);
            bufferFree(d_size_pyr[i]);
            bufferFree(d_desc_pyr[i]);
        }

        // Sets number of output features
        *out_feat = total_feat;
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

} //namespace kernel

} //namespace opencl
