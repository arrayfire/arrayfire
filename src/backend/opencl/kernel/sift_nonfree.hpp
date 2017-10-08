/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// The source code contained in this file is based on the original code by
// Rob Hess. Please note that SIFT is an algorithm patented and protected
// by US law, before using this code or any binary forms generated from it,
// verify that you have permission to do so. The original license by Rob Hess
// can be read below:
//
// Copyright (c) 2006-2012, Rob Hess <rob@iqengines.com>
// All rights reserved.
//
// The following patent has been issued for methods embodied in this
// software: "Method and apparatus for identifying scale invariant features
// in an image and use of same for locating an object in an image," David
// G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
// filed March 8, 1999. Asignee: The University of British Columbia. For
// further details, contact David Lowe (lowe@cs.ubc.ca) or the
// University-Industry Liaison Office of the University of British
// Columbia.
//
// Note that restrictions imposed by this patent (and possibly others)
// exist independently of and may be in conflict with the freedoms granted
// in this license, which refers to copyright of the program, not patents
// for any methods that it implements.  Both copyright and patent law must
// be obeyed to legally use and redistribute this program and it is not the
// purpose of this license to induce you to infringe any patents or other
// property right claims or to contest validity of any such claims.  If you
// redistribute or use the program, then this license merely protects you
// from committing copyright infringement.  It does not protect you from
// committing patent infringement.  So, before you do anything with this
// program, make sure that you have permission to do so not merely in terms
// of copyright, but also in terms of patent law.
//
// Please note that this license is not to be understood as a guarantee
// either.  If you use the program according to this license, but in
// conflict with patent law, it does not mean that the licensor will refund
// you for any losses that you incur if you are sued for your patent
// infringement.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//     * Redistributions of source code must retain the above copyright and
//       patent notices, this list of conditions and the following
//       disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the
//       distribution.
//     * Neither the name of Oregon State University nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <af/defines.h>
#include <program.hpp>
#include <common/dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <boost/compute/core.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/algorithm/gather.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>

#pragma GCC diagnostic pop

#include <kernel/convolve_separable.hpp>
#include <kernel/fast.hpp>
#include <kernel/resize.hpp>
#include <kernel_headers/sift_nonfree.hpp>
#include <memory.hpp>
#include <cache.hpp>
#include <vector>

namespace compute = boost::compute;

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
static const int SIFT_THREADS   = 256;
static const int SIFT_THREADS_X = 32;
static const int SIFT_THREADS_Y = 8;

// assumed gaussian blur for input image
static const float InitSigma = 0.5f;

// width of border in which to ignore keypoints
static const int ImgBorder = 5;

// default width of descriptor histogram array
static const int DescrWidth = 4;

// default number of bins per histogram in descriptor array
static const int DescrHistBins = 8;

// default number of bins in histogram for orientation assignment
static const int OriHistBins = 36;

// Number of GLOH bins in radial direction
static const unsigned GLOHRadialBins = 3;

// Number of GLOH angular bins (excluding the inner-most radial section)
static const unsigned GLOHAngularBins = 8;

// Number of GLOH bins per histogram in descriptor
static const unsigned GLOHHistBins = 16;

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

template<typename T>
Param gaussFilter(float sigma)
{
    // Using 6-sigma rule
    unsigned gauss_len = std::min((unsigned)round(sigma * 6 + 1) | 1, 31u);

    T* h_gauss = new T[gauss_len];
    gaussian1D(h_gauss, gauss_len, sigma);

    Param gauss_filter;
    gauss_filter.info.offset = 0;
    gauss_filter.info.dims[0] = gauss_len;
    gauss_filter.info.strides[0] = 1;

    for (int k = 1; k < 4; k++) {
        gauss_filter.info.dims[k] = 1;
        gauss_filter.info.strides[k] = gauss_filter.info.dims[k-1] * gauss_filter.info.strides[k-1];
    }

    dim_t gauss_elem = gauss_filter.info.strides[3] * gauss_filter.info.dims[3];
    gauss_filter.data = bufferAlloc(gauss_elem * sizeof(T));
    getQueue().enqueueWriteBuffer(*gauss_filter.data, CL_TRUE, 0, gauss_elem * sizeof(T), h_gauss);

    delete[] h_gauss;

    return gauss_filter;
}

template<typename T, typename convAccT>
void convSepFull(Param& dst, Param src, Param filter)
{
    Param tmp;
    tmp.info.offset = 0;
    for (int k = 0; k < 4; k++) {
        tmp.info.dims[k] = src.info.dims[k];
        tmp.info.strides[k] = src.info.strides[k];
    }

    const dim_t src_el = src.info.dims[3] * src.info.strides[3];
    tmp.data = bufferAlloc(src_el * sizeof(T));

    convSep<T, convAccT, 0, false>(tmp, src, filter);
    convSep<T, convAccT, 1, false>(dst, tmp, filter);

    bufferFree(tmp.data);
}

template<typename T, typename convAccT>
Param createInitialImage(
    Param img,
    const float init_sigma,
    const bool double_input)
{
    Param init_img;
    init_img.info.offset = 0;
    init_img.info.dims[0] = (double_input) ? img.info.dims[0] * 2 : img.info.dims[0];
    init_img.info.dims[1] = (double_input) ? img.info.dims[1] * 2 : img.info.dims[1];
    init_img.info.strides[0] = 1;
    init_img.info.strides[1] = init_img.info.dims[0];

    for (int k = 2; k < 4; k++) {
        init_img.info.dims[k] = 1;
        init_img.info.strides[k] = init_img.info.dims[k-1] * init_img.info.strides[k-1];
    }

    dim_t init_img_el = init_img.info.strides[3] * init_img.info.dims[3];
    init_img.data = bufferAlloc(init_img_el * sizeof(T));

    float s = (double_input) ? std::max((float)sqrt(init_sigma * init_sigma - InitSigma * InitSigma * 4.f), 0.1f)
                             : std::max((float)sqrt(init_sigma * init_sigma - InitSigma * InitSigma), 0.1f);

    const Param filter = gaussFilter<convAccT>(s);

    if (double_input)
        resize<T, AF_INTERP_BILINEAR>(init_img, img);

    convSepFull<T, convAccT>(init_img, (double_input) ? init_img : img, filter);

    bufferFree(filter.data);

    return init_img;
}

template<typename T, typename convAccT>
std::vector<Param> buildGaussPyr(
    Param init_img,
    const unsigned n_octaves,
    const unsigned n_layers,
    const float init_sigma)
{
    // Precompute Gaussian sigmas using the following formula:
    // \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    std::vector<float> sig_layers(n_layers + 3);
    sig_layers[0] = init_sigma;
    float k = std::pow(2.0f, 1.0f / n_layers);
    for (unsigned i = 1; i < n_layers + 3; i++) {
        float sig_prev = std::pow(k, i-1) * init_sigma;
        float sig_total = sig_prev * k;
        sig_layers[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    // Gaussian Pyramid
    std::vector<Param> gauss_pyr(n_octaves);
    std::vector<Param> tmp_pyr(n_octaves * (n_layers+3));
    for (unsigned o = 0; o < n_octaves; o++) {
        gauss_pyr[o].info.offset = 0;
        gauss_pyr[o].info.dims[0] = (o == 0) ? init_img.info.dims[0] : gauss_pyr[o-1].info.dims[0] / 2;
        gauss_pyr[o].info.dims[1] = (o == 0) ? init_img.info.dims[1] : gauss_pyr[o-1].info.dims[1] / 2;
        gauss_pyr[o].info.dims[2] = n_layers+3;
        gauss_pyr[o].info.dims[3] = 1;

        gauss_pyr[o].info.strides[0] = 1;
        gauss_pyr[o].info.strides[1] = gauss_pyr[o].info.dims[0] * gauss_pyr[o].info.strides[0];
        gauss_pyr[o].info.strides[2] = gauss_pyr[o].info.dims[1] * gauss_pyr[o].info.strides[1];
        gauss_pyr[o].info.strides[3] = gauss_pyr[o].info.dims[2] * gauss_pyr[o].info.strides[2];

        const unsigned nel = gauss_pyr[o].info.dims[3] * gauss_pyr[o].info.strides[3];
        gauss_pyr[o].data = bufferAlloc(nel * sizeof(T));

        for (unsigned l = 0; l < n_layers+3; l++) {
            unsigned src_idx = (l == 0) ? (o-1)*(n_layers+3) + n_layers : o*(n_layers+3) + l-1;
            unsigned idx = o*(n_layers+3) + l;

            tmp_pyr[o].info.offset = 0;
            if (o == 0 && l == 0) {
                for (int k = 0; k < 4; k++) {
                    tmp_pyr[idx].info.dims[k] = init_img.info.dims[k];
                    tmp_pyr[idx].info.strides[k] = init_img.info.strides[k];
                }
                tmp_pyr[idx].data = init_img.data;
            }
            else if (l == 0) {
                tmp_pyr[idx].info.dims[0] = tmp_pyr[src_idx].info.dims[0] / 2;
                tmp_pyr[idx].info.dims[1] = tmp_pyr[src_idx].info.dims[1] / 2;
                tmp_pyr[idx].info.strides[0] = 1;
                tmp_pyr[idx].info.strides[1] = tmp_pyr[idx].info.dims[0];

                for (int k = 2; k < 4; k++) {
                    tmp_pyr[idx].info.dims[k] = 1;
                    tmp_pyr[idx].info.strides[k] = tmp_pyr[idx].info.dims[k-1] * tmp_pyr[idx].info.strides[k-1];
                }

                dim_t lvl_el = tmp_pyr[idx].info.strides[3] * tmp_pyr[idx].info.dims[3];
                tmp_pyr[idx].data = bufferAlloc(lvl_el * sizeof(T));

                resize<T, AF_INTERP_BILINEAR>(tmp_pyr[idx], tmp_pyr[src_idx]);
            }
            else {
                for (int k = 0; k < 4; k++) {
                    tmp_pyr[idx].info.dims[k] = tmp_pyr[src_idx].info.dims[k];
                    tmp_pyr[idx].info.strides[k] = tmp_pyr[src_idx].info.strides[k];
                }
                dim_t lvl_el = tmp_pyr[idx].info.strides[3] * tmp_pyr[idx].info.dims[3];
                tmp_pyr[idx].data = bufferAlloc(lvl_el * sizeof(T));

                Param filter = gaussFilter<convAccT>(sig_layers[l]);

                convSepFull<T, convAccT>(tmp_pyr[idx], tmp_pyr[src_idx], filter);

                bufferFree(filter.data);
            }

            const unsigned imel = tmp_pyr[idx].info.dims[3] * tmp_pyr[idx].info.strides[3];
            const unsigned offset = imel * l;

            getQueue().enqueueCopyBuffer(*tmp_pyr[idx].data, *gauss_pyr[o].data, 0, offset*sizeof(T), imel * sizeof(T));
        }
    }

    for (unsigned o = 0; o < n_octaves; o++) {
        for (unsigned l = 0; l < n_layers+3; l++) {
            unsigned idx = o*(n_layers+3) + l;
            bufferFree(tmp_pyr[idx].data);
        }
    }

    return gauss_pyr;
}

template<typename T>
std::vector<Param> buildDoGPyr(
    std::vector<Param> gauss_pyr,
    const unsigned n_octaves,
    const unsigned n_layers,
    Kernel* suKernel)
{
    // DoG Pyramid
    std::vector<Param> dog_pyr(n_octaves);
    for (unsigned o = 0; o < n_octaves; o++) {
        for (int k = 0; k < 4; k++) {
            dog_pyr[o].info.dims[k] = (k == 2) ? gauss_pyr[o].info.dims[k]-1 : gauss_pyr[o].info.dims[k];
            dog_pyr[o].info.strides[k] = (k == 0) ? 1 : dog_pyr[o].info.dims[k-1] * dog_pyr[o].info.strides[k-1];
        }
        dog_pyr[o].info.offset = 0;

        dog_pyr[o].data = bufferAlloc(dog_pyr[o].info.dims[3] * dog_pyr[o].info.strides[3] * sizeof(T));

        const unsigned nel = dog_pyr[o].info.dims[1] * dog_pyr[o].info.strides[1];
        const unsigned dog_layers = n_layers+2;

        const int blk_x = divup(nel, SIFT_THREADS);
        const NDRange local(SIFT_THREADS, 1);
        const NDRange global(blk_x * SIFT_THREADS, 1);

        auto suOp = KernelFunctor<Buffer, Buffer, unsigned, unsigned> (*suKernel);

        suOp(EnqueueArgs(getQueue(), global, local),
             *dog_pyr[o].data, *gauss_pyr[o].data, nel, dog_layers);
        CL_DEBUG_FINISH(getQueue());
    }

    return dog_pyr;
}

template <typename T>
void update_permutation(compute::buffer_iterator<T>& keys, compute::vector<int>& permutation, compute::command_queue& queue)
{
    // temporary storage for keys
    compute::vector<T> temp(permutation.size(), 0, queue);

    // permute the keys with the current reordering
    compute::gather(permutation.begin(), permutation.end(), keys, temp.begin(), queue);

    // stable_sort the permuted keys and update the permutation
    compute::sort_by_key(temp.begin(), temp.end(), permutation.begin(), queue);
}

template <typename T>
void apply_permutation(compute::buffer_iterator<T>& keys, compute::vector<int>& permutation, compute::command_queue& queue)
{
    // copy keys to temporary vector
    compute::vector<T> temp(keys, keys+permutation.size(), queue);

    // permute the keys
    compute::gather(permutation.begin(), permutation.end(), temp.begin(), keys, queue);
}

template<typename T>
std::array<cl::Kernel*, 7> getSiftKernels()
{
    static const unsigned NUM_KERNELS = 7;
    static const char* kernelNames[NUM_KERNELS] =
        {"sub", "detectExtrema", "interpolateExtrema", "calcOrientation", "removeDuplicates",
         "computeDescriptor", "computeGLOHDescriptor"};

    kc_entry_t entries[NUM_KERNELS];

    int device = getActiveDeviceId();

    std::string checkName = kernelNames[0] + std::string("_") + std::string(dtype_traits<T>::getName());

    entries[0] = kernelCache(device, checkName);

    if (entries[0].prog==0 && entries[0].ker==0)
    {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        cl::Program prog;
        buildProgram(prog, sift_nonfree_cl, sift_nonfree_cl_len, options.str());

        for (unsigned i=0; i<NUM_KERNELS; ++i)
        {
            entries[i].prog = new Program(prog);
            entries[i].ker  = new Kernel(*entries[i].prog, kernelNames[i]);

            std::string name = kernelNames[i] + std::string("_") +
                std::string(dtype_traits<T>::getName());

            addKernelToCache(device, name, entries[i]);
        }
    } else {
        for (unsigned i=1; i<NUM_KERNELS; ++i) {
            std::string name = kernelNames[i] + std::string("_") +
                std::string(dtype_traits<T>::getName());

            entries[i] = kernelCache(device, name);
        }
    }

    std::array<cl::Kernel*, NUM_KERNELS> retVal;
    for (unsigned i=0; i<NUM_KERNELS; ++i)
        retVal[i] = entries[i].ker;

    return retVal;
}

template<typename T, typename convAccT>
void sift(unsigned* out_feat, unsigned* out_dlen, Param& x_out, Param& y_out,
          Param& score_out, Param& ori_out, Param& size_out, Param& desc_out,
          Param img, const unsigned n_layers, const float contrast_thr, const float edge_thr,
          const float init_sigma, const bool double_input, const float img_scale,
          const float feature_ratio, const bool compute_GLOH)
{
    auto kernels = getSiftKernels<T>();

    unsigned min_dim = min(img.info.dims[0], img.info.dims[1]);
    if (double_input) min_dim *= 2;

    const unsigned n_octaves = floor(log(min_dim) / log(2)) - 2;

    Param init_img = createInitialImage<T, convAccT>(img, init_sigma, double_input);

    std::vector<Param> gauss_pyr = buildGaussPyr<T, convAccT>(init_img, n_octaves, n_layers, init_sigma);

    std::vector<Param> dog_pyr = buildDoGPyr<T>(gauss_pyr, n_octaves, n_layers, kernels[0]);

    std::vector<cl::Buffer*> d_x_pyr(n_octaves, NULL);
    std::vector<cl::Buffer*> d_y_pyr(n_octaves, NULL);
    std::vector<cl::Buffer*> d_response_pyr(n_octaves, NULL);
    std::vector<cl::Buffer*> d_size_pyr(n_octaves, NULL);
    std::vector<cl::Buffer*> d_ori_pyr(n_octaves, NULL);
    std::vector<cl::Buffer*> d_desc_pyr(n_octaves, NULL);
    std::vector<unsigned> feat_pyr(n_octaves, 0);
    unsigned total_feat = 0;

    const unsigned d = DescrWidth;
    const unsigned n = DescrHistBins;
    const unsigned rb = GLOHRadialBins;
    const unsigned ab = GLOHAngularBins;
    const unsigned hb = GLOHHistBins;
    const unsigned desc_len = (compute_GLOH) ? (1 + (rb-1) * ab) * hb : d*d*n;

    cl::Buffer* d_count = bufferAlloc(sizeof(unsigned));

    for (unsigned o = 0; o < n_octaves; o++) {
        if (dog_pyr[o].info.dims[0]-2*ImgBorder < 1 ||
            dog_pyr[o].info.dims[1]-2*ImgBorder < 1)
            continue;

        const unsigned imel = dog_pyr[o].info.dims[0] * dog_pyr[o].info.dims[1];
        const unsigned max_feat = ceil(imel * feature_ratio);

        cl::Buffer* d_extrema_x     = bufferAlloc(max_feat * sizeof(float));
        cl::Buffer* d_extrema_y     = bufferAlloc(max_feat * sizeof(float));
        cl::Buffer* d_extrema_layer = bufferAlloc(max_feat * sizeof(unsigned));

        unsigned extrema_feat = 0;
        getQueue().enqueueWriteBuffer(*d_count, CL_TRUE, 0, sizeof(unsigned), &extrema_feat);

        int dim0 = dog_pyr[o].info.dims[0];
        int dim1 = dog_pyr[o].info.dims[1];

        const int blk_x = divup(dim0-2*ImgBorder, SIFT_THREADS_X);
        const int blk_y = divup(dim1-2*ImgBorder, SIFT_THREADS_Y);
        const NDRange local(SIFT_THREADS_X, SIFT_THREADS_Y);
        const NDRange global(blk_x * SIFT_THREADS_X, blk_y * SIFT_THREADS_Y);

        float extrema_thr = 0.5f * contrast_thr / n_layers;

        auto deOp = KernelFunctor<Buffer, Buffer, Buffer, Buffer,
                                Buffer, KParam, unsigned, float,
                                LocalSpaceArg> (*kernels[1]);

        deOp(EnqueueArgs(getQueue(), global, local),
              *d_extrema_x, *d_extrema_y, *d_extrema_layer, *d_count,
              *dog_pyr[o].data, dog_pyr[o].info, max_feat, extrema_thr,
              cl::Local((SIFT_THREADS_X+2) * (SIFT_THREADS_Y+2) * 3 * sizeof(float)));
        CL_DEBUG_FINISH(getQueue());

        getQueue().enqueueReadBuffer(*d_count, CL_TRUE, 0, sizeof(unsigned), &extrema_feat);
        extrema_feat = min(extrema_feat, max_feat);

        if (extrema_feat == 0) {
            bufferFree(d_extrema_x);
            bufferFree(d_extrema_y);
            bufferFree(d_extrema_layer);

            continue;
        }

        unsigned interp_feat = 0;
        getQueue().enqueueWriteBuffer(*d_count, CL_TRUE, 0, sizeof(unsigned), &interp_feat);

        cl::Buffer* d_interp_x = bufferAlloc(extrema_feat * sizeof(float));
        cl::Buffer* d_interp_y = bufferAlloc(extrema_feat * sizeof(float));
        cl::Buffer* d_interp_layer = bufferAlloc(extrema_feat * sizeof(unsigned));
        cl::Buffer* d_interp_response = bufferAlloc(extrema_feat * sizeof(float));
        cl::Buffer* d_interp_size = bufferAlloc(extrema_feat * sizeof(float));

        const int blk_x_interp = divup(extrema_feat, SIFT_THREADS);
        const NDRange local_interp(SIFT_THREADS, 1);
        const NDRange global_interp(blk_x_interp * SIFT_THREADS, 1);

        auto ieOp = KernelFunctor<Buffer, Buffer, Buffer,
                                Buffer, Buffer, Buffer,
                                Buffer, Buffer, Buffer, unsigned,
                                Buffer, KParam, unsigned, unsigned, unsigned,
                                float, float, float, float> (*kernels[2]);

        ieOp(EnqueueArgs(getQueue(), global_interp, local_interp),
              *d_interp_x, *d_interp_y, *d_interp_layer,
              *d_interp_response, *d_interp_size, *d_count,
              *d_extrema_x, *d_extrema_y, *d_extrema_layer, extrema_feat,
              *dog_pyr[o].data, dog_pyr[o].info, extrema_feat, o, n_layers,
              contrast_thr, edge_thr, init_sigma, img_scale);
        CL_DEBUG_FINISH(getQueue());

        bufferFree(d_extrema_x);
        bufferFree(d_extrema_y);
        bufferFree(d_extrema_layer);

        getQueue().enqueueReadBuffer(*d_count, CL_TRUE, 0, sizeof(unsigned), &interp_feat);
        interp_feat = min(interp_feat, extrema_feat);

        if (interp_feat == 0) {
            bufferFree(d_interp_x);
            bufferFree(d_interp_y);
            bufferFree(d_interp_layer);
            bufferFree(d_interp_response);
            bufferFree(d_interp_size);

            continue;
        }

        compute::command_queue queue(getQueue()());
        compute::context context(getContext()());

        compute::buffer buf_interp_x((*d_interp_x)(), true);
        compute::buffer buf_interp_y((*d_interp_y)(), true);
        compute::buffer buf_interp_layer((*d_interp_layer)(), true);
        compute::buffer buf_interp_response((*d_interp_response)(), true);
        compute::buffer buf_interp_size((*d_interp_size)(), true);

        compute::buffer_iterator<float> interp_x_begin = compute::make_buffer_iterator<float>(buf_interp_x, 0);
        compute::buffer_iterator<float> interp_y_begin = compute::make_buffer_iterator<float>(buf_interp_y, 0);
        compute::buffer_iterator<unsigned> interp_layer_begin = compute::make_buffer_iterator<unsigned>(buf_interp_layer, 0);
        compute::buffer_iterator<float> interp_response_begin = compute::make_buffer_iterator<float>(buf_interp_response, 0);
        compute::buffer_iterator<float> interp_size_begin = compute::make_buffer_iterator<float>(buf_interp_size, 0);

        compute::vector<int> permutation(interp_feat, context);
        compute::iota(permutation.begin(), permutation.end(), 0, queue);

        update_permutation<float>(interp_x_begin, permutation, queue);
        update_permutation<float>(interp_y_begin, permutation, queue);
        update_permutation<unsigned>(interp_layer_begin, permutation, queue);
        update_permutation<float>(interp_response_begin, permutation, queue);
        update_permutation<float>(interp_size_begin, permutation, queue);

        apply_permutation<float>(interp_x_begin, permutation, queue);
        apply_permutation<float>(interp_y_begin, permutation, queue);
        apply_permutation<unsigned>(interp_layer_begin, permutation, queue);
        apply_permutation<float>(interp_response_begin, permutation, queue);
        apply_permutation<float>(interp_size_begin, permutation, queue);

        unsigned nodup_feat = 0;
        getQueue().enqueueWriteBuffer(*d_count, CL_TRUE, 0, sizeof(unsigned), &nodup_feat);

        cl::Buffer* d_nodup_x = bufferAlloc(interp_feat * sizeof(float));
        cl::Buffer* d_nodup_y = bufferAlloc(interp_feat * sizeof(float));
        cl::Buffer* d_nodup_layer = bufferAlloc(interp_feat * sizeof(unsigned));
        cl::Buffer* d_nodup_response = bufferAlloc(interp_feat * sizeof(float));
        cl::Buffer* d_nodup_size = bufferAlloc(interp_feat * sizeof(float));

        const int blk_x_nodup = divup(extrema_feat, SIFT_THREADS);
        const NDRange local_nodup(SIFT_THREADS, 1);
        const NDRange global_nodup(blk_x_nodup * SIFT_THREADS, 1);

        auto rdOp = KernelFunctor<Buffer, Buffer, Buffer, Buffer, Buffer, Buffer,
                                Buffer, Buffer, Buffer, Buffer, Buffer,
                                unsigned> (*kernels[4]);

        rdOp(EnqueueArgs(getQueue(), global_nodup, local_nodup),
              *d_nodup_x, *d_nodup_y, *d_nodup_layer,
              *d_nodup_response, *d_nodup_size, *d_count,
              *d_interp_x, *d_interp_y, *d_interp_layer,
              *d_interp_response, *d_interp_size, interp_feat);
        CL_DEBUG_FINISH(getQueue());

        getQueue().enqueueReadBuffer(*d_count, CL_TRUE, 0, sizeof(unsigned), &nodup_feat);
        nodup_feat = min(nodup_feat, interp_feat);

        bufferFree(d_interp_x);
        bufferFree(d_interp_y);
        bufferFree(d_interp_layer);
        bufferFree(d_interp_response);
        bufferFree(d_interp_size);

        unsigned oriented_feat = 0;
        getQueue().enqueueWriteBuffer(*d_count, CL_TRUE, 0, sizeof(unsigned), &oriented_feat);
        const unsigned max_oriented_feat = nodup_feat * 3;

        cl::Buffer* d_oriented_x        = bufferAlloc(max_oriented_feat * sizeof(float));
        cl::Buffer* d_oriented_y        = bufferAlloc(max_oriented_feat * sizeof(float));
        cl::Buffer* d_oriented_layer    = bufferAlloc(max_oriented_feat * sizeof(unsigned));
        cl::Buffer* d_oriented_response = bufferAlloc(max_oriented_feat * sizeof(float));
        cl::Buffer* d_oriented_size     = bufferAlloc(max_oriented_feat * sizeof(float));
        cl::Buffer* d_oriented_ori      = bufferAlloc(max_oriented_feat * sizeof(float));

        const int blk_x_ori = divup(nodup_feat, SIFT_THREADS_Y);
        const NDRange local_ori(SIFT_THREADS_X, SIFT_THREADS_Y);
        const NDRange global_ori(SIFT_THREADS_X, blk_x_ori * SIFT_THREADS_Y);

        auto coOp = KernelFunctor<Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer,
                                Buffer, Buffer, Buffer, Buffer, Buffer, unsigned,
                                Buffer, KParam, unsigned, unsigned, int,
                                LocalSpaceArg> (*kernels[3]);

        coOp(EnqueueArgs(getQueue(), global_ori, local_ori),
              *d_oriented_x, *d_oriented_y, *d_oriented_layer,
              *d_oriented_response, *d_oriented_size, *d_oriented_ori, *d_count,
              *d_nodup_x, *d_nodup_y, *d_nodup_layer,
              *d_nodup_response, *d_nodup_size, nodup_feat,
              *gauss_pyr[o].data, gauss_pyr[o].info, max_oriented_feat, o, (int)double_input,
              cl::Local(OriHistBins * SIFT_THREADS_Y * 2 * sizeof(float)));
        CL_DEBUG_FINISH(getQueue());

        bufferFree(d_nodup_x);
        bufferFree(d_nodup_y);
        bufferFree(d_nodup_layer);
        bufferFree(d_nodup_response);
        bufferFree(d_nodup_size);

        getQueue().enqueueReadBuffer(*d_count, CL_TRUE, 0, sizeof(unsigned), &oriented_feat);
        oriented_feat = min(oriented_feat, max_oriented_feat);

        if (oriented_feat == 0) {
            bufferFree(d_oriented_x);
            bufferFree(d_oriented_y);
            bufferFree(d_oriented_layer);
            bufferFree(d_oriented_response);
            bufferFree(d_oriented_size);

            continue;
        }

        cl::Buffer* d_desc = bufferAlloc(oriented_feat * desc_len * sizeof(float));

        float scale = 1.f/(1 << o);
        if (double_input) scale *= 2.f;

        const int blk_x_desc = divup(oriented_feat, 1);
        const NDRange local_desc(SIFT_THREADS, 1);
        const NDRange global_desc(SIFT_THREADS, blk_x_desc);

        const unsigned histsz = 8;

        if (compute_GLOH) {
            auto cgOp = KernelFunctor<Buffer, unsigned, unsigned,
                                    Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, unsigned,
                                    Buffer, KParam, int, unsigned, unsigned, unsigned, float, int,
                                    LocalSpaceArg> (*kernels[6]);

            cgOp(EnqueueArgs(getQueue(), global_desc, local_desc),
                 *d_desc, desc_len, histsz,
                 *d_oriented_x, *d_oriented_y, *d_oriented_layer,
                 *d_oriented_response, *d_oriented_size, *d_oriented_ori, oriented_feat,
                 *gauss_pyr[o].data, gauss_pyr[o].info, d, rb, ab, hb, scale, n_layers,
                 cl::Local(desc_len * (histsz+1) * sizeof(float)));
        }
        else {
            auto cdOp = KernelFunctor<Buffer, unsigned, unsigned,
                                    Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, unsigned,
                                    Buffer, KParam, int, int, float, int,
                                    LocalSpaceArg> (*kernels[5]);

            cdOp(EnqueueArgs(getQueue(), global_desc, local_desc),
                  *d_desc, desc_len, histsz,
                  *d_oriented_x, *d_oriented_y, *d_oriented_layer,
                  *d_oriented_response, *d_oriented_size, *d_oriented_ori, oriented_feat,
                  *gauss_pyr[o].data, gauss_pyr[o].info, d, n, scale, n_layers,
                  cl::Local(desc_len * (histsz+1) * sizeof(float)));
        }
        CL_DEBUG_FINISH(getQueue());

        total_feat += oriented_feat;
        feat_pyr[o] = oriented_feat;

        if (oriented_feat > 0) {
            d_x_pyr[o] = d_oriented_x;
            d_y_pyr[o] = d_oriented_y;
            d_response_pyr[o] = d_oriented_response;
            d_ori_pyr[o] = d_oriented_ori;
            d_size_pyr[o] = d_oriented_size;
            d_desc_pyr[o] = d_desc;
        }
    }

    bufferFree(d_count);

    for (size_t i = 0; i < gauss_pyr.size(); i++)
        bufferFree(gauss_pyr[i].data);
    for (size_t i = 0; i < dog_pyr.size(); i++)
        bufferFree(dog_pyr[i].data);

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

    desc_out.info.dims[0] = desc_len;
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

        size_t desc_sz = total_feat * desc_len * sizeof(unsigned);
        desc_out.data  = bufferAlloc(desc_sz);
    }

    unsigned offset = 0;
    for (unsigned i = 0; i < n_octaves; i++) {
        if (feat_pyr[i] == 0)
            continue;

        getQueue().enqueueCopyBuffer(*d_x_pyr[i], *x_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_y_pyr[i], *y_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_response_pyr[i], *score_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_ori_pyr[i], *ori_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_size_pyr[i], *size_out.data, 0, offset*sizeof(float), feat_pyr[i] * sizeof(float));
        getQueue().enqueueCopyBuffer(*d_desc_pyr[i], *desc_out.data, 0, offset*desc_len*sizeof(unsigned), feat_pyr[i] * desc_len * sizeof(unsigned));

        bufferFree(d_x_pyr[i]);
        bufferFree(d_y_pyr[i]);
        bufferFree(d_response_pyr[i]);
        bufferFree(d_ori_pyr[i]);
        bufferFree(d_size_pyr[i]);
        bufferFree(d_desc_pyr[i]);

        offset += feat_pyr[i];
    }

    // Sets number of output features and descriptor length
    *out_feat = total_feat;
    *out_dlen = desc_len;
}
} //namespace kernel
} //namespace opencl
