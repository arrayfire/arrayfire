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
#include <cache.hpp>
#include <kernel_headers/fast.hpp>
#include <memory.hpp>
#include <map>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::EnqueueArgs;
using cl::LocalSpaceArg;
using cl::NDRange;

namespace opencl
{

namespace kernel
{

static const int FAST_THREADS_X = 16;
static const int FAST_THREADS_Y = 16;
static const int FAST_THREADS_NONMAX_X = 32;
static const int FAST_THREADS_NONMAX_Y = 8;

template<typename T, const bool nonmax>
void fast(const unsigned arc_length,
          unsigned* out_feat,
          Param &x_out,
          Param &y_out,
          Param &score_out,
          Param in,
          const float thr,
          const float feature_ratio,
          const unsigned edge)
{
    try {
        std::string ref_name =
            std::string("fast_") +
            std::to_string(arc_length) +
            std::string("_") +
            std::to_string(nonmax) +
            std::string("_") +
            std::string(dtype_traits<T>::getName());

        int device = getActiveDeviceId();
        kc_t::iterator cache_idx = kernelCaches[device].find(ref_name);

        kc_entry_t entry;
        if (cache_idx == kernelCaches[device].end()) {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D ARC_LENGTH=" << arc_length
                        << " -D NONMAX=" << static_cast<unsigned>(nonmax);

                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                cl::Program prog;
                buildProgram(prog, fast_cl, fast_cl_len, options.str());
                entry.prog = new Program(prog);
                entry.ker = new Kernel[3];

                entry.ker[0] = Kernel(*entry.prog, "locate_features");
                entry.ker[1] = Kernel(*entry.prog, "non_max_counts");
                entry.ker[2] = Kernel(*entry.prog, "get_features");

                kernelCaches[device][ref_name] = entry;
        } else {
            entry = cache_idx -> second;
        }

        const unsigned max_feat = ceil(in.info.dims[0] * in.info.dims[1] * feature_ratio);

        // Matrix containing scores for detected features, scores are stored in the
        // same coordinates as features, dimensions should be equal to in.
        cl::Buffer *d_score = bufferAlloc(in.info.dims[0] * in.info.dims[1] * sizeof(float));
        std::vector<float> score_init(in.info.dims[0] * in.info.dims[1], (float)0);
        getQueue().enqueueWriteBuffer(*d_score, CL_TRUE, 0, in.info.dims[0] * in.info.dims[1] * sizeof(float), &score_init[0]);

        cl::Buffer *d_flags = d_score;
        if (nonmax) {
            d_flags = bufferAlloc(in.info.dims[0] * in.info.dims[1] * sizeof(T));
        }

        const int blk_x = divup(in.info.dims[0]-edge*2, FAST_THREADS_X);
        const int blk_y = divup(in.info.dims[1]-edge*2, FAST_THREADS_Y);

        // Locate features kernel sizes
        const NDRange local(FAST_THREADS_X, FAST_THREADS_Y);
        const NDRange global(blk_x * FAST_THREADS_X, blk_y * FAST_THREADS_Y);

        auto lfOp = make_kernel<Buffer, KParam,
                                Buffer, const float, const unsigned,
                                LocalSpaceArg> (entry.ker[0]);

        lfOp(EnqueueArgs(getQueue(), global, local),
             *in.data, in.info, *d_score, thr, edge,
             cl::Local((FAST_THREADS_X + 6) * (FAST_THREADS_Y + 6) * sizeof(T)));
        CL_DEBUG_FINISH(getQueue());

        const int blk_nonmax_x = divup(in.info.dims[0], 64);
        const int blk_nonmax_y = divup(in.info.dims[1], 64);

        // Nonmax kernel sizes
        const NDRange local_nonmax(FAST_THREADS_NONMAX_X, FAST_THREADS_NONMAX_Y);
        const NDRange global_nonmax(blk_nonmax_x * FAST_THREADS_NONMAX_X, blk_nonmax_y * FAST_THREADS_NONMAX_Y);

        unsigned count_init = 0;
        cl::Buffer *d_total = bufferAlloc(sizeof(unsigned));
        getQueue().enqueueWriteBuffer(*d_total, CL_TRUE, 0, sizeof(unsigned), &count_init);

        //size_t *global_nonmax_dims = global_nonmax();
        size_t blocks_sz = blk_nonmax_x * FAST_THREADS_NONMAX_X * blk_nonmax_y * FAST_THREADS_NONMAX_Y * sizeof(unsigned);
        cl::Buffer *d_counts  = bufferAlloc(blocks_sz);
        cl::Buffer *d_offsets = bufferAlloc(blocks_sz);

        auto nmOp = make_kernel<Buffer, Buffer, Buffer,
                                Buffer, Buffer,
                                KParam, const unsigned> (entry.ker[1]);
        nmOp(EnqueueArgs(getQueue(), global_nonmax, local_nonmax),
                         *d_counts, *d_offsets, *d_total, *d_flags, *d_score, in.info, edge);
        CL_DEBUG_FINISH(getQueue());

        unsigned total;
        getQueue().enqueueReadBuffer(*d_total, CL_TRUE, 0, sizeof(unsigned), &total);
        total = total < max_feat ? total : max_feat;

        if (total > 0) {
            size_t out_sz = total * sizeof(float);
            x_out.data = bufferAlloc(out_sz);
            y_out.data = bufferAlloc(out_sz);
            score_out.data = bufferAlloc(out_sz);

            auto gfOp = make_kernel<Buffer, Buffer, Buffer,
                                    Buffer, Buffer, Buffer,
                                    KParam, const unsigned,
                                    const unsigned> (entry.ker[2]);
            gfOp(EnqueueArgs(getQueue(), global_nonmax, local_nonmax),
                             *x_out.data, *y_out.data, *score_out.data,
                             *d_flags, *d_counts, *d_offsets,
                             in.info, total, edge);
            CL_DEBUG_FINISH(getQueue());
        }

        *out_feat = total;

        x_out.info.dims[0] = total;
        x_out.info.strides[0] = 1;
        y_out.info.dims[0] = total;
        y_out.info.strides[0] = 1;
        score_out.info.dims[0] = total;
        score_out.info.strides[0] = 1;

        for (int k = 1; k < 4; k++) {
            x_out.info.dims[k] = 1;
            x_out.info.strides[k] = total;
            y_out.info.dims[k] = 1;
            y_out.info.strides[k] = total;
            score_out.info.dims[k] = 1;
            score_out.info.strides[k] = total;
        }

        bufferFree(d_score);
        if (nonmax) bufferFree(d_flags);
        bufferFree(d_total);
        bufferFree(d_counts);
        bufferFree(d_offsets);
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename T>
void fast_dispatch(const unsigned arc_length, const bool nonmax,
                   unsigned* out_feat,
                   Param &x_out,
                   Param &y_out,
                   Param &score_out,
                   Param in,
                   const float thr,
                   const float feature_ratio,
                   const unsigned edge)
{
    if (!nonmax) {
        fast<T, 0>(arc_length, out_feat, x_out, y_out, score_out, in,
                   thr, feature_ratio, edge);
    } else {
        fast<T, 1>(arc_length, out_feat, x_out, y_out, score_out, in,
                   thr, feature_ratio, edge);
    }
}

} //namespace kernel

} //namespace opencl
