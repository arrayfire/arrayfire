/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/regions.hpp>
#include <program.hpp>
#include <af/defines.h>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <math.hpp>
#include <stdio.h>
#include <map>
#include <memory.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/adjacent_difference.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/iterator/counting_iterator.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/lambda/placeholders.hpp>
#include <boost/compute/lambda.hpp>

#pragma GCC diagnostic pop

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::EnqueueArgs;
using cl::NDRange;
namespace compute = boost::compute;

namespace opencl
{

namespace kernel
{

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T, bool full_conn, int n_per_thread>
void regions(Param out, Param in)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> regionsProgs;
        static std::map<int, Kernel *>     ilKernel;
        static std::map<int, Kernel *>     frKernel;
        static std::map<int, Kernel *>     ueKernel;

        int device = getActiveDeviceId();

        static const int block_dim = 16;
        static const int num_warps = 8;

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                if (full_conn) {
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D BLOCK_DIM=" << block_dim
                            << " -D NUM_WARPS=" << num_warps
                            << " -D N_PER_THREAD=" << n_per_thread
                            << " -D LIMIT_MAX=" << limit_max<T>()
                            << " -D FULL_CONN";
                }
                else {
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D BLOCK_DIM=" << block_dim
                            << " -D NUM_WARPS=" << num_warps
                            << " -D N_PER_THREAD=" << n_per_thread
                            << " -D LIMIT_MAX=" << limit_max<T>();
                }
                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                Program prog;
                buildProgram(prog, regions_cl, regions_cl_len, options.str());
                regionsProgs[device] = new Program(prog);

                ilKernel[device] = new Kernel(*regionsProgs[device], "initial_label");
                frKernel[device] = new Kernel(*regionsProgs[device], "final_relabel");
                ueKernel[device] = new Kernel(*regionsProgs[device], "update_equiv");
            });

        const NDRange local(THREADS_X, THREADS_Y);

        const int blk_x = divup(in.info.dims[0], THREADS_X*2);
        const int blk_y = divup(in.info.dims[1], THREADS_Y*2);

        const NDRange global(blk_x * THREADS_X, blk_y * THREADS_Y);

        auto ilOp = make_kernel<Buffer, KParam,
                                Buffer, KParam> (*ilKernel[device]);

        ilOp(EnqueueArgs(getQueue(), global, local),
             *out.data, out.info, *in.data, in.info);

        CL_DEBUG_FINISH(getQueue());

        int h_continue = 1;
        cl::Buffer *d_continue = bufferAlloc(sizeof(int));

        while (h_continue) {
            h_continue = 0;
            getQueue().enqueueWriteBuffer(*d_continue, CL_TRUE, 0, sizeof(int), &h_continue);

            auto ueOp = make_kernel<Buffer, KParam,
                                    Buffer> (*ueKernel[device]);

            ueOp(EnqueueArgs(getQueue(), global, local),
                 *out.data, out.info, *d_continue);
            CL_DEBUG_FINISH(getQueue());

            getQueue().enqueueReadBuffer(*d_continue, CL_TRUE, 0, sizeof(int), &h_continue);
        }

        bufferFree(d_continue);

        // Now, perform the final relabeling.  This converts the equivalency
        // map from having unique labels based on the lowest pixel in the
        // component to being sequentially numbered components starting at
        // 1.
        int size = in.info.dims[0] * in.info.dims[1];

        compute::command_queue c_queue(getQueue()());

        // Wrap raw device ptr
        compute::context context(getContext()());
        compute::vector<T> tmp(size, context);
        clEnqueueCopyBuffer(getQueue()(), (*out.data)(), tmp.get_buffer().get(), 0, 0, size * sizeof(T), 0, NULL, NULL);

        // Sort the copy
        compute::sort(tmp.begin(), tmp.end(), c_queue);

        // Take the max element, this is the number of label assignments to
        // compute.
        //int num_bins = tmp[size - 1] + 1;
        T last_label;
        clEnqueueReadBuffer(getQueue()(), tmp.get_buffer().get(), CL_TRUE, (size - 1) * sizeof(T), sizeof(T), &last_label, 0, NULL, NULL);
        int num_bins = (int)last_label + 1;

        Buffer labels(getContext(), CL_MEM_READ_WRITE, num_bins * sizeof(T));
        compute::buffer c_labels(labels());
        compute::buffer_iterator<T> labels_begin = compute::make_buffer_iterator<T>(c_labels, 0);
        compute::buffer_iterator<T> labels_end   = compute::make_buffer_iterator<T>(c_labels, num_bins);

        // Find the end of each section of values
        compute::counting_iterator<T> search_begin(0);

        int tmp_size = size;
        BOOST_COMPUTE_CLOSURE(int, upper_bound_closure, (int v), (tmp, tmp_size),
        {
            int start = 0, n = tmp_size, i;
            while(start < n)
            {
                i = (start + n) / 2;
                if(v < tmp[i])
                {
                    n = i;
                }
                else
                {
                    start = i + 1;
                }
            }

            return start;
        });

        BOOST_COMPUTE_FUNCTION(int, clamp_to_one, (int i),
        {
            return (i >= 1) ? 1 : i;
        });

        compute::transform(search_begin, search_begin + num_bins,
                           labels_begin,
                           upper_bound_closure,
                           c_queue);
        compute::adjacent_difference(labels_begin, labels_end, labels_begin, c_queue);

        // Perform the scan -- this can computes the correct labels for each
        // component
        compute::transform(labels_begin, labels_end,
                           labels_begin,
                           clamp_to_one,
                           c_queue);
        compute::exclusive_scan(labels_begin,
                                labels_end,
                                labels_begin,
                                c_queue);

        // Apply the correct labels to the equivalency map
        auto frOp = make_kernel<Buffer, KParam,
                                Buffer, KParam,
                                Buffer> (*frKernel[device]);

        //Buffer labels_buf(tmp.get_buffer().get());
        frOp(EnqueueArgs(getQueue(), global, local),
             *out.data, out.info, *in.data, in.info, labels);
        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

} //namespace kernel

} //namespace opencl
