/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/deprecated.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/regions.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <traits.hpp>
#include <af/defines.h>

AF_DEPRECATED_WARNINGS_OFF
#include <boost/compute/algorithm/adjacent_difference.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/counting_iterator.hpp>
#include <boost/compute/lambda.hpp>
#include <boost/compute/lambda/placeholders.hpp>
AF_DEPRECATED_WARNINGS_ON

#include <array>
#include <string>
#include <vector>

namespace compute = boost::compute;

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
std::array<Kernel, 3> getRegionsKernels(const bool full_conn,
                                        const int n_per_thread) {
    using std::string;
    using std::vector;

    constexpr int block_dim = 16;
    constexpr int num_warps = 8;

    ToNumStr<T> toNumStr;
    vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(full_conn),
        TemplateArg(n_per_thread),
    };
    vector<string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(BLOCK_DIM, block_dim),
        DefineKeyValue(NUM_WARPS, num_warps),
        DefineKeyValue(N_PER_THREAD, n_per_thread),
        DefineKeyValue(LIMIT_MAX, toNumStr(maxval<T>())),
    };
    if (full_conn) { options.emplace_back(DefineKey(FULL_CONN)); }
    options.emplace_back(getTypeBuildDefinition<T>());

    return {
        common::getKernel("initial_label", {{regions_cl_src}}, targs, options),
        common::getKernel("final_relabel", {{regions_cl_src}}, targs, options),
        common::getKernel("update_equiv", {{regions_cl_src}}, targs, options),
    };
}

template<typename T>
void regions(Param out, Param in, const bool full_conn,
             const int n_per_thread) {
    using cl::Buffer;
    using cl::EnqueueArgs;
    using cl::NDRange;

    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;

    auto kernels = getRegionsKernels<T>(full_conn, n_per_thread);

    const NDRange local(THREADS_X, THREADS_Y);

    const int blk_x = divup(in.info.dims[0], THREADS_X * 2);
    const int blk_y = divup(in.info.dims[1], THREADS_Y * 2);

    const NDRange global(blk_x * THREADS_X, blk_y * THREADS_Y);

    auto ilOp = kernels[0];
    auto ueOp = kernels[2];
    auto frOp = kernels[1];

    ilOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data,
         in.info);
    CL_DEBUG_FINISH(getQueue());

    int h_continue     = 1;
    Buffer* d_continue = bufferAlloc(sizeof(int));

    while (h_continue) {
        h_continue = 0;
        getQueue().enqueueFillBuffer(*d_continue, h_continue, 0, sizeof(int));
        ueOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *d_continue);
        CL_DEBUG_FINISH(getQueue());
        getQueue().enqueueReadBuffer(*d_continue, CL_TRUE, 0, sizeof(int),
                                     &h_continue);
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
    clEnqueueCopyBuffer(getQueue()(), (*out.data)(), tmp.get_buffer().get(), 0,
                        0, size * sizeof(T), 0, NULL, NULL);

    // Sort the copy
    compute::sort(tmp.begin(), tmp.end(), c_queue);

    // Take the max element which is the number
    // of label assignments to compute.
    T last_label;
    clEnqueueReadBuffer(getQueue()(), tmp.get_buffer().get(), CL_TRUE,
                        (size - 1) * sizeof(T), sizeof(T), &last_label, 0, NULL,
                        NULL);
    const int num_bins = (int)last_label + 1;

    // If the number of label assignments is two,
    // then either the entire input image is one big
    // component(1's) or it has only one component other than
    // background(0's). Either way, no further
    // post-processing of labels is required.
    if (num_bins <= 2) return;

    Buffer labels(getContext(), CL_MEM_READ_WRITE, num_bins * sizeof(T));
    compute::buffer c_labels(labels());
    compute::buffer_iterator<T> labels_begin =
        compute::make_buffer_iterator<T>(c_labels, 0);
    compute::buffer_iterator<T> labels_end =
        compute::make_buffer_iterator<T>(c_labels, num_bins);

    // Find the end of each section of values
    compute::counting_iterator<T> search_begin(0);

    int tmp_size = size;
    BOOST_COMPUTE_CLOSURE(int, upper_bound_closure, (int v), (tmp, tmp_size), {
        int start = 0, n = tmp_size, i;
        while (start < n) {
            i = (start + n) / 2;
            if (v < tmp[i]) {
                n = i;
            } else {
                start = i + 1;
            }
        }
        return start;
    });

    BOOST_COMPUTE_FUNCTION(int, clamp_to_one, (int i),
                           { return (i >= 1) ? 1 : i; });

    compute::transform(search_begin, search_begin + num_bins, labels_begin,
                       upper_bound_closure, c_queue);
    compute::adjacent_difference(labels_begin, labels_end, labels_begin,
                                 c_queue);

    // Perform the scan -- this can computes the correct labels for each
    // component
    compute::transform(labels_begin, labels_end, labels_begin, clamp_to_one,
                       c_queue);

    compute::exclusive_scan(labels_begin, labels_end, labels_begin, c_queue);

    // Apply the correct labels to the equivalency map
    // Buffer labels_buf(tmp.get_buffer().get());
    frOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data,
         in.info, labels);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
