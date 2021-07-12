/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/compile_module.hpp>
#include <common/dispatch.hpp>
#include <common/jit/Node.hpp>
#include <common/kernel_cache.hpp>
#include <common/util.hpp>
#include <copy.hpp>
#include <device_manager.hpp>
#include <err_opencl.hpp>
#include <kernel_headers/jit.hpp>
#include <af/dim4.hpp>
#include <af/opencl.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using common::getFuncName;
using common::Node;
using common::Node_ids;
using common::Node_map_t;

using cl::Kernel;
using cl::NDRange;
using cl::NullRange;

using std::array;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

namespace opencl {

string getKernelString(const string &funcName, const vector<Node *> &full_nodes,
                       const vector<Node_ids> &full_ids,
                       const vector<int> &output_ids, bool is_linear) {
    // Common OpenCL code
    // This part of the code does not change with the kernel.

    static const char *kernelVoid = "__kernel void\n";
    static const char *dimParams =
        "KParam oInfo, "
        "int inc0, int inc1, int inc2, int inc3, "
        "char decode0, char decode1, char decode2, char decode3";
    static const char *blockStart = "{\n\n";
    static const char *blockEnd   = "\n\n}";

    static const char *linearIndexStart = R"JIT(
        int idx = get_global_id(0);
        if (idx < (int)oInfo.dims[2] * (int)oInfo.strides[2]) {
            const int ostrides3 = oInfo.strides[3];
            const int idxEnd = idx + (int)oInfo.dims[3] * ostrides3;
            do {
        )JIT";
    static const char *linearIndexEnd   = R"JIT(
                idx += ostrides3;
            } while (idx != idxEnd);
        }
        )JIT";

    static const char *generalIndexStart = R"JIT(
        //optimized dims
        const int od[4] = {(int)get_global_id(0),
                           (int)get_global_id(1),
                           (int)get_global_id(2),
                           0};
        const bool valid = (od[0] < (int)oInfo.dims[0]) && 
                           (od[1] < (int)oInfo.dims[1]);
        if (valid) {
            int idx = (int)oInfo.strides[0] * od[0] +
                      (int)oInfo.strides[1] * od[1] +
                      (int)oInfo.strides[2] * od[2] + oInfo.offset;
            const int ostrides3 = oInfo.strides[3];

            int id0 = od[decode0];  //input dim[0]
            int id1 = od[decode1];  //input dim[1]
            int id2 = od[decode2];  //input dim[2]
            int id3 = od[decode3];  //input dim[3]

            const int idxEnd = idx + (int)oInfo.dims[3] * ostrides3;
            do {
        )JIT";
    static const char *generalIndexEnd   = R"JIT(
                idx += ostrides3;
                id0 += inc0;
                id1 += inc1;
                id2 += inc2;
                id3 += inc3;
            } while (idx != idxEnd);
        }
        )JIT";

    stringstream inParamStream;
    stringstream outParamStream;
    stringstream outWriteStream;
    stringstream offsetsStream;
    stringstream opsStream;

    for (size_t i = 0; i < full_nodes.size(); i++) {
        const auto &node     = full_nodes[i];
        const auto &ids_curr = full_ids[i];
        // Generate input parameters, only needs current id
        node->genParams(inParamStream, ids_curr.id, is_linear);
        // Generate input offsets, only needs current id
        node->genOffsets(offsetsStream, ids_curr.id, is_linear);
        // Generate the core function body, needs children ids as well
        node->genFuncs(opsStream, ids_curr);
    }

    for (int id : output_ids) {
        // Generate output parameters
        outParamStream << "__global " << full_nodes[id]->getTypeStr() << " *out"
                       << id << ", \n";
        // Generate code to write the output
        outWriteStream << "out" << id << "[idx] = val" << id << ";\n";
    }

    // Put various blocks into a single stream
    stringstream kerStream;
    kerStream << kernelVoid;
    kerStream << funcName;
    kerStream << "(\n";
    kerStream << inParamStream.str();
    kerStream << outParamStream.str();
    kerStream << dimParams;
    kerStream << ")\n";
    kerStream << blockStart;
    if (is_linear) {
        kerStream << linearIndexStart;
    } else {
        kerStream << generalIndexStart;
    }
    kerStream << offsetsStream.str();
    kerStream << opsStream.str();
    kerStream << outWriteStream.str();
    if (is_linear) {
        kerStream << linearIndexEnd;
    } else {
        kerStream << generalIndexEnd;
    }
    kerStream << blockEnd;

    return kerStream.str();
}

cl::Kernel getKernel(const vector<Node *> &output_nodes,
                     const vector<int> &output_ids,
                     const vector<Node *> &full_nodes,
                     const vector<Node_ids> &full_ids, const bool is_linear) {
    const string funcName =
        getFuncName(output_nodes, full_nodes, full_ids, is_linear);
    const size_t moduleKey = deterministicHash(funcName);

    // A forward lookup in module cache helps avoid recompiling the jit
    // source generated from identical jit-trees. It also enables us
    // with a way to save jit kernels to disk only once
    auto entry = common::findModule(getActiveDeviceId(), moduleKey);

    if (!entry) {
        string jitKer = getKernelString(funcName, full_nodes, full_ids,
                                        output_ids, is_linear);
        common::Source jitKer_cl_src{
            jitKer.data(), jitKer.size(),
            deterministicHash(jitKer.data(), jitKer.size())};
        int device = getActiveDeviceId();
        vector<string> options;
        if (isDoubleSupported(device)) {
            options.emplace_back(DefineKey(USE_DOUBLE));
        }
        if (isHalfSupported(device)) {
            options.emplace_back(DefineKey(USE_HALF));
        }

        saveKernel(funcName, jitKer, ".cl");

        return common::getKernel(funcName, {jit_cl_src, jitKer_cl_src}, {},
                                 options, true)
            .get();
    }
    return common::getKernel(entry, funcName, true).get();
}

void evalNodes(vector<Param> &outputs, const vector<Node *> &output_nodes) {
    if (outputs.empty()) { return; }

    // Assume all ouputs are of same size
    // FIXME: Add assert to check if all outputs are same size?
    KParam out_info      = outputs[0].info;
    dim_t *outDims       = out_info.dims;
    unsigned numOutElems = static_cast<unsigned>(outDims[0] * outDims[1] *
                                                 outDims[2] * outDims[3]);
    int ndims            = outDims[3] > 1   ? 4
                           : outDims[2] > 1 ? 3
                           : outDims[1] > 1 ? 2
                           : outDims[0] > 0 ? 1
                                            : 0;
    if (numOutElems == 0 || ndims == 0) { return; }

    // Use thread local to reuse the memory every time you are here.
    thread_local Node_map_t nodes;
    thread_local vector<Node *> full_nodes;
    thread_local vector<Node_ids> full_ids;
    thread_local vector<int> output_ids;

    // Reserve some space to improve performance at smaller sizes
    if (nodes.empty()) {
        nodes.reserve(1024);
        output_ids.reserve(output_nodes.size());
        full_nodes.reserve(1024);
        full_ids.reserve(1024);
    }

    for (const auto &node : output_nodes) {
        int id = node->getNodesMap(nodes, full_nodes, full_ids);
        output_ids.push_back(id);
    }

    bool is_linear = true;
    for (const auto &node : full_nodes) {
        is_linear &= node->isLinear(outDims);
    }
    {
        dim_t elements = 1;
        for (int dim = 0; dim < ndims; ++dim) {
            is_linear &= (elements == out_info.strides[dim]);
            elements *= outDims[dim];
        }
    }
    is_linear &= (out_info.offset == 0);

    auto ker =
        getKernel(output_nodes, output_ids, full_nodes, full_ids, is_linear);
    const cl::Device dev = opencl::getDevice();
    const unsigned WG    = static_cast<unsigned>(
        ker.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
            dev));

    // decode is used to reconstruct the original dimensions inside the kernel
    // 0..3 represents the id number
    //
    // Suppose:
    // original dims[dim0,dim1,1,dim3]  // Columns with 1 waste a grid counter
    // optimized dims [od[0]=dim0,
    //                 od[1]=dim1,
    //                 od[2]=dim3,
    //                 od[3]=1..1 (internal looping)]
    // decode{0,1,3,2}  --> id in internal kernel {id0 = od[0] in [0..dim0[,
    //                                             id1 = od[1] in [0..dim1[,
    //                                             id2 = od[3] in [0..1[,
    //                                             id3 = od[2] in [0..dim3[ }
    char decode[AF_MAX_DIMS]{0, 1, 2, 3};
    int incr[AF_MAX_DIMS]{0, 0, 0, 1};
    NDRange local, global;

    if (is_linear) {
        outDims[0]          = numOutElems;
        outDims[1]          = 1;
        outDims[2]          = 1;
        outDims[3]          = 1;
        out_info.strides[0] = 1;
        out_info.strides[1] = outDims[0];
        out_info.strides[2] = outDims[0];
        out_info.strides[3] = outDims[0];
        ndims               = 1;

        if (numOutElems >= 8192 * 2) {
            for (unsigned i : {3, 4, 5, 7, 11, 2}) {
                if (numOutElems >= 8192 * i && (outDims[ndims - 1] % i) == 0) {
                    outDims[ndims - 1] /= i;
                    outDims[AF_MAX_DIMS - 1] = i;
                    for (int c = 1; c < AF_MAX_DIMS; ++c) {
                        out_info.strides[c] = outDims[0];
                    }
                    incr[AF_MAX_DIMS - 1] = 0;
                    incr[ndims - 1] = static_cast<int>(outDims[ndims - 1]);
                    ndims           = AF_MAX_DIMS;
                    // Once is sufficient
                    break;
                }
            }
        }
        local  = bestBlockSize<NDRange>(outDims, WG);
        global = NDRange(local[0] * divup(outDims[0], local[0]));
    } else {
        // Push all active dimensions to the front, so that the OpenCL WG
        // indexes cover a larger range
        for (int c = 0, d = 0; c < ndims - 1; ++c, ++d) {
            // Eliminate the column with 1, so that we have more
            // appropriate indexes in the WG
            if (outDims[c] == 1) {
                for (int i = c; i < ndims - 1; ++i) {
                    outDims[i]          = outDims[i + 1];
                    out_info.strides[i] = out_info.strides[i + 1];
                }
                // Reallocation of the WG indexes to the internal indexes
                for (int i = d + 1; i < AF_MAX_DIMS; ++i) { --decode[i]; }
                --ndims;
                // Replace the internal index with a fixed 1
                decode[d]      = 3;
                outDims[ndims] = 1;
                --c;  // Redo this column, since it is eliminated now!!
            }
        }
        // Increase work inside each thread
        // if last dim is free && some valid columns remain
        if (numOutElems >= 8192 * 2 && ndims != AF_MAX_DIMS && ndims != 0) {
            for (unsigned i : {3, 4, 5, 7, 11, 2}) {
                if (numOutElems >= 8192 * i && (outDims[ndims - 1] % i) == 0) {
                    outDims[ndims - 1] /= i;
                    outDims[AF_MAX_DIMS - 1] = i;
                    const dim_t stride =
                        outDims[ndims - 1] * out_info.strides[ndims - 1];
                    for (int c = ndims; c < AF_MAX_DIMS; ++c) {
                        out_info.strides[c] = stride;
                    }
                    incr[AF_MAX_DIMS - 1] = 0;
                    // Search the internal id to be incremented
                    for (int c = 0; c < AF_MAX_DIMS; ++c) {
                        if (decode[c] == ndims - 1) {
                            incr[c] = static_cast<int>(outDims[ndims - 1]);
                            break;
                        }
                    }
                    ndims = AF_MAX_DIMS;
                    // Once is sufficient
                    break;
                }
            }
        }
        local  = bestBlockSize<NDRange>(outDims, WG);
        global = NDRange(local[0] * divup(outDims[0], local[0]),
                         local[1] * divup(outDims[1], local[1]),
                         local[2] * divup(outDims[2], local[2]));
    }

    int nargs = 0;
    for (const auto &node : full_nodes) {
        nargs = node->setArgs(nargs, is_linear,
                              [&](int id, const void *ptr, size_t arg_size) {
                                  ker.setArg(id, arg_size, ptr);
                              });
    }

    // Set output parameters
    for (const auto &output : outputs) { ker.setArg(nargs++, *(output.data)); }

    // Set dimensions
    // All outputs are asserted to be of same size
    // Just use the size from the first output
    ker.setArg(nargs++, out_info);
    for (const auto inc : incr) { ker.setArg(nargs++, inc); }
    for (const auto dec : decode) { ker.setArg(nargs++, dec); }

    getQueue().enqueueNDRangeKernel(ker, NullRange, global, local);

    // Reset the thread local vectors
    nodes.clear();
    output_ids.clear();
    full_nodes.clear();
    full_ids.clear();
}

void evalNodes(Param &out, Node *node) {
    vector<Param> outputs{out};
    vector<Node *> nodes{node};
    return evalNodes(outputs, nodes);
}

}  // namespace opencl
