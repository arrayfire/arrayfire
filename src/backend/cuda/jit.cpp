/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <Kernel.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/jit/Node.hpp>
#include <common/kernel_cache.hpp>
#include <common/util.hpp>
#include <copy.hpp>
#include <debug_cuda.hpp>
#include <device_manager.hpp>
#include <err_cuda.hpp>
#include <kernel_headers/jit_cuh.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <af/dim4.hpp>

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using common::findModule;
using common::getFuncName;
using common::half;
using common::Node;
using common::Node_ids;
using common::Node_map_t;

using std::array;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

namespace cuda {

static string getKernelString(const string &funcName,
                              const vector<Node *> &full_nodes,
                              const vector<Node_ids> &full_ids,
                              const vector<int> &output_ids, bool is_linear) {
    const std::string includeFileStr(jit_cuh, jit_cuh_len);

    const std::string paramTStr = R"JIT(
template<typename T>
struct Param {
    dim_t dims[4];
    dim_t strides[4];
    T *ptr;
};
)JIT";

    std::string typedefStr = "typedef unsigned int uint;\n";
    typedefStr += "typedef ";
    typedefStr += getFullName<dim_t>();
    typedefStr += " dim_t;\n";

    // Common CUDA code
    // This part of the code does not change with the kernel.

    static const char *kernelVoid = "extern \"C\" __global__ void\n";
    static const char *dimParams =
        "int inc0, int inc1, int inc2, int inc3, "
        "char decode0, char decode1, char decode2, char decode3";

    static const char *blockStart = "{\n";
    static const char *blockEnd   = "\n}\n\n";

    static const char *linearIndexStart = R"JIT(
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // only dim0 since linear
    if (idx < (int)outref.dims[2] * (int)outref.strides[2]) {
        const int ostrides3 = outref.strides[3];
        const int idxEnd = idx + (int)outref.dims[3] * ostrides3;
        do {
    )JIT";

    static const char *linearIndexEnd = R"JIT(
            idx += ostrides3;
        } while (idx != idxEnd);
    }
    )JIT";

    static const char *generalIndexStart = R"JIT(
    //optimized dims
    int od[4] {(int)(blockIdx.x * blockDim.x + threadIdx.x),
               (int)(blockIdx.y * blockDim.y + threadIdx.y),
               0,       // Filled in later
               0};
    const bool valid = (od[0] < (int)outref.dims[0]) &&
                       (od[1] < (int)outref.dims[1]);
    if (valid) {
        const int odims1 = outref.dims[1];
        const int odims2 = outref.dims[2];
        const int odims3 = outref.dims[3];
        int ostrides1 = outref.strides[1];
        const int ostrides2 = outref.strides[2];
        const int ostrides3 = outref.strides[3];
        int offset = od[0] * (int)outref.strides[0] + od[1] * ostrides1;
        ostrides1 *= gridDim.y;

        do {
            od[2] = (int)(blockIdx.z * blockDim.z + threadIdx.z);
            do {
                int idx = offset + od[2] * ostrides2;
                const int idxEnd = idx + odims3 * ostrides3;

                //convert from optimized dims to internal dims
                int id0 = od[decode0];    // input dim[0]
                int id1 = od[decode1];    // input dim[1]
                int id2 = od[decode2];    // input dim[2]
                int id3 = od[decode3];    // input dim[3]
                do {
    )JIT";

    static const char *generalIndexEnd = R"JIT(
                    idx += ostrides3;
                    id0 += inc0;
                    id1 += inc1;
                    id2 += inc2;
                    id3 += inc3;
                } while (idx != idxEnd);
                od[2] += gridDim.z;
            } while (od[2] < odims2);
            od[1] += gridDim.y;
            offset += ostrides1;
        } while (od[1] < odims1);
    }
    )JIT";

    stringstream inParamStream;
    stringstream outParamStream;
    stringstream outWriteStream;
    stringstream offsetsStream;
    stringstream opsStream;
    stringstream outrefstream;

    for (int i = 0; i < static_cast<int>(full_nodes.size()); i++) {
        const auto &node     = full_nodes[i];
        const auto &ids_curr = full_ids[i];
        // Generate input parameters, only needs current id
        node->genParams(inParamStream, ids_curr.id, is_linear);
        // Generate input offsets, only needs current id
        node->genOffsets(offsetsStream, ids_curr.id, is_linear);
        // Generate the core function body, needs children ids as well
        node->genFuncs(opsStream, ids_curr);
    }

    outrefstream << "const Param<" << full_nodes[output_ids[0]]->getTypeStr()
                 << "> &outref = out" << output_ids[0] << ";\n";

    for (int id : output_ids) {
        // Generate output parameters
        outParamStream << "Param<" << full_nodes[id]->getTypeStr() << "> out"
                       << id << ", \n";
        // Generate code to write the output
        outWriteStream << "out" << id << ".ptr[idx] = val" << id << ";\n";
    }

    // Put various blocks into a single stream
    stringstream kerStream;
    kerStream << typedefStr;
    kerStream << includeFileStr << "\n\n";
    kerStream << paramTStr << "\n";
    kerStream << kernelVoid;
    kerStream << funcName;
    kerStream << "(\n";
    kerStream << inParamStream.str();
    kerStream << outParamStream.str();
    kerStream << dimParams;
    kerStream << ")\n";
    kerStream << blockStart;
    kerStream << outrefstream.str();
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

static CUfunction getKernel(const vector<Node *> &output_nodes,
                            const vector<int> &output_ids,
                            const vector<Node *> &full_nodes,
                            const vector<Node_ids> &full_ids,
                            const bool is_linear) {
    const string funcName =
        getFuncName(output_nodes, full_nodes, full_ids, is_linear);
    const size_t moduleKey = deterministicHash(funcName);

    // A forward lookup in module cache helps avoid recompiling the jit
    // source generated from identical jit-trees. It also enables us
    // with a way to save jit kernels to disk only once
    auto entry = findModule(getActiveDeviceId(), moduleKey);

    if (entry.get() == nullptr) {
        const string jitKer = getKernelString(funcName, full_nodes, full_ids,
                                              output_ids, is_linear);
        saveKernel(funcName, jitKer, ".cu");

        common::Source jit_src{jitKer.c_str(), jitKer.size(),
                               deterministicHash(jitKer)};

        return common::getKernel(funcName, {jit_src}, {}, {}, true).get();
    }
    return common::getKernel(entry, funcName, true).get();
}

template<typename T>
void evalNodes(vector<Param<T>> &outputs, const vector<Node *> &output_nodes) {
    if (outputs.empty()) { return; }

    dim_t *outDims       = outputs[0].dims;
    dim_t *outStrides    = outputs[0].strides;
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

    for (auto &node : output_nodes) {
        int id = node->getNodesMap(nodes, full_nodes, full_ids);
        output_ids.push_back(id);
    }

    bool is_linear = true;
    for (const auto &node : full_nodes) {
        is_linear &= node->isLinear(outDims);
    }
    for (int dim = 0, elements = 1; dim < ndims; ++dim) {
        is_linear &= (elements == (int)outputs[0].strides[dim]);
        elements *= (int)outDims[dim];
    }

    auto ker =
        getKernel(output_nodes, output_ids, full_nodes, full_ids, is_linear);

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
    dim3 threads, blocks;

    if (is_linear) {
        outDims[0]    = numOutElems;
        outDims[1]    = 1;
        outDims[2]    = 1;
        outDims[3]    = 1;
        outStrides[0] = 1;
        outStrides[1] = outDims[0];
        outStrides[2] = outDims[0];
        outStrides[3] = outDims[0];
        ndims         = 1;
        if (numOutElems >= 8192 * 2) {
            for (unsigned i : {3, 4, 5, 7, 11, 2}) {
                if (numOutElems >= 8192 * i && (outDims[ndims - 1] % i) == 0) {
                    outDims[ndims - 1] /= i;
                    outDims[AF_MAX_DIMS - 1] = i;
                    for (int c = 1; c < AF_MAX_DIMS; ++c) {
                        outStrides[c] = outDims[0];
                    }
                    incr[AF_MAX_DIMS - 1] = 0;
                    incr[ndims - 1] = static_cast<int>(outDims[ndims - 1]);
                    ndims           = AF_MAX_DIMS;
                    // Once is sufficient
                    break;
                }
            }
        }
        threads = dim3(128U);
        blocks  = dim3(divup((int)outDims[0], threads.x));
    } else {
        // Push all active dimensions to the front, so that the OpenCL WG
        // indexes cover a larger range
        for (int c = 0, d = 0; c < ndims - 1; ++c, ++d) {
            // Eliminate the column with 1, so that we have more
            // appropriate indexes in the WG
            if (outDims[c] == 1) {
                for (int i = c; i < ndims - 1; ++i) {
                    outDims[i]    = outDims[i + 1];
                    outStrides[i] = outStrides[i + 1];
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
                    for (int c = ndims; c < AF_MAX_DIMS; ++c) {
                        // since we are beyond the ndims, the array is by
                        // definition linear here
                        outStrides[c] = outDims[c - 1] * outStrides[c - 1];
                    }
                    incr[AF_MAX_DIMS - 1] = 0;
                    // Search the internal id to be incremented
                    for (int c = 0; c < AF_MAX_DIMS; ++c) {
                        if (decode[c] == ndims - 1) {
                            incr[c] = static_cast<int>(outDims[ndims - 1]);
                        }
                    }
                    ndims = AF_MAX_DIMS;
                    // Once is sufficient
                    break;
                }
            }
        }
        const int *maxGridSize =
            cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize;
        threads = bestBlockSize<dim3>(outDims, 32);
        blocks =
            dim3(divup(static_cast<unsigned>(outDims[0]), threads.x),
                 std::min(static_cast<unsigned>(maxGridSize[1]),
                          divup(static_cast<unsigned>(outDims[1]), threads.y)),
                 std::min(static_cast<unsigned>(maxGridSize[2]),
                          divup(static_cast<unsigned>(outDims[2]), threads.z)));
    }
    vector<void *> args;
    for (const auto &node : full_nodes) {
        node->setArgs(0, is_linear,
                      [&](int /*id*/, const void *ptr, size_t /*size*/) {
                          args.push_back(const_cast<void *>(ptr));
                      });
    }

    for (auto &out : outputs) { args.push_back(static_cast<void *>(&out)); }
    for (auto &inc : incr) { args.push_back(static_cast<void *>(&inc)); }
    for (auto &dec : decode) { args.push_back(static_cast<void *>(&dec)); }

    {
        using namespace cuda::kernel_logger;
        AF_TRACE("Launching : Blocks: [{}] Threads: [{}] ",
                 dim3(blocks.x, blocks.y, blocks.z),
                 dim3(threads.x, threads.y, threads.z));
    }
    CU_CHECK(cuLaunchKernel(ker, blocks.x, blocks.y, blocks.z, threads.x,
                            threads.y, threads.z, 0, getActiveStream(),
                            args.data(), NULL));

    // Reset the thread local vectors
    nodes.clear();
    output_ids.clear();
    full_nodes.clear();
    full_ids.clear();
}

template<typename T>
void evalNodes(Param<T> out, Node *node) {
    vector<Param<T>> outputs{out};
    vector<Node *> nodes{node};
    evalNodes(outputs, nodes);
}

template void evalNodes<float>(Param<float> out, Node *node);
template void evalNodes<double>(Param<double> out, Node *node);
template void evalNodes<cfloat>(Param<cfloat> out, Node *node);
template void evalNodes<cdouble>(Param<cdouble> out, Node *node);
template void evalNodes<int>(Param<int> out, Node *node);
template void evalNodes<uint>(Param<uint> out, Node *node);
template void evalNodes<char>(Param<char> out, Node *node);
template void evalNodes<uchar>(Param<uchar> out, Node *node);
template void evalNodes<intl>(Param<intl> out, Node *node);
template void evalNodes<uintl>(Param<uintl> out, Node *node);
template void evalNodes<short>(Param<short> out, Node *node);
template void evalNodes<ushort>(Param<ushort> out, Node *node);
template void evalNodes<half>(Param<half> out, Node *node);

template void evalNodes<float>(vector<Param<float>> &out,
                               const vector<Node *> &node);
template void evalNodes<double>(vector<Param<double>> &out,
                                const vector<Node *> &node);
template void evalNodes<cfloat>(vector<Param<cfloat>> &out,
                                const vector<Node *> &node);
template void evalNodes<cdouble>(vector<Param<cdouble>> &out,
                                 const vector<Node *> &node);
template void evalNodes<int>(vector<Param<int>> &out,
                             const vector<Node *> &node);
template void evalNodes<uint>(vector<Param<uint>> &out,
                              const vector<Node *> &node);
template void evalNodes<char>(vector<Param<char>> &out,
                              const vector<Node *> &node);
template void evalNodes<uchar>(vector<Param<uchar>> &out,
                               const vector<Node *> &node);
template void evalNodes<intl>(vector<Param<intl>> &out,
                              const vector<Node *> &node);
template void evalNodes<uintl>(vector<Param<uintl>> &out,
                               const vector<Node *> &node);
template void evalNodes<short>(vector<Param<short>> &out,
                               const vector<Node *> &node);
template void evalNodes<ushort>(vector<Param<ushort>> &out,
                                const vector<Node *> &node);
template void evalNodes<half>(vector<Param<half>> &out,
                              const vector<Node *> &node);
}  // namespace cuda
