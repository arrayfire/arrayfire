/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/dispatch.hpp>
#include <err_cuda.hpp>
#include <kernel_headers/jit.hpp>

#include <Array.hpp>
#include <common/jit/Node.hpp>
#include <common/jit/NodeIterator.hpp>
#include <copy.hpp>
#include <jit/BufferNode.hpp>
#include <jit/ShiftNode.hpp>
#include <math.hpp>
#include <platform.hpp>

#include <nvrtc/cache.hpp>
#include <af/dim4.hpp>

#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cuda {

using common::Node;
using common::Node_ids;
using common::Node_map_t;
using common::NodeIterator;
using common::requiresGlobalMemoryAccess;
using cuda::jit::BufferNode;
using cuda::jit::ShiftNode;

using std::hash;
using std::map;
using std::string;
using std::stringstream;
using std::vector;

template<typename TL, typename TR>
bool equal_shape(const Param<TL> &lhs, const Param<TR> &rhs) {
    return std::equal(lhs.dims, lhs.dims + 4, rhs.dims) &&
           std::equal(lhs.strides, lhs.strides + 4, rhs.strides);
}

static string getFuncName(const vector<Node *> &output_nodes,
                          const vector<const Node *> &full_nodes,
                          const vector<Node_ids> &full_ids, bool is_linear) {
    stringstream funcName;
    stringstream hashName;

    if (is_linear)
        funcName << "L_";  // Kernel Linear
    else
        funcName << "G_";  // Kernel General

    for (const auto &node : output_nodes) {
        funcName << node->getNameStr() << "_";
    }

    for (int i = 0; i < (int)full_nodes.size(); i++) {
        full_nodes[i]->genKerName(funcName, full_ids[i]);
    }

    hash<string> hash_fn;

    hashName << "KER";
    hashName << hash_fn(funcName.str());
    return hashName.str();
}

static string getKernelString(const string funcName,
                              const vector<const Node *> &full_nodes,
                              const vector<Node_ids> &full_ids,
                              const vector<int> &output_ids, bool is_linear) {
    const std::string includeFileStr(jit_cuh, jit_cuh_len);

    const std::string paramTStr = R"JIT(
struct Param
{
  dim_t dims[4];
  dim_t strides[4];
  void *ptr;
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
        "uint blocks_x, uint blocks_y, uint "
        "blocks_x_total, uint num_odims";
    static const char *globalParams = "Param* dims, int num_params, ";

    static const char *loopStart = R"JIT(
    for (int blockIdx_x = blockIdx.x; blockIdx_x < blocks_x_total; blockIdx_x += gridDim.x) {
    )JIT";
    static const char *loopEnd   = "}\n\n";

    static const char *blockStart = "{\n\n";
    static const char *blockEnd   = "\n}\n";

    static const char *linearIndex = R"JIT(
    uint threadId = threadIdx.x;
    long long idx = blockIdx_x * blockDim.x * blockDim.y + threadId;
    if (idx >= outref.dims[3] * outref.strides[3]) return;
    )JIT";

    static const char *generalIndex = R"JIT(
    long long id0 = 0, id1 = 0, id2 = 0, id3 = 0;
    long blockIdx_y = blockIdx.z * gridDim.y + blockIdx.y;
    if (num_odims > 2) {
        id2 = blockIdx_x / blocks_x;
        id0 = blockIdx_x - id2 * blocks_x;
        id0 = threadIdx.x + id0 * blockDim.x;
        if (num_odims > 3) {
            id3 = blockIdx_y / blocks_y;
            id1 = blockIdx_y - id3 * blocks_y;
            id1 = threadIdx.y + id1 * blockDim.y;
        } else {
            id1 = threadIdx.y + blockDim.y * blockIdx_y;
        }
    } else {
        id3 = 0;
        id2 = 0;
        id1 = threadIdx.y + blockDim.y * blockIdx_y;
        id0 = threadIdx.x + blockDim.x * blockIdx_x;
    }

    bool cond = id0 < outref.dims[0] &&
                id1 < outref.dims[1] &&
                id2 < outref.dims[2] &&
                id3 < outref.dims[3];

    if (!cond) { continue; }

    long long idx = outref.strides[3] * id3 +
                    outref.strides[2] * id2 +
                    outref.strides[1] * id1 + id0;

    if (threadIdx.x < num_params && threadIdx.y == 0) { //(int i = 0; i < num_params; i++) {
        int i = threadIdx.x;
        block_offsets[i] = (id3 < params[i].dims[3]) * params[i].strides[3] * id3 +
                           (id2 < params[i].dims[2]) * params[i].strides[2] * id2;
    }
    __syncthreads();)JIT";

    stringstream inParamStream;
    stringstream outParamStream;
    stringstream outWriteStream;
    stringstream offsetsStream;
    stringstream opsStream;
    stringstream outrefstream;
    stringstream paramreadstream;

    outrefstream << "const Param &outref = out" << output_ids[0] << ";\n";

    for (int i = 0; i < (int)full_nodes.size(); i++) {
        const auto &node     = full_nodes[i];
        const auto &ids_curr = full_ids[i];
        // Generate input parameters, only needs current id
        node->genParams(inParamStream, ids_curr.id, is_linear);
        // Generate input offsets, only needs current id
        node->genOffsets(offsetsStream, ids_curr.id, is_linear);
        // Generate the core function body, needs children ids as well
        node->genFuncs(opsStream, ids_curr);
    }

    for (int i = 0; i < (int)output_ids.size(); i++) {
        int id = output_ids[i];
        // Generate output parameters
        outParamStream << "Param out" << id << ", \n";

        // Generate code to write the output
        outWriteStream << "((" << full_nodes[id]->getTypeStr() << "*)(out" << id
                       << ".ptr))"
                       << "[idx] = val" << id << ";\n";
    }

    paramreadstream << R"JIT(
        extern __shared__ char smem[];
        Param *params = reinterpret_cast<Param*>(smem);
        dim_t *block_offsets = reinterpret_cast<dim_t*>(smem+(num_params * sizeof(Param)));

        if (threadIdx.x < num_params) { params[threadIdx.x] = dims[threadIdx.x]; }
        __syncthreads();
    )JIT";

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
    if (!is_linear) { kerStream << globalParams; }
    kerStream << dimParams;
    kerStream << ")\n";
    kerStream << blockStart;
    kerStream << outrefstream.str();
    if (!is_linear) { kerStream << paramreadstream.str(); }
    kerStream << loopStart;
    if (is_linear) {
        kerStream << linearIndex;
    } else {
        kerStream << generalIndex;
    }
    kerStream << offsetsStream.str();
    kerStream << opsStream.str();
    kerStream << outWriteStream.str();
    kerStream << loopEnd;
    kerStream << blockEnd;

    return kerStream.str();
}

static CUfunction getKernel(const vector<Node *> &output_nodes,
                            const vector<int> &output_ids,
                            const vector<const Node *> &full_nodes,
                            const vector<Node_ids> &full_ids,
                            const bool is_linear) {
    typedef map<string, Kernel> kc_t;

    thread_local kc_t kernelCaches[DeviceManager::MAX_DEVICES];

    string funcName =
        getFuncName(output_nodes, full_nodes, full_ids, is_linear);
    int device = getActiveDeviceId();

    kc_t::iterator idx = kernelCaches[device].find(funcName);
    Kernel entry{nullptr, nullptr};

    if (idx == kernelCaches[device].end()) {
        string jit_ker = getKernelString(funcName, full_nodes, full_ids,
                                         output_ids, is_linear);
        saveKernel(funcName, jit_ker, ".cu");
        entry = buildKernel(funcName, jit_ker, {}, true);
        kernelCaches[device][funcName] = entry;
    } else {
        entry = idx->second;
    }

    return entry.ker;
}

template<typename T>
void evalNodes(vector<Param<T>> &outputs, vector<Node *> output_nodes) {
    int num_outputs = (int)outputs.size();
    int device      = getActiveDeviceId();

    if (num_outputs == 0) return;

    thread_local Node_map_t nodes;
    thread_local vector<const Node *> full_nodes;
    thread_local vector<Node_ids> full_ids;
    thread_local vector<int> output_ids;

    // Reserve some space to improve performance at smaller sizes
    if (nodes.size() == 0) {
        nodes.reserve(1024);
        output_ids.reserve(output_nodes.size());
        full_nodes.reserve(1024);
        full_ids.reserve(1024);
    }

    vector<Param<T>> params;
    for (auto &node : output_nodes) {
        int id = node->getNodesMap(nodes, full_nodes, full_ids);
        output_ids.push_back(id);

        NodeIterator<> end_node;
        auto bufit = NodeIterator<>(node);
        while (bufit != end_node) {
            bufit = find_if(bufit, end_node, requiresGlobalMemoryAccess);
            if (bufit != end_node) {
                Param<T> param;

                // TODO(umar): This is a hack. We need to clean up this API
                // so that the if statement is not necessary
                if (bufit->isBuffer()) {
                    param = static_cast<BufferNode<T> &>(*bufit).getParam();
                } else {
                    param = static_cast<ShiftNode<T> &>(*bufit).getParam();
                }

                auto it = find_if(begin(params), end(params),
                                  [&param](const Param<T> &p) {
                                      return equal_shape<T, T>(param, p);
                                  });
                if (it == end(params)) {
                    params.push_back(param);
                    bufit->setParamIndex(params.size() - 1);
                } else {
                    bufit->setParamIndex(distance(begin(params), it));
                }
                bufit++;
            }
        }
    }

    bool is_linear = true;
    for (auto node : full_nodes) {
        is_linear &= node->isLinear(outputs[0].dims);
    }

    CUfunction ker =
        getKernel(output_nodes, output_ids, full_nodes, full_ids, is_linear);

    int threads_x = 1, threads_y = 1;
    int blocks_x_ = 1, blocks_y_ = 1;
    int blocks_x = 1, blocks_y = 1, blocks_z = 1, blocks_x_total;

    cudaDeviceProp properties    = getDeviceProp(device);
    const long long max_blocks_x = properties.maxGridSize[0];
    const long long max_blocks_y = properties.maxGridSize[1];

    int num_odims = 4;
    while (num_odims >= 1) {
        if (outputs[0].dims[num_odims - 1] == 1)
            num_odims--;
        else
            break;
    }

    if (is_linear) {
        threads_x = 256;
        threads_y = 1;

        blocks_x_total = divup((outputs[0].dims[0] * outputs[0].dims[1] *
                                outputs[0].dims[2] * outputs[0].dims[3]),
                               threads_x);

        int repeat_x = divup(blocks_x_total, max_blocks_x);
        blocks_x     = divup(blocks_x_total, repeat_x);
    } else {
        threads_x = 32;
        threads_y = 8;

        blocks_x_ = divup(outputs[0].dims[0], threads_x);
        blocks_y_ = divup(outputs[0].dims[1], threads_y);

        blocks_x = blocks_x_ * outputs[0].dims[2];
        blocks_y = blocks_y_ * outputs[0].dims[3];

        blocks_z = divup(blocks_y, max_blocks_y);
        blocks_y = divup(blocks_y, blocks_z);

        blocks_x_total = blocks_x;
        int repeat_x   = divup(blocks_x_total, max_blocks_x);
        blocks_x       = divup(blocks_x_total, repeat_x);
    }

    vector<void *> args;
    for (const auto &node : full_nodes) {
        node->setArgs(0, is_linear,
                      [&](int /*id*/, const void *ptr, size_t /*size*/) {
                          args.push_back(const_cast<void *>(ptr));
                      });
    }

    for (int i = 0; i < num_outputs; i++) {
        args.push_back((void *)&outputs[i]);
    }

    uptr<uchar> dparam;
    void *ptr       = nullptr;
    int param_count = params.size();
    if (!is_linear) {
        dparam = memAlloc<uchar>(params.size() * sizeof(Param<T>));
        CUDA_CHECK(cudaMemcpyAsync(dparam.get(), params.data(),
                                   params.size() * sizeof(Param<T>),
                                   cudaMemcpyHostToDevice, getActiveStream()));
        ptr = dparam.get();
        args.push_back(&ptr);
        args.push_back((void *)&param_count);
    }

    args.push_back((void *)&blocks_x_);
    args.push_back((void *)&blocks_y_);
    args.push_back((void *)&blocks_x_total);
    args.push_back((void *)&num_odims);

    CU_CHECK(cuLaunchKernel(ker, blocks_x, blocks_y, blocks_z, threads_x,
                            threads_y, 1, (sizeof(Param<T>) + sizeof(dim_t)) * params.size(),
                            getActiveStream(), args.data(), NULL));

    // Reset the thread local vectors
    nodes.clear();
    output_ids.clear();
    full_nodes.clear();
    full_ids.clear();
}

template<typename T>
void evalNodes(Param<T> out, Node *node) {
    vector<Param<T>> outputs;
    vector<Node *> output_nodes;

    outputs.push_back(out);
    output_nodes.push_back(node);
    evalNodes(outputs, output_nodes);
    return;
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

template void evalNodes<float>(vector<Param<float>> &out, vector<Node *> node);
template void evalNodes<double>(vector<Param<double>> &out,
                                vector<Node *> node);
template void evalNodes<cfloat>(vector<Param<cfloat>> &out,
                                vector<Node *> node);
template void evalNodes<cdouble>(vector<Param<cdouble>> &out,
                                 vector<Node *> node);
template void evalNodes<int>(vector<Param<int>> &out, vector<Node *> node);
template void evalNodes<uint>(vector<Param<uint>> &out, vector<Node *> node);
template void evalNodes<char>(vector<Param<char>> &out, vector<Node *> node);
template void evalNodes<uchar>(vector<Param<uchar>> &out, vector<Node *> node);
template void evalNodes<intl>(vector<Param<intl>> &out, vector<Node *> node);
template void evalNodes<uintl>(vector<Param<uintl>> &out, vector<Node *> node);
template void evalNodes<short>(vector<Param<short>> &out, vector<Node *> node);
template void evalNodes<ushort>(vector<Param<ushort>> &out,
                                vector<Node *> node);
}  // namespace cuda
