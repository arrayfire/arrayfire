/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/jit.hpp>
#include <common/dispatch.hpp>
#include <err_cuda.hpp>

#include <Array.hpp>
#include <copy.hpp>
#include <JIT/Node.hpp>
#include <platform.hpp>
#include <math.hpp>

#include <af/dim4.hpp>
#include <nvrtc.h>

#include <array>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cuda
{

using JIT::Node;
using JIT::Node_ids;
using JIT::Node_map_t;

using std::array;
using std::hash;
using std::lock_guard;
using std::map;
using std::mutex;
using std::recursive_mutex;
using std::string;
using std::stringstream;
using std::unique_ptr;
using std::vector;

static string getFuncName(const vector<Node *> &output_nodes,
                          const vector<Node *> &full_nodes,
                          const vector<Node_ids> &full_ids,
                          bool is_linear)
{
    stringstream funcName;
    stringstream hashName;

    if (is_linear) funcName << "L_"; //Kernel Linear
    else           funcName << "G_"; //Kernel General

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
                              const vector<Node *> &full_nodes,
                              const vector<Node_ids> &full_ids,
                              const vector<int> &output_ids,
                              bool is_linear)
{

    const std::string includeFileStr(jit_cuh, jit_cuh_len);

    const std::string paramTStr = R"JIT(
template<typename T>
struct Param
{
  T *ptr;
  dim_t dims[4];
  dim_t strides[4];
};
)JIT";

    std::string typedefStr = "typedef unsigned int uint;\n";
    typedefStr += "typedef ";
    typedefStr += getFullName<dim_t>();
    typedefStr += " dim_t;\n";

    // Common CUDA code
    // This part of the code does not change with the kernel.

    static const char *kernelVoid = "extern \"C\" __global__ void\n";
    static const char *dimParams = "uint blocks_x, uint blocks_y, uint blocks_x_total, uint num_odims";

    static const char * loopStart = R"JIT(
    for (int blockIdx_x = blockIdx.x; blockIdx_x < blocks_x_total; blockIdx_x += gridDim.x) {
    )JIT";
    static const char *loopEnd = "}\n\n";

    static const char *blockStart = "{\n\n";
    static const char *blockEnd = "\n\n}";

    static const char *linearIndex = R"JIT(
        uint threadId = threadIdx.x;
        int idx = blockIdx_x * blockDim.x * blockDim.y + threadId;
        if (idx >= outref.dims[3] * outref.strides[3]) return;
        )JIT";

    static const char *generalIndex = R"JIT(
        uint id0 = 0, id1 = 0, id2 = 0, id3 = 0;
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
        if (!cond) return;

        int idx = outref.strides[3] * id3 +
                  outref.strides[2] * id2 +
                  outref.strides[1] * id1 + id0;
        )JIT";

    stringstream inParamStream;
    stringstream outParamStream;
    stringstream outWriteStream;
    stringstream offsetsStream;
    stringstream opsStream;
    stringstream outrefstream;

    for (int i = 0; i < (int)full_nodes.size(); i++) {
        const auto &node = full_nodes[i];
        const auto &ids_curr = full_ids[i];
        // Generate input parameters, only needs current id
        node->genParams(inParamStream, ids_curr.id, is_linear);
        // Generate input offsets, only needs current id
        node->genOffsets(offsetsStream, ids_curr.id, is_linear);
        // Generate the core function body, needs children ids as well
        node->genFuncs(opsStream, ids_curr);
    }

    outrefstream << "Param<" << full_nodes[output_ids[0]]->getTypeStr()
                 << "> outref = out" << output_ids[0] << ";\n";

    for (int i = 0; i < (int)output_ids.size(); i++) {
        int id = output_ids[i];
        // Generate output parameters
        outParamStream << "Param<" << full_nodes[id]->getTypeStr() << "> out" << id << ", \n";
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

typedef struct {
    CUmodule prog;
    CUfunction ker;
} kc_entry_t;

#define CU_CHECK(fn) do {                                 \
        CUresult res = fn;                                \
        if (res == CUDA_SUCCESS) break;                   \
        char cu_err_msg[1024];                            \
        const char *cu_err_name;                          \
        const char *cu_err_string;                        \
        cuGetErrorName(res, &cu_err_name);                \
        cuGetErrorString(res, &cu_err_string);            \
        snprintf(cu_err_msg,                              \
                 sizeof(cu_err_msg),                      \
                 "CU Error %s(%d): %s\n",                 \
                 cu_err_name, (int)(res), cu_err_string); \
        AF_ERROR(cu_err_msg,                              \
                 AF_ERR_INTERNAL);                        \
    } while(0)

#ifndef NDEBUG
#define CU_LINK_CHECK(fn) do {                            \
        CUresult res = fn;                                \
        if (res == CUDA_SUCCESS) break;                   \
        char cu_err_msg[1024];                            \
        const char *cu_err_name;                          \
        cuGetErrorName(res, &cu_err_name);                \
        snprintf(cu_err_msg,                              \
                 sizeof(cu_err_msg),                      \
                 "CU Error %s(%d): %s\n",                 \
                 cu_err_name, (int)(res), linkError);     \
        AF_ERROR(cu_err_msg,                              \
                 AF_ERR_INTERNAL);                        \
    } while(0)
#else
#define CU_LINK_CHECK(fn) CU_CHECK(fn)
#endif

#ifndef NDEBUG
#define NVRTC_CHECK(fn) do {                            \
        nvrtcResult res = fn;                           \
        if (res == NVRTC_SUCCESS) break;                \
        size_t logSize;                                 \
        nvrtcGetProgramLogSize(prog, &logSize);         \
        unique_ptr<char []> log(new char[logSize +1]);  \
        char *logptr = log.get();                       \
        nvrtcGetProgramLog(prog, logptr);               \
        logptr[logSize] = '\x0';                        \
        printf("%s\n", logptr);                         \
        AF_ERROR("NVRTC ERROR",                         \
                 AF_ERR_INTERNAL);                      \
    } while(0)
#else
#define NVRTC_CHECK(fn) do {                        \
        nvrtcResult res = fn;                       \
        if (res == NVRTC_SUCCESS) break;            \
        char nvrtc_err_msg[1024];                   \
        snprintf(nvrtc_err_msg,                     \
                 sizeof(nvrtc_err_msg),             \
                 "NVRTC Error(%d): %s\n",           \
                 res, nvrtcGetErrorString(res));    \
        AF_ERROR(nvrtc_err_msg,                     \
                 AF_ERR_INTERNAL);                  \
    } while(0)
#endif

std::vector<char> compileToPTX(const char *ker_name, string jit_ker)
{
    nvrtcProgram prog;
    size_t ptx_size;
    std::vector<char> ptx;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, jit_ker.c_str(),
                                   ker_name, 0, NULL, NULL));

    auto dev = getDeviceProp(getActiveDeviceId());
    array<char, 32> arch;
    snprintf(arch.data(), arch.size(), "--gpu-architecture=compute_%d%d",
             dev.major, dev.minor);
    const char* compiler_options[] = {
      arch.data(),
#if !(defined(NDEBUG) || defined(__aarch64__) || defined(__LP64__))
      "--device-debug",
      "--generate-line-info"
#endif
    };
    int num_options = std::extent<decltype(compiler_options)>::value;
    NVRTC_CHECK(nvrtcCompileProgram(prog, num_options, compiler_options));

    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    return ptx;
}

static kc_entry_t compileKernel(const char *ker_name, string jit_ker)
{
    const size_t linkLogSize = 1024;
    char linkInfo[linkLogSize] = {0};
    char linkError[linkLogSize] = {0};

    auto ptx = compileToPTX(ker_name, jit_ker);

    CUlinkState linkState;
    CUjit_option linkOptions[] = {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_LOG_VERBOSE
    };

    void *linkOptionValues[] = {
        linkInfo,
        reinterpret_cast<void*>(linkLogSize),
        linkError,
        reinterpret_cast<void*>(linkLogSize),
        reinterpret_cast<void*>(1)
    };

    CU_LINK_CHECK(cuLinkCreate(5, linkOptions, linkOptionValues, &linkState));
    CU_LINK_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx.data(),
                                ptx.size(), ker_name, 0, NULL, NULL));

    void *cubin;
    size_t cubinSize;

    CUmodule module;
    CUfunction kernel;
    CU_LINK_CHECK(cuLinkComplete(linkState, &cubin, &cubinSize));
    CU_CHECK(cuModuleLoadDataEx(&module, cubin, 0, 0, 0));
    CU_CHECK(cuModuleGetFunction(&kernel, module, ker_name));
    CU_LINK_CHECK(cuLinkDestroy(linkState));
    kc_entry_t entry = {module, kernel};
    return entry;
}

static CUfunction getKernel(const vector<Node *> &output_nodes,
                            const vector<int> &output_ids,
                            const vector<Node *> &full_nodes,
                            const vector<Node_ids> &full_ids,
                            const bool is_linear)
{
    typedef map<string, kc_entry_t> kc_t;

    thread_local kc_t kernelCaches[DeviceManager::MAX_DEVICES];

    string funcName = getFuncName(output_nodes, full_nodes, full_ids, is_linear);
    int device      = getActiveDeviceId();

    kc_t::iterator idx = kernelCaches[device].find(funcName);
    kc_entry_t entry{nullptr, nullptr};

    if (idx == kernelCaches[device].end()) {
        string jit_ker = getKernelString(funcName, full_nodes, full_ids, output_ids, is_linear);
        entry = compileKernel(funcName.c_str(), jit_ker);
        kernelCaches[device][funcName] = entry;
    } else {
        entry = idx->second;
    }

    return entry.ker;
}

template<typename T>
void evalNodes(vector<Param<T>>& outputs, vector<Node *> output_nodes)
{
    int num_outputs = (int)outputs.size();

    if (num_outputs == 0) return;

    // Use thread local to reuse the memory every time you are here.
    thread_local Node_map_t nodes;
    thread_local vector<Node *> full_nodes;
    thread_local vector<Node_ids> full_ids;
    thread_local vector<int> output_ids;

    // Reserve some space to improve performance at smaller sizes
    if (nodes.size() == 0) {
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
    for (auto node : full_nodes) {
        is_linear &= node->isLinear(outputs[0].dims);
    }

    CUfunction ker = getKernel(output_nodes, output_ids,
                               full_nodes, full_ids,
                               is_linear);

    int threads_x = 1, threads_y = 1;
    int blocks_x_ = 1, blocks_y_ = 1;
    int blocks_x  = 1, blocks_y = 1, blocks_z = 1, blocks_x_total;
    const int max_blocks = 65535;

    int num_odims = 4;

    while (num_odims >= 1) {
        if (outputs[0].dims[num_odims - 1] == 1) num_odims--;
        else break;
    }

    if (is_linear) {

        threads_x = 256;
        threads_y =  1;

        blocks_x_total = divup((outputs[0].dims[0] *
                                outputs[0].dims[1] *
                                outputs[0].dims[2] *
                                outputs[0].dims[3]), threads_x);

        int repeat_x = divup(blocks_x_total, max_blocks);
        blocks_x = divup(blocks_x_total, repeat_x);
    } else {

        threads_x = 32;
        threads_y =  8;

        blocks_x_ = divup(outputs[0].dims[0], threads_x);
        blocks_y_ = divup(outputs[0].dims[1], threads_y);

        blocks_x = blocks_x_ * outputs[0].dims[2];
        blocks_y = blocks_y_ * outputs[0].dims[3];

        blocks_z = divup(blocks_y, max_blocks);
        blocks_y = divup(blocks_y, blocks_z);

        blocks_x_total = blocks_x;
        int repeat_x = divup(blocks_x_total, max_blocks);
        blocks_x = divup(blocks_x_total, repeat_x);
    }

    vector<void *> args;

    for (const auto &node : full_nodes) {
        node->setArgs(args, is_linear);
    }

    for (int i = 0; i < num_outputs; i++) {
        args.push_back((void *)&outputs[i]);
    }

    args.push_back((void *)&blocks_x_);
    args.push_back((void *)&blocks_y_);
    args.push_back((void *)&blocks_x_total);
    args.push_back((void *)&num_odims);

    CU_CHECK(cuLaunchKernel(ker,
                            blocks_x,
                            blocks_y,
                            blocks_z,
                            threads_x,
                            threads_y,
                            1,
                            0,
                            getActiveStream(),
                            &args.front(),
                            NULL));

    // Reset the thread local vectors
    nodes.clear();
    output_ids.clear();
    full_nodes.clear();
    full_ids.clear();
}

template<typename T>
void evalNodes(Param<T> out, Node *node)
{
    vector<Param<T>> outputs;
    vector<Node *> output_nodes;

    outputs.push_back(out);
    output_nodes.push_back(node);
    evalNodes(outputs, output_nodes);
    return;
}

template void evalNodes<float  >(Param<float  > out, Node *node);
template void evalNodes<double >(Param<double > out, Node *node);
template void evalNodes<cfloat >(Param<cfloat > out, Node *node);
template void evalNodes<cdouble>(Param<cdouble> out, Node *node);
template void evalNodes<int    >(Param<int    > out, Node *node);
template void evalNodes<uint   >(Param<uint   > out, Node *node);
template void evalNodes<char   >(Param<char   > out, Node *node);
template void evalNodes<uchar  >(Param<uchar  > out, Node *node);
template void evalNodes<intl   >(Param<intl   > out, Node *node);
template void evalNodes<uintl  >(Param<uintl  > out, Node *node);
template void evalNodes<short  >(Param<short  > out, Node *node);
template void evalNodes<ushort >(Param<ushort > out, Node *node);

template void evalNodes<float  >(vector<Param<float  > > &out, vector<Node *> node);
template void evalNodes<double >(vector<Param<double > > &out, vector<Node *> node);
template void evalNodes<cfloat >(vector<Param<cfloat > > &out, vector<Node *> node);
template void evalNodes<cdouble>(vector<Param<cdouble> > &out, vector<Node *> node);
template void evalNodes<int    >(vector<Param<int    > > &out, vector<Node *> node);
template void evalNodes<uint   >(vector<Param<uint   > > &out, vector<Node *> node);
template void evalNodes<char   >(vector<Param<char   > > &out, vector<Node *> node);
template void evalNodes<uchar  >(vector<Param<uchar  > > &out, vector<Node *> node);
template void evalNodes<intl   >(vector<Param<intl   > > &out, vector<Node *> node);
template void evalNodes<uintl  >(vector<Param<uintl  > > &out, vector<Node *> node);
template void evalNodes<short  >(vector<Param<short  > > &out, vector<Node *> node);
template void evalNodes<ushort >(vector<Param<ushort > > &out, vector<Node *> node);
}
