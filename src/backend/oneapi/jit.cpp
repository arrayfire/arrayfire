/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <CL/cl.h>
#include <jit/ShiftNode.hpp>
#include <jit/kernel_generators.hpp>

#include <kernel_headers/KParam.hpp>
#include <kernel_headers/jit.hpp>

#include <Array.hpp>
#include <Kernel.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/jit/ModdimNode.hpp>
#include <common/jit/Node.hpp>
#include <common/jit/NodeIterator.hpp>
#include <common/jit/ShiftNodeBase.hpp>
#include <common/util.hpp>
#include <copy.hpp>
#include <device_manager.hpp>
#include <err_oneapi.hpp>
#include <jit/BufferNode.hpp>
#include <platform.hpp>
#include <type_util.hpp>
#include <af/dim4.hpp>

#include <sycl/backend.hpp>
#include <sycl/sycl.hpp>

#include <array>
#include <cstdio>
#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using arrayfire::common::getFuncName;
using arrayfire::common::half;
using arrayfire::common::kNodeType;
using arrayfire::common::ModdimNode;
using arrayfire::common::Node;
using arrayfire::common::Node_ids;
using arrayfire::common::Node_map_t;
using arrayfire::common::Node_ptr;
using arrayfire::common::NodeIterator;
using arrayfire::common::ShiftNodeBase;
using arrayfire::oneapi::getActiveDeviceBaseBuildFlags;
using arrayfire::oneapi::jit::BufferNode;
using arrayfire::oneapi::jit::ShiftNode;

using std::array;
using std::begin;
using std::end;
using std::find;
using std::find_if;
using std::string;
using std::stringstream;
using std::to_string;
using std::unordered_map;
using std::vector;

using sycl::backend;

namespace arrayfire {

namespace opencl {

const static string DEFAULT_MACROS_STR(R"JIT(
#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#ifdef USE_HALF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#else
#define half short
#endif
#ifndef M_PI
#define
 M_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164
#endif
)JIT");

string getKernelString(const string& funcName,
                       const nonstd::span<Node* const> full_nodes,
                       nonstd::span<const Node_ids> full_ids,
                       const nonstd::span<int const> output_ids,
                       const bool is_linear, const bool loop0, const bool loop1,
                       const bool loop3) {
    // Common OpenCL code
    // This part of the code does not change with the kernel.

    static const char* kernelVoid = R"JIT(
__kernel void )JIT";
    static const char* dimParams  = "KParam oInfo";
    static const char* blockStart = "{";
    static const char* blockEnd   = "\n}\n";

    static const char* linearInit = R"JIT(
   int idx = get_global_id(0);
   const int idxEnd = oInfo.dims[0];
   if (idx < idxEnd) {
)JIT";
    static const char* linearEnd  = R"JIT(
   })JIT";

    static const char* linearLoop0Start = R"JIT(
        const int idxID0Inc = get_global_size(0);
        do {)JIT";
    static const char* linearLoop0End   = R"JIT(
            idx += idxID0Inc;
            if (idx >= idxEnd) break;
        } while (true);)JIT";

    // ///////////////////////////////////////////////
    // oInfo = output optimized information (dims, strides, offset).
    //         oInfo has removed dimensions, to optimized block scheduling
    // iInfo = input internal information (dims, strides, offset)
    //         iInfo has the original dimensions, auto generated code
    //
    // Loop3 is fastest and becomes inside loop, since
    //      - #of loops is known upfront
    // Loop1 is used for extra dynamic looping (writing into cache)
    // All loops are conditional and idependent
    // Format Loop1 & Loop3
    // ////////////////////////////
    //  *stridedLoopNInit               // Always
    //  *stridedLoop1Init               // Conditional
    //  *stridedLoop2Init               // Conditional
    //  *stridedLoop3Init               // Conditional
    //  *stridedLoop1Start              // Conditional
    //      *stridedLoop3Start          // Conditional
    //          auto generated code     // Always
    //      *stridedLoop3End            // Conditional
    //  *stridedLoop1End                // Conditional
    //  *StridedEnd                     // Always
    //
    // format loop0 (Vector only)
    // //////////////////////////
    // *stridedLoop0Init                // Always
    // *stridedLoop0Start               // Always
    //      auto generated code         // Always
    // *stridedLoop0End                 // Always
    // *stridedEnd                      // Always

    static const char* stridedLoop0Init  = R"JIT(
    int id0 = get_global_id(0);
    const int id0End = oInfo.dims[0];
    if (id0 < id0End) {
#define id1 0
#define id2 0
#define id3 0
        const int ostrides0 = oInfo.strides[0];
        int idx = ostrides0*id0;)JIT";
    static const char* stridedLoop0Start = R"JIT(
        const int id0Inc = get_global_size(0);
        const int idxID0Inc = ostrides0*id0Inc;
        do {)JIT";
    static const char* stridedLoop0End   = R"JIT(
            id0 += id0Inc;
            if (id0 >= id0End) break;
            idx += idxID0Inc;
        } while (true);)JIT";

    // -------------
    static const char* stridedLoopNInit = R"JIT(
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);
    const int id0End = oInfo.dims[0];
    const int id1End = oInfo.dims[1];
    if ((id0 < id0End) & (id1 < id1End)) {
        const int id2 = get_global_id(2);
#define id3 0
        const int ostrides1 = oInfo.strides[1];
        int idx = (int)oInfo.strides[0]*id0 + ostrides1*id1 + (int)oInfo.strides[2]*id2;)JIT";
    static const char* stridedEnd       = R"JIT(
    })JIT";

    static const char* stridedLoop3Init  = R"JIT(
#undef id3
        int id3 = 0;
        const int id3End = oInfo.dims[3];
        const int idxID3Inc = oInfo.strides[3];)JIT";
    static const char* stridedLoop3Start = R"JIT(
                const int idxBaseID3 = idx;
                do {)JIT";
    static const char* stridedLoop3End   = R"JIT(
                    ++id3;
                    if (id3 == id3End) break;
                    idx += idxID3Inc;
                } while (true);
                id3 = 0;
                idx = idxBaseID3;)JIT";

    static const char* stridedLoop1Init  = R"JIT(
        const int id1Inc = get_global_size(1);
        const int idxID1Inc = id1Inc * ostrides1;)JIT";
    static const char* stridedLoop1Start = R"JIT(
        do {)JIT";
    static const char* stridedLoop1End   = R"JIT(
            id1 += id1Inc;
            if (id1 >= id1End) break;
            idx += idxID1Inc;
        } while (true);)JIT";

    // Reuse stringstreams, because they are very costly during initilization
    thread_local stringstream inParamStream;
    thread_local stringstream outParamStream;
    thread_local stringstream outOffsetStream;
    thread_local stringstream inOffsetsStream;
    thread_local stringstream opsStream;

    int oid{0};
    for (size_t i{0}; i < full_nodes.size(); i++) {
        const auto& node{full_nodes[i]};
        const auto& ids_curr{full_ids[i]};
        // Generate input parameters, only needs current id
        node->genParams(inParamStream, ids_curr.id, is_linear);
        // Generate input offsets, only needs current id
        node->genOffsets(inOffsetsStream, ids_curr.id, is_linear);
        // Generate the core function body, needs children ids as well
        node->genFuncs(opsStream, ids_curr);
        for (auto outIt{begin(output_ids)}, endIt{end(output_ids)};
             (outIt = find(outIt, endIt, ids_curr.id)) != endIt; ++outIt) {
            // Generate also output parameters
            outParamStream << "__global "
                           << full_nodes[ids_curr.id]->getTypeStr() << " *out"
                           << oid << ", int offset" << oid << ",\n";
            // Apply output offset
            outOffsetStream << "\nout" << oid << " += offset" << oid << ';';
            // Generate code to write the output
            opsStream << "out" << oid << "[idx] = val" << ids_curr.id << ";\n";
            ++oid;
        }
    }

    thread_local stringstream kerStream;
    kerStream << DEFAULT_MACROS_STR << kernelVoid << funcName << "(\n"
              << inParamStream.str() << outParamStream.str() << dimParams << ")"
              << blockStart;
    if (is_linear) {
        kerStream << linearInit << inOffsetsStream.str()
                  << outOffsetStream.str() << '\n';
        if (loop0) kerStream << linearLoop0Start;
        kerStream << "\n\n" << opsStream.str();
        if (loop0) kerStream << linearLoop0End;
        kerStream << linearEnd;
    } else {
        if (loop0) {
            kerStream << stridedLoop0Init << outOffsetStream.str() << '\n'
                      << stridedLoop0Start;
        } else {
            kerStream << stridedLoopNInit << outOffsetStream.str() << '\n';
            if (loop3) kerStream << stridedLoop3Init;
            if (loop1) kerStream << stridedLoop1Init << stridedLoop1Start;
            if (loop3) kerStream << stridedLoop3Start;
        }
        kerStream << "\n\n" << inOffsetsStream.str() << opsStream.str();
        if (loop3) kerStream << stridedLoop3End;
        if (loop1) kerStream << stridedLoop1End;
        if (loop0) kerStream << stridedLoop0End;
        kerStream << stridedEnd;
    }
    kerStream << blockEnd;
    const string ret{kerStream.str()};

    // Prepare for next round, limit memory
    inParamStream.str("");
    outParamStream.str("");
    inOffsetsStream.str("");
    outOffsetStream.str("");
    opsStream.str("");
    kerStream.str("");

    return ret;
}

// cl::Kernel getKernel(const vector<Node*>& output_nodes,
//                      const vector<int>& output_ids,
//                      const vector<Node*>& full_nodes,
//                      const vector<Node_ids>& full_ids, const bool is_linear)
//                      {
//     ONEAPI_NOT_SUPPORTED("");
//     return common::getKernel("", "", true).get();
// }

static unordered_map<cl_device_id, std::string> device_name_map;
static std::mutex device_name_map_mutex;
static unordered_map<std::string, cl_kernel> kernel_map;
static std::mutex kernel_map_mutex;

template<typename T>
cl_kernel getKernel(
    std::string funcName, cl_context ctx, cl_device_id dev, cl_command_queue q,
    const nonstd::span<Node* const> full_nodes,
    nonstd::span<Node_ids const> full_ids, nonstd::span<int const> output_ids,
    nonstd::span<oneapi::AParam<T, sycl::access_mode::write> const> ap,
    bool is_linear) {
    std::string devName;
    {
        std::lock_guard<std::mutex> lock(device_name_map_mutex);

        auto devNameIt = device_name_map.find(dev);
        if (devNameIt == device_name_map.end()) {
            size_t devNameSz;
            CL_CHECK(
                clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &devNameSz));
            string newDevName(devNameSz, '\0');
            CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_NAME, devNameSz,
                                     newDevName.data(), nullptr));
            device_name_map[dev] = newDevName;
            devName              = newDevName;
        } else {
            devName = devNameIt->second;
        }
    }

    vector<cl_kernel> kernels(10);
    bool kernel_found;
    string kernelHash = funcName + devName;
    {
        std::lock_guard<std::mutex> lock(kernel_map_mutex);
        kernel_found = !(kernel_map.find(kernelHash) == end(kernel_map));
    }
    if (kernel_found) {
        std::lock_guard<std::mutex> lock(kernel_map_mutex);
        kernels[0] = kernel_map[kernelHash];
    } else {
        string jitstr = arrayfire::opencl::getKernelString(
            funcName, full_nodes, full_ids, output_ids, is_linear, false, false,
            ap[0].dims[2] > 1);

        cl_int err;
        vector<const char*> jitsources = {
            {arrayfire::oneapi::opencl::KParam_hpp,
             arrayfire::oneapi::opencl::jit_cl, jitstr.c_str()}};
        vector<size_t> jitsizes = {arrayfire::oneapi::opencl::KParam_hpp_len,
                                   arrayfire::oneapi::opencl::jit_cl_len,
                                   jitstr.size()};

        cl_program prog = clCreateProgramWithSource(
            ctx, jitsources.size(), jitsources.data(), jitsizes.data(), &err);

        std::string options = getActiveDeviceBaseBuildFlags();

        CL_CHECK_BUILD(
            clBuildProgram(prog, 1, &dev, options.c_str(), nullptr, nullptr));

        cl_uint ret_kernels = 0;
        CL_CHECK(
            clCreateKernelsInProgram(prog, 1, kernels.data(), &ret_kernels));

        std::lock_guard<std::mutex> lock(kernel_map_mutex);
        kernel_map[kernelHash] = kernels[0];
        CL_CHECK(clReleaseProgram(prog));
    }
    return kernels[0];
}

}  // namespace opencl

namespace oneapi {

template<typename T>
void evalNodes(vector<Param<T>>& outputs, const vector<Node*>& output_nodes) {
    if (outputs.empty()) return;
    Node_map_t nodes;
    vector<Node*> full_nodes;
    vector<Node_ids> full_ids;
    vector<int> output_ids;
    vector<Node_ptr> node_clones;

    bool is_linear{true};
    dim_t numOutElems{1};
    KParam& out_info{outputs[0].info};
    dim_t* outDims{out_info.dims};
    dim_t* outStrides{out_info.strides};

    dim_t ndims{outDims[3] > 1   ? 4
                : outDims[2] > 1 ? 3
                : outDims[1] > 1 ? 2
                : outDims[0] > 0 ? 1
                                 : 0};
    for (dim_t dim{0}; dim < ndims; ++dim) {
        is_linear &= (numOutElems == outStrides[dim]);
        numOutElems *= outDims[dim];
    }
    if (numOutElems == 0) { return; }

    for (Node* node : output_nodes) {
        const int id{node->getNodesMap(nodes, full_nodes, full_ids)};
        output_ids.push_back(id);
    }

    node_clones.clear();
    node_clones.reserve(full_nodes.size());
    for (Node* node : full_nodes) { node_clones.emplace_back(node->clone()); }

    bool moddimsFound{false};
    for (const Node* node : full_nodes) {
        is_linear &= node->isLinear(outDims);
        moddimsFound |= (node->getOp() == af_moddims_t);
    }

    bool emptyColumnsFound{false};
    if (is_linear) {
        outDims[0]    = numOutElems;
        outDims[1]    = 1;
        outDims[2]    = 1;
        outDims[3]    = 1;
        outStrides[0] = 1;
        outStrides[1] = numOutElems;
        outStrides[2] = numOutElems;
        outStrides[3] = numOutElems;
        ndims         = 1;
    } else {
        emptyColumnsFound = ndims > (outDims[0] == 1   ? 1
                                     : outDims[1] == 1 ? 2
                                     : outDims[2] == 1 ? 3
                                                       : 4);
    }

    //  Keep in global scope, so that the nodes remain active for later
    //  referral in case moddims operations or column elimination have to
    //  take place Avoid all cloning/copying when no moddims node is present
    //  (high chance)
    if (moddimsFound || emptyColumnsFound) {
        for (const Node_ids& ids : full_ids) {
            auto& children{node_clones[ids.id]->m_children};
            for (int i{0}; i < Node::kMaxChildren && children[i] != nullptr;
                 i++) {
                children[i] = node_clones[ids.child_ids[i]];
            }
        }

        if (moddimsFound) {
            const auto isModdim{[](const Node_ptr& ptr) {
                return ptr->getOp() == af_moddims_t;
            }};
            for (auto nodeIt{begin(node_clones)}, endIt{end(node_clones)};
                 (nodeIt = find_if(nodeIt, endIt, isModdim)) != endIt;
                 ++nodeIt) {
                const ModdimNode* mn{static_cast<ModdimNode*>(nodeIt->get())};

                const auto new_strides{calcStrides(mn->m_new_shape)};
                const auto isBuffer{
                    [](const Node& node) { return node.isBuffer(); }};
                for (NodeIterator<> it{nodeIt->get()}, end{NodeIterator<>()};
                     (it = find_if(it, end, isBuffer)) != end; ++it) {
                    jit::BufferNode<T>* buf{
                        static_cast<jit::BufferNode<T>*>(&(*it))};
                    buf->m_param.dims[0]    = mn->m_new_shape[0];
                    buf->m_param.dims[1]    = mn->m_new_shape[1];
                    buf->m_param.dims[2]    = mn->m_new_shape[2];
                    buf->m_param.dims[3]    = mn->m_new_shape[3];
                    buf->m_param.strides[0] = new_strides[0];
                    buf->m_param.strides[1] = new_strides[1];
                    buf->m_param.strides[2] = new_strides[2];
                    buf->m_param.strides[3] = new_strides[3];
                }
            }
        }
        if (emptyColumnsFound) {
            common::removeEmptyDimensions<Param<T>, BufferNode<T>,
                                          ShiftNode<T>>(outputs, node_clones);
        }
    }

    full_nodes.clear();
    for (Node_ptr& node : node_clones) { full_nodes.push_back(node.get()); }

    const string funcName{getFuncName(output_nodes, full_nodes, full_ids,
                                      is_linear, false, false, false,
                                      outputs[0].info.dims[2] > 1)};

    getQueue().submit([&](sycl::handler& h) {
        for (Node* node : full_nodes) {
            switch (node->getNodeType()) {
                case kNodeType::Buffer: {
                    BufferNode<T>* n = static_cast<BufferNode<T>*>(node);
                    n->m_param.require(h);
                } break;
                case kNodeType::Shift: {
                    ShiftNodeBase<jit::BufferNode<T>>* sn =
                        static_cast<ShiftNodeBase<jit::BufferNode<T>>*>(node);
                    sn->getBufferNode().m_param.require(h);
                } break;
                default: break;
            }
        }
        vector<AParam<T, sycl::access_mode::write>> ap;
        transform(begin(outputs), end(outputs), back_inserter(ap),
                  [&](const Param<T>& p) {
                      return AParam<T, sycl::access_mode::write>(
                          h, *p.data, p.info.dims, p.info.strides,
                          p.info.offset);
                  });

        h.host_task([ap, full_nodes, output_ids, full_ids, is_linear, funcName,
                     node_clones, nodes, outputs](sycl::interop_handle hh) {
            switch (hh.get_backend()) {
                case backend::opencl: {
                    auto ncc = node_clones;

                    cl_command_queue q = hh.get_native_queue<backend::opencl>();
                    cl_context ctx   = hh.get_native_context<backend::opencl>();
                    cl_device_id dev = hh.get_native_device<backend::opencl>();

                    cl_kernel kernel = arrayfire::opencl::getKernel<T>(
                        funcName, ctx, dev, q, full_nodes, full_ids, output_ids,
                        ap, is_linear);
                    int nargs{0};
                    for (Node* node : full_nodes) {
                        nargs = node->setArgs(
                            nargs, is_linear,
                            [&kernel, &hh, &is_linear](int id, const void* ptr,
                                                       size_t arg_size,
                                                       bool is_buffer) {
                                if (is_buffer) {
                                    auto* info = static_cast<
                                        AParam<T, sycl::access_mode::read>*>(
                                        const_cast<void*>(ptr));
                                    vector<cl_mem> mem =
                                        hh.get_native_mem<backend::opencl>(
                                            info->data);
                                    if (is_linear) {
                                        CL_CHECK(clSetKernelArg(kernel, id++,
                                                                sizeof(cl_mem),
                                                                &mem[0]));
                                        CL_CHECK(clSetKernelArg(kernel, id++,
                                                                sizeof(dim_t),
                                                                &info->offset));
                                    } else {
                                        CL_CHECK(clSetKernelArg(kernel, id++,
                                                                sizeof(cl_mem),
                                                                &mem[0]));
                                        KParam ooo = *info;
                                        CL_CHECK(clSetKernelArg(kernel, id++,
                                                                sizeof(KParam),
                                                                &ooo));
                                    }

                                } else {
                                    CL_CHECK(clSetKernelArg(kernel, id,
                                                            arg_size, ptr));
                                }
                            });
                    }

                    // Set output parameters
                    vector<cl_mem> mem;
                    for (const auto& output : ap) {
                        mem = hh.get_native_mem<backend::opencl>(output.data);
                        cl_mem mmm = mem[0];
                        CL_CHECK(clSetKernelArg(kernel, nargs++, sizeof(cl_mem),
                                                &mmm));
                        int off = output.offset;
                        CL_CHECK(
                            clSetKernelArg(kernel, nargs++, sizeof(int), &off));
                    }
                    const KParam ooo = ap[0];
                    CL_CHECK(
                        clSetKernelArg(kernel, nargs++, sizeof(KParam), &ooo));
                    array<size_t, 3> offset{0, 0, 0};
                    array<size_t, 3> global;
                    int ndims = 0;
                    if (is_linear) {
                        global = {(size_t)ap[0].dims.elements(), 0, 0};
                        ndims  = 1;
                    } else {
                        global = {(size_t)ap[0].dims[0], (size_t)ap[0].dims[1],
                                  (size_t)ap[0].dims[2]};
                        ndims  = 3;
                    }

                    {
                        using namespace oneapi::kernel_logger;
                        AF_TRACE(
                            "Launching {}: Dims: [{},{},{},{}] Global: "
                            "[{},{},{}] threads: {}",
                            funcName, ap[0].dims[0], ap[0].dims[1],
                            ap[0].dims[2], ap[0].dims[3], global[0], global[1],
                            global[2],
                            global[0] * std::max<size_t>(1, global[1]) *
                                std::max<size_t>(1, global[2]));
                    }

                    cl_event kernel_event;
                    CL_CHECK(clEnqueueNDRangeKernel(
                        q, kernel, ndims, offset.data(), global.data(), nullptr,
                        0, nullptr, &kernel_event));
                    CL_CHECK(clEnqueueBarrierWithWaitList(q, 1, &kernel_event,
                                                          nullptr));
                    CL_CHECK(clReleaseEvent(kernel_event));

                    CL_CHECK(clReleaseDevice(dev));
                    CL_CHECK(clReleaseContext(ctx));
                    CL_CHECK(clReleaseCommandQueue(q));

                } break;
                default: ONEAPI_NOT_SUPPORTED("Backend not supported");
            }
        });
    });
}

template<typename T>
void evalNodes(Param<T>& out, Node* node) {
    vector<Param<T>> outputs{out};
    vector<Node*> nodes{node};
    oneapi::evalNodes(outputs, nodes);
}

template void evalNodes<float>(Param<float>& out, Node* node);
template void evalNodes<double>(Param<double>& out, Node* node);
template void evalNodes<cfloat>(Param<cfloat>& out, Node* node);
template void evalNodes<cdouble>(Param<cdouble>& out, Node* node);
template void evalNodes<int>(Param<int>& out, Node* node);
template void evalNodes<uint>(Param<uint>& out, Node* node);
template void evalNodes<char>(Param<char>& out, Node* node);
template void evalNodes<uchar>(Param<uchar>& out, Node* node);
template void evalNodes<intl>(Param<intl>& out, Node* node);
template void evalNodes<uintl>(Param<uintl>& out, Node* node);
template void evalNodes<short>(Param<short>& out, Node* node);
template void evalNodes<ushort>(Param<ushort>& out, Node* node);
template void evalNodes<half>(Param<half>& out, Node* node);

template void evalNodes<float>(vector<Param<float>>& out,
                               const vector<Node*>& node);
template void evalNodes<double>(vector<Param<double>>& out,
                                const vector<Node*>& node);
template void evalNodes<cfloat>(vector<Param<cfloat>>& out,
                                const vector<Node*>& node);
template void evalNodes<cdouble>(vector<Param<cdouble>>& out,
                                 const vector<Node*>& node);
template void evalNodes<int>(vector<Param<int>>& out,
                             const vector<Node*>& node);
template void evalNodes<uint>(vector<Param<uint>>& out,
                              const vector<Node*>& node);
template void evalNodes<char>(vector<Param<char>>& out,
                              const vector<Node*>& node);
template void evalNodes<uchar>(vector<Param<uchar>>& out,
                               const vector<Node*>& node);
template void evalNodes<intl>(vector<Param<intl>>& out,
                              const vector<Node*>& node);
template void evalNodes<uintl>(vector<Param<uintl>>& out,
                               const vector<Node*>& node);
template void evalNodes<short>(vector<Param<short>>& out,
                               const vector<Node*>& node);
template void evalNodes<ushort>(vector<Param<ushort>>& out,
                                const vector<Node*>& node);
template void evalNodes<half>(vector<Param<half>>& out,
                              const vector<Node*>& node);

}  // namespace oneapi
}  // namespace arrayfire
