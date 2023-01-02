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
#include <common/deterministicHash.hpp>
#include <common/jit/ModdimNode.hpp>
#include <common/jit/Node.hpp>
#include <common/jit/NodeIterator.hpp>
#include <common/kernel_cache.hpp>
#include <common/util.hpp>
#include <copy.hpp>
#include <device_manager.hpp>
#include <err_opencl.hpp>
#include <jit/BufferNode.hpp>
#include <kernel_headers/jit.hpp>
#include <threadsMgt.hpp>
#include <type_util.hpp>
#include <af/dim4.hpp>
#include <af/opencl.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using arrayfire::common::findModule;
using arrayfire::common::getFuncName;
using arrayfire::common::ModdimNode;
using arrayfire::common::Node;
using arrayfire::common::Node_ids;
using arrayfire::common::Node_map_t;
using arrayfire::common::Node_ptr;
using arrayfire::common::NodeIterator;
using arrayfire::common::saveKernel;

using cl::Kernel;
using cl::NDRange;
using cl::NullRange;

using std::equal;
using std::for_each;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

namespace arrayfire {
namespace opencl {
using jit::BufferNode;

string getKernelString(const string& funcName, const vector<Node*>& full_nodes,
                       const vector<Node_ids>& full_ids,
                       const vector<int>& output_ids, const bool is_linear,
                       const bool loop0, const bool loop1, const bool loop3) {
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
    kerStream << kernelVoid << funcName << "(\n"
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

cl::Kernel getKernel(const vector<Node*>& output_nodes,
                     const vector<int>& output_ids,
                     const vector<Node*>& full_nodes,
                     const vector<Node_ids>& full_ids, const bool is_linear,
                     const bool loop0, const bool loop1, const bool loop3) {
    const string funcName{getFuncName(output_nodes, full_nodes, full_ids,
                                      is_linear, loop0, loop1, false, loop3)};
    // A forward lookup in module cache helps avoid recompiling the JIT
    // source generated from identical JIT-trees.
    const auto entry{
        findModule(getActiveDeviceId(), deterministicHash(funcName))};

    if (!entry) {
        const string jitKer{getKernelString(funcName, full_nodes, full_ids,
                                            output_ids, is_linear, loop0, loop1,
                                            loop3)};
        saveKernel(funcName, jitKer, ".cl");

        const common::Source jitKer_cl_src{
            jitKer.data(), jitKer.size(),
            deterministicHash(jitKer.data(), jitKer.size())};
        const cl::Device device{getDevice()};
        vector<string> options;
        if (isDoubleSupported(device)) {
            options.emplace_back(DefineKey(USE_DOUBLE));
        }
        if (isHalfSupported(device)) {
            options.emplace_back(DefineKey(USE_HALF));
        }
        return common::getKernel(funcName, {{jit_cl_src, jitKer_cl_src}}, {},
                                 options, true)
            .get();
    }
    return common::getKernel(entry, funcName, true).get();
}

void evalNodes(vector<Param>& outputs, const vector<Node*>& output_nodes) {
    const unsigned nrOutputs{static_cast<unsigned>(outputs.size())};
    if (nrOutputs == 0) { return; }
    assert(outputs.size() == output_nodes.size());
    KParam& out_info{outputs[0].info};
    dim_t* outDims{out_info.dims};
    dim_t* outStrides{out_info.strides};
#ifndef NDEBUG
    for_each(begin(outputs)++, end(outputs),
             [outDims, outStrides](Param& output) {
                 assert(equal(output.info.dims, output.info.dims + AF_MAX_DIMS,
                              outDims) &&
                        equal(output.info.strides,
                              output.info.strides + AF_MAX_DIMS, outStrides));
             });
#endif

    dim_t ndims{outDims[3] > 1   ? 4
                : outDims[2] > 1 ? 3
                : outDims[1] > 1 ? 2
                : outDims[0] > 0 ? 1
                                 : 0};
    bool is_linear{true};
    dim_t numOutElems{1};
    for (dim_t dim{0}; dim < ndims; ++dim) {
        is_linear &= (numOutElems == outStrides[dim]);
        numOutElems *= outDims[dim];
    }
    if (numOutElems == 0) { return; }

    // Use thread local to reuse the memory every time you are here.
    thread_local Node_map_t nodes;
    thread_local vector<Node*> full_nodes;
    thread_local vector<Node_ids> full_ids;
    thread_local vector<int> output_ids;

    // Reserve some space to improve performance at smaller sizes
    constexpr size_t CAP{1024};
    if (full_nodes.capacity() < CAP) {
        nodes.reserve(CAP);
        output_ids.reserve(10);
        full_nodes.reserve(CAP);
        full_ids.reserve(CAP);
    }

    const af::dtype outputType{output_nodes[0]->getType()};
    const size_t outputSizeofType{size_of(outputType)};
    for (Node* node : output_nodes) {
        assert(node->getType() == outputType);
        const int id{node->getNodesMap(nodes, full_nodes, full_ids)};
        output_ids.push_back(id);
    }

    const size_t outputSize{numOutElems * outputSizeofType * nrOutputs};
    size_t inputSize{0};
    unsigned nrInputs{0};
    bool moddimsFound{false};
    for (const Node* node : full_nodes) {
        is_linear &= node->isLinear(outDims);
        moddimsFound |= (node->getOp() == af_moddims_t);
        if (node->isBuffer()) {
            ++nrInputs;
            inputSize += node->getBytes();
        }
    }
    const size_t totalSize{inputSize + outputSize};

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

    // Keep in global scope, so that the nodes remain active for later referral
    // in case moddims operations or column elimination have to take place
    vector<Node_ptr> node_clones;
    // Avoid all cloning/copying when no moddims node is present (high chance)
    if (moddimsFound | emptyColumnsFound) {
        node_clones.reserve(full_nodes.size());
        for (Node* node : full_nodes) {
            node_clones.emplace_back(node->clone());
        }

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
                    BufferNode* buf{static_cast<BufferNode*>(&(*it))};
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
            const auto isBuffer{
                [](const Node_ptr& ptr) { return ptr->isBuffer(); }};
            for (auto nodeIt{begin(node_clones)}, endIt{end(node_clones)};
                 (nodeIt = find_if(nodeIt, endIt, isBuffer)) != endIt;
                 ++nodeIt) {
                BufferNode* buf{static_cast<BufferNode*>(nodeIt->get())};
                removeEmptyColumns(outDims, ndims, buf->m_param.dims,
                                   buf->m_param.strides);
            }
            for_each(++begin(outputs), end(outputs),
                     [outDims, ndims](Param& output) {
                         removeEmptyColumns(outDims, ndims, output.info.dims,
                                            output.info.strides);
                     });
            ndims = removeEmptyColumns(outDims, ndims, outDims, outStrides);
        }

        full_nodes.clear();
        for (Node_ptr& node : node_clones) { full_nodes.push_back(node.get()); }
    }

    threadsMgt<dim_t> th(outDims, ndims, nrInputs, nrOutputs, totalSize,
                         outputSizeofType);
    auto ker = getKernel(output_nodes, output_ids, full_nodes, full_ids,
                         is_linear, th.loop0, th.loop1, th.loop3);
    const cl::NDRange local{th.genLocal(ker)};
    const cl::NDRange global{th.genGlobal(local)};

    int nargs{0};
    for (const Node* node : full_nodes) {
        nargs = node->setArgs(nargs, is_linear,
                              [&ker](int id, const void* ptr, size_t arg_size) {
                                  ker.setArg(id, arg_size, ptr);
                              });
    }

    // Set output parameters
    for (const auto& output : outputs) {
        ker.setArg(nargs++, *(output.data));
        ker.setArg(nargs++, static_cast<int>(output.info.offset));
    }

    // Set dimensions
    // All outputs are asserted to be of same size
    // Just use the size from the first output
    ker.setArg(nargs++, out_info);

    {
        using namespace opencl::kernel_logger;
        AF_TRACE(
            "Launching : Dims: [{},{},{},{}] Global: [{},{},{}] Local: "
            "[{},{},{}] threads: {}",
            outDims[0], outDims[1], outDims[2], outDims[3], global[0],
            global[1], global[2], local[0], local[1], local[2],
            global[0] * global[1] * global[2]);
    }
    getQueue().enqueueNDRangeKernel(ker, NullRange, global, local);

    // Reset the thread local vectors
    nodes.clear();
    output_ids.clear();
    full_nodes.clear();
    full_ids.clear();
}

void evalNodes(Param& out, Node* node) {
    vector<Param> outputs{out};
    vector<Node*> nodes{node};
    return evalNodes(outputs, nodes);
}

}  // namespace opencl
}  // namespace arrayfire
