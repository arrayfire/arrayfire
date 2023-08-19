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
#include <common/deterministicHash.hpp>
#include <common/half.hpp>
#include <common/jit/ModdimNode.hpp>
#include <common/jit/Node.hpp>
#include <common/jit/NodeIterator.hpp>
#include <common/kernel_cache.hpp>
#include <common/util.hpp>
#include <copy.hpp>
#include <debug_cuda.hpp>
#include <device_manager.hpp>
#include <err_cuda.hpp>
#include <jit/ShiftNode.hpp>
#include <kernel_headers/jit_cuh.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <threadsMgt.hpp>
#include <type_util.hpp>
#include <af/dim4.hpp>

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using arrayfire::common::findModule;
using arrayfire::common::getEnvVar;
using arrayfire::common::getFuncName;
using arrayfire::common::half;
using arrayfire::common::isBufferOrShift;
using arrayfire::common::kNodeType;
using arrayfire::common::ModdimNode;
using arrayfire::common::Node;
using arrayfire::common::Node_ids;
using arrayfire::common::Node_map_t;
using arrayfire::common::Node_ptr;
using arrayfire::common::NodeIterator;
using arrayfire::common::saveKernel;
using arrayfire::cuda::jit::BufferNode;
using arrayfire::cuda::jit::ShiftNode;

using std::array;
using std::equal;
using std::find_if;
using std::for_each;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

namespace arrayfire {
namespace cuda {

static string getKernelString(const string& funcName,
                              const vector<Node*>& full_nodes,
                              const vector<Node_ids>& full_ids,
                              const vector<int>& output_ids,
                              const bool is_linear, const bool loop0,
                              const bool loop1, const bool loop2,
                              const bool loop3) {
    const std::string includeFileStr(jit_cuh, jit_cuh_len);

    const std::string paramTStr = R"JIT(
template<typename T>
struct Param {
    dim_t dims[4];
    dim_t strides[4];
    T *ptr;
};
)JIT";

    std::string typedefStr{"typedef unsigned int uint;\ntypedef "};
    typedefStr += getFullName<dim_t>();
    typedefStr += " dim_t;\n";

    // Common CUDA code
    // This part of the code does not change with the kernel.

    static const char* kernelVoid = "extern \"C\" __global__ void\n";
    static const char* dimParams  = "";

    static const char* blockStart = "{";
    static const char* blockEnd   = "\n}\n";

    static const char* linearInit = R"JIT(
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxEnd = outref.dims[0];
    if (idx < idxEnd) {)JIT";
    static const char* linearEnd  = R"JIT(
    })JIT";

    static const char* linearLoop0Start = R"JIT(
        const int idxID0Inc = gridDim.x*blockDim.x;
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
    // Loop0 is used for extra dynamic looping (writing into cache),
    //       VECTORS ONLY!!
    // All loops are conditional and idependent Format Loop1 & Loop3
    // ////////////////////////////
    //  *stridedLoopNInit               // Always
    //  *stridedLoop1Init               // Conditional
    //  *stridedLoop2Init               // Conditional
    //  *stridedLoop3Init               // Conditional
    //  *stridedLoop1Start              // Conditional
    //      *stridedLoop2Start          // Conditional
    //          *stridedLoop3Start      // Conditional
    //              auto generated code // Always
    //          *stridedLoop3End        // Conditional
    //      *stridedLoop2End            // Conditional
    //  *stridedLoop1End                // Conditional
    //  *stridedEnd                     // Always
    //
    // Format loop0 (Vector only)
    // //////////////////////////
    // *stridedLoop0Init                // Always
    // *stridedLoop0Start               // Always
    //      auto generated code         // Always
    // *stridedLoop0End                 // Always
    // *stridedEnd                      // Always

    // -----
    static const char* stridedLoop0Init  = R"JIT(
    int id0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int id0End = outref.dims[0];
    if (id0 < id0End) {
#define id1 0
#define id2 0
#define id3 0
        const int ostrides0 = outref.strides[0];
        int idx = ostrides0*id0;)JIT";
    static const char* stridedLoop0Start = R"JIT(
        const int id0Inc = gridDim.x*blockDim.x;
        const int idxID0Inc = ostrides0*id0Inc;
        do {)JIT";
    static const char* stridedLoop0End   = R"JIT(
            id0 += id0Inc;
            if (id0 >= id0End) break;
            idx += idxID0Inc;
        } while (true);)JIT";

    static const char* stridedLoopNInit = R"JIT(
    int id0 = blockIdx.x * blockDim.x + threadIdx.x;
    int id1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int id0End = outref.dims[0];
    const int id1End = outref.dims[1];
    if ((id0 < id0End) & (id1 < id1End)) {
        int id2 = blockIdx.z * blockDim.z + threadIdx.z;
#define id3 0
        const int ostrides1 = outref.strides[1];
        int idx = (int)outref.strides[0]*id0 + ostrides1*id1 + (int)outref.strides[2]*id2;)JIT";
    static const char* stridedEnd       = R"JIT(
    })JIT";

    static const char* stridedLoop3Init  = R"JIT(
#undef id3
        int id3 = 0;
        const int id3End = outref.dims[3];
        const int idxID3Inc = outref.strides[3];)JIT";
    static const char* stridedLoop3Start = R"JIT(
                    const int idxBaseID3 = idx;
                    do {)JIT";
    // Looping over outside dim3 means that all dimensions are present,
    // so the internal id3 can be used directly
    static const char* stridedLoop3End = R"JIT(
                       ++id3;
                       if (id3 == id3End) break;
                       idx += idxID3Inc;
                    } while (true);
                    id3 = 0;
                    idx = idxBaseID3;)JIT";

    static const char* stridedLoop2Init  = R"JIT(
        const int id2End = outref.dims[2];
        const int id2Inc = gridDim.z*blockDim.z;
        const int idxID2Inc = (int)outref.strides[2]*id2Inc;)JIT";
    static const char* stridedLoop2Start = R"JIT(
                const int idxBaseID2 = idx;
                const int baseID2 = id2;
                do {)JIT";
    static const char* stridedLoop2End   = R"JIT(
                    id2 += id2Inc;
                    if (id2 >= id2End) break;
                    idx += idxID2Inc;
                } while (true);
                id2 = baseID2;
                idx = idxBaseID2;)JIT";

    // No reset of od1/id[decode.dim1] is necessary since this is the overall
    // loop
    static const char* stridedLoop1Init  = R"JIT(
        const int id1Inc = gridDim.y*blockDim.y;
        const int idxID1Inc = ostrides1*id1Inc;)JIT";
    static const char* stridedLoop1Start = R"JIT(
            do {)JIT";
    static const char* stridedLoop1End   = R"JIT(
                id1 += id1Inc;
                if (id1 >= id1End) break;
                idx += idxID1Inc;
            } while (true);)JIT";

    // Reuse stringstreams, because they are very costly during initialization
    thread_local stringstream inParamStream;
    thread_local stringstream outParamStream;
    thread_local stringstream inOffsetsStream;
    thread_local stringstream opsStream;
    thread_local stringstream outrefStream;
    thread_local stringstream kerStream;

    string ret;
    try {
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
                outParamStream << (oid == 0 ? "" : ",\n") << "Param<"
                               << full_nodes[ids_curr.id]->getTypeStr()
                               << "> out" << oid;
                // Generate code to write the output (offset already in ptr)
                opsStream << "out" << oid << ".ptr[idx] = val" << ids_curr.id
                          << ";\n";
                ++oid;
            }
        }

        outrefStream << "\n    const Param<"
                     << full_nodes[output_ids[0]]->getTypeStr()
                     << "> &outref = out0;";

        // Put various blocks into a single stream
        kerStream << typedefStr << includeFileStr << "\n\n"
                  << paramTStr << '\n'
                  << kernelVoid << funcName << "(\n"
                  << inParamStream.str() << outParamStream.str() << dimParams
                  << ')' << blockStart << outrefStream.str();
        if (is_linear) {
            kerStream << linearInit;
            if (loop0) kerStream << linearLoop0Start;
            kerStream << "\n\n" << inOffsetsStream.str() << opsStream.str();
            if (loop0) kerStream << linearLoop0End;
            kerStream << linearEnd;
        } else {
            if (loop0) {
                kerStream << stridedLoop0Init << stridedLoop0Start;
            } else {
                kerStream << stridedLoopNInit;
                if (loop3) kerStream << stridedLoop3Init;
                if (loop2) kerStream << stridedLoop2Init;
                if (loop1) kerStream << stridedLoop1Init << stridedLoop1Start;
                if (loop2) kerStream << stridedLoop2Start;
                if (loop3) kerStream << stridedLoop3Start;
            }
            kerStream << "\n\n" << inOffsetsStream.str() << opsStream.str();
            if (loop3) kerStream << stridedLoop3End;
            if (loop2) kerStream << stridedLoop2End;
            if (loop1) kerStream << stridedLoop1End;
            if (loop0) kerStream << stridedLoop0End;
            kerStream << stridedEnd;
        }
        kerStream << blockEnd;
        ret = kerStream.str();
    } catch (...) {
        // Prepare for next round
        inParamStream.str("");
        outParamStream.str("");
        inOffsetsStream.str("");
        opsStream.str("");
        outrefStream.str("");
        kerStream.str("");
        throw;
    }

    // Prepare for next round
    inParamStream.str("");
    outParamStream.str("");
    inOffsetsStream.str("");
    opsStream.str("");
    outrefStream.str("");
    kerStream.str("");

    return ret;
}

static CUfunction getKernel(const vector<Node*>& output_nodes,
                            const vector<int>& output_ids,
                            const vector<Node*>& full_nodes,
                            const vector<Node_ids>& full_ids,
                            const bool is_linear, const bool loop0,
                            const bool loop1, const bool loop2,
                            const bool loop3) {
    const string funcName{getFuncName(output_nodes, full_nodes, full_ids,
                                      is_linear, loop0, loop1, loop2, loop3)};
    // A forward lookup in module cache helps avoid recompiling
    // the JIT source generated from identical JIT-trees.
    const auto entry{
        findModule(getActiveDeviceId(), deterministicHash(funcName))};

    if (!entry) {
        const string jitKer{getKernelString(funcName, full_nodes, full_ids,
                                            output_ids, is_linear, loop0, loop1,
                                            loop2, loop3)};
        saveKernel(funcName, jitKer, ".cu");

        const common::Source jit_src{jitKer.c_str(), jitKer.size(),
                                     deterministicHash(jitKer)};

        return common::getKernel(funcName, {{jit_src}}, {}, {}, true).get();
    }
    return common::getKernel(entry, funcName, true).get();
}

template<typename T>
void evalNodes(vector<Param<T>>& outputs, const vector<Node*>& output_nodes) {
    const unsigned nrOutputs{static_cast<unsigned>(output_nodes.size())};
    if (nrOutputs == 0) { return; }
    assert(outputs.size() == output_nodes.size());
    dim_t* outDims{outputs[0].dims};
    dim_t* outStrides{outputs[0].strides};
#ifndef NDEBUG
    for_each(
        begin(outputs)++, end(outputs),
        [outDims, outStrides](Param<T>& output) {
            assert(equal(output.dims, output.dims + AF_MAX_DIMS, outDims) &&
                   equal(output.strides, output.strides + AF_MAX_DIMS,
                         outStrides));
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

    // Use thread local to reuse the memory every time you are
    // here.
    thread_local Node_map_t nodes;
    thread_local vector<Node*> full_nodes;
    thread_local vector<Node_ids> full_ids;
    thread_local vector<int> output_ids;

    try {
        // Reserve some space to improve performance at smaller
        // sizes
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
            const int id = node->getNodesMap(nodes, full_nodes, full_ids);
            output_ids.push_back(id);
        }

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
        const size_t outputSize{numOutElems * outputSizeofType * nrOutputs};
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

        // Keep node_clones in scope, so that the nodes remain active for later
        // referral in case moddims or Column elimination operations have to
        // take place
        vector<Node_ptr> node_clones;
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
                const auto isModdim{[](const Node_ptr& node) {
                    return node->getOp() == af_moddims_t;
                }};
                for (auto nodeIt{begin(node_clones)}, endIt{end(node_clones)};
                     (nodeIt = find_if(nodeIt, endIt, isModdim)) != endIt;
                     ++nodeIt) {
                    const ModdimNode* mn{
                        static_cast<ModdimNode*>(nodeIt->get())};

                    const auto new_strides{calcStrides(mn->m_new_shape)};
                    const auto isBuffer{
                        [](const Node& ptr) { return ptr.isBuffer(); }};
                    for (NodeIterator<> it{nodeIt->get()},
                         end{NodeIterator<>()};
                         (it = find_if(it, end, isBuffer)) != end; ++it) {
                        BufferNode<T>* buf{static_cast<BufferNode<T>*>(&(*it))};
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
                                              ShiftNode<T>>(outputs,
                                                            node_clones);
            }

            full_nodes.clear();
            for (Node_ptr& node : node_clones) {
                full_nodes.push_back(node.get());
            }
        }

        threadsMgt<dim_t> th(outDims, ndims);
        const dim3 threads{th.genThreads()};
        const dim3 blocks{th.genBlocks(threads, nrInputs, nrOutputs, totalSize,
                                       outputSizeofType)};
        auto ker = getKernel(output_nodes, output_ids, full_nodes, full_ids,
                             is_linear, th.loop0, th.loop1, th.loop2, th.loop3);

        vector<void*> args;
        for (const Node* node : full_nodes) {
            node->setArgs(0, is_linear,
                          [&](int /*id*/, const void* ptr, size_t /*size*/,
                              bool /*is_buffer*/) {
                              args.push_back(const_cast<void*>(ptr));
                          });
        }

        for (auto& out : outputs) { args.push_back(static_cast<void*>(&out)); }

        {
            using namespace arrayfire::cuda::kernel_logger;
            AF_TRACE(
                "Launching : Dims: [{},{},{},{}] Blocks: [{}] "
                "Threads: [{}] threads: {}",
                outDims[0], outDims[1], outDims[2], outDims[3], blocks, threads,
                blocks.x * threads.x * blocks.y * threads.y * blocks.z *
                    threads.z);
        }
        CU_CHECK(cuLaunchKernel(ker, blocks.x, blocks.y, blocks.z, threads.x,
                                threads.y, threads.z, 0, getActiveStream(),
                                args.data(), NULL));
    } catch (...) {
        // Reset the thread local vectors
        nodes.clear();
        output_ids.clear();
        full_nodes.clear();
        full_ids.clear();
        throw;
    }

    // Reset the thread local vectors
    nodes.clear();
    output_ids.clear();
    full_nodes.clear();
    full_ids.clear();
}

template<typename T>
void evalNodes(Param<T> out, Node* node) {
    vector<Param<T>> outputs{out};
    vector<Node*> nodes{node};
    evalNodes(outputs, nodes);
}

template void evalNodes<float>(Param<float> out, Node* node);
template void evalNodes<double>(Param<double> out, Node* node);
template void evalNodes<cfloat>(Param<cfloat> out, Node* node);
template void evalNodes<cdouble>(Param<cdouble> out, Node* node);
template void evalNodes<int>(Param<int> out, Node* node);
template void evalNodes<uint>(Param<uint> out, Node* node);
template void evalNodes<char>(Param<char> out, Node* node);
template void evalNodes<uchar>(Param<uchar> out, Node* node);
template void evalNodes<intl>(Param<intl> out, Node* node);
template void evalNodes<uintl>(Param<uintl> out, Node* node);
template void evalNodes<short>(Param<short> out, Node* node);
template void evalNodes<ushort>(Param<ushort> out, Node* node);
template void evalNodes<half>(Param<half> out, Node* node);

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
}  // namespace cuda
}  // namespace arrayfire
