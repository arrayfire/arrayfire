/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/dispatch.hpp>
#include <common/jit/Node.hpp>
#include <common/jit/NodeIterator.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>
#include <jit/BufferNode.hpp>
#include <jit/ShiftNode.hpp>
#include <kernel_headers/jit.hpp>
#include <program.hpp>
#include <af/dim4.hpp>
#include <af/opencl.h>

#include <functional>
#include <stdexcept>
#include <vector>

using common::Node;
using common::Node_ids;
using common::Node_map_t;
using common::NodeIterator;
using common::requiresGlobalMemoryAccess;
using opencl::jit::BufferNode;
using opencl::jit::ShiftNode;

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::Local;
using cl::NDRange;
using cl::NullRange;
using cl::Program;

using std::hash;
using std::string;
using std::stringstream;
using std::vector;

namespace opencl {

bool equal_shape(const KParam &lhs, const KParam &rhs) {
    return std::equal(lhs.dims, lhs.dims + 4, rhs.dims) &&
           std::equal(lhs.strides, lhs.strides + 4, rhs.strides);
}

static string getFuncName(const vector<Node *> &output_nodes,
                          const vector<const Node *> &full_nodes,
                          const vector<Node_ids> &full_ids, bool is_linear) {
    stringstream hashName;
    stringstream funcName;

    if (is_linear) {
        funcName << "L_";
    } else {
        funcName << "G_";
    }

    for (auto node : output_nodes) { funcName << node->getNameStr() << "_"; }

    for (int i = 0; i < (int)full_nodes.size(); i++) {
        full_nodes[i]->genKerName(funcName, full_ids[i]);
    }

    hash<string> hash_fn;
    hashName << "KER" << hash_fn(funcName.str());
    return hashName.str();
}

static string getKernelString(const string funcName,
                              const vector<const Node *> &full_nodes,
                              const vector<Node_ids> &full_ids,
                              const vector<int> &output_ids, bool is_linear) {
    // Common OpenCL code
    // This part of the code does not change with the kernel.
    static const char *kernelVoid = "__kernel void\n";
    static const char *nonLinearParams =
        "__global KParam* dims, int num_params,\n"
        "__local KParam* params, __local dim_t* block_offsets,\n";
    static const char *dimParams =
        "KParam oInfo, uint groups_0, uint groups_1, uint num_odims";
    static const char *blockStart = "{\n\n";
    static const char *blockEnd   = "\n}\n";

    static const char *linearIndex = R"JIT(
        size_t groupId  = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        size_t threadId = get_local_id(0);
        size_t idx = groupId * get_local_size(0) * get_local_size(1) + threadId;
        if (idx >= oInfo.dims[3] * oInfo.strides[3]) return;
        )JIT";

    static const char *generalIndex = R"JIT(
        int lidx = get_local_id(1) * get_local_size(0) + get_local_id(0);
        if (lidx < num_params) {
            params[lidx] = dims[lidx];
        }
        dim_t id0 = 0, id1 = 0, id2 = 0, id3 = 0;
        if (num_odims > 2) {
            id2 = get_group_id(0) / groups_0;
            id0 = get_group_id(0) - id2 * groups_0;
            id0 = get_local_id(0) + id0 * get_local_size(0);
            if (num_odims > 3) {
                id3 = get_group_id(1) / groups_1;
                id1 = get_group_id(1) - id3 * groups_1;
                id1 = get_local_id(1) + id1 * get_local_size(1);
            } else {
                id1 = get_global_id(1);
            }
        } else {
            id3 = 0;
            id2 = 0;
            id1 = get_global_id(1);
            id0 = get_global_id(0);
        }
        bool cond = id0 < oInfo.dims[0] &&
                    id1 < oInfo.dims[1] &&
                    id2 < oInfo.dims[2] &&
                    id3 < oInfo.dims[3];
        size_t idx = oInfo.strides[3] * id3 +
                     oInfo.strides[2] * id2 +
                     oInfo.strides[1] * id1 +
                     id0 + oInfo.offset;

        if (lidx < num_params) {
            block_offsets[lidx] = (id3 < params[lidx].dims[3]) * params[lidx].strides[3] * id3 +
                                  (id2 < params[lidx].dims[2]) * params[lidx].strides[2] * id2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (!cond) return;
        )JIT";

    stringstream inParamStream;
    stringstream outParamStream;
    stringstream outWriteStream;
    stringstream offsetsStream;
    stringstream opsStream;
    stringstream outrefstream;
    outrefstream << "const Param outref = out" << output_ids[0] << ";\n";

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
    if (!is_linear) { kerStream << nonLinearParams; }
    kerStream << dimParams;
    kerStream << ")\n";
    kerStream << blockStart;
    if (is_linear) {
        kerStream << linearIndex;
    } else {
        kerStream << generalIndex;
    }
    kerStream << offsetsStream.str();
    kerStream << opsStream.str();
    kerStream << outWriteStream.str();
    kerStream << blockEnd;

    return kerStream.str();
}

static Kernel getKernel(const vector<Node *> &output_nodes,
                        const vector<int> &output_ids,
                        const vector<const Node *> &full_nodes,
                        const vector<Node_ids> &full_ids,
                        const bool is_linear) {
    string funcName =
        getFuncName(output_nodes, full_nodes, full_ids, is_linear);
    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, funcName);

    if (entry.prog == 0 && entry.ker == 0) {
        string jit_ker = getKernelString(funcName, full_nodes, full_ids,
                                         output_ids, is_linear);
        saveKernel(funcName, jit_ker, ".cl");
        const char *ker_strs[] = {jit_cl, jit_ker.c_str()};
        const int ker_lens[]   = {jit_cl_len, (int)jit_ker.size()};

        Program prog;
        buildProgram(
            prog, 2, ker_strs, ker_lens,
            isDoubleSupported(device) ? string(" -D USE_DOUBLE") : string(""));

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, funcName.c_str());

        addKernelToCache(device, funcName, entry);
    }

    return *entry.ker;
}

void evalNodes(vector<Param> &outputs, vector<Node *> output_nodes) {
    if (outputs.size() == 0) return;

    // Assume all ouputs are of same size
    // FIXME: Add assert to check if all outputs are same size?
    KParam out_info = outputs[0].info;

    // Use thread local to reuse the memory every time you are here.
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

    vector<KParam> params;
    for (auto &node : output_nodes) {
        int id = node->getNodesMap(nodes, full_nodes, full_ids);
        output_ids.push_back(id);

        NodeIterator<> end_node;
        auto bufit = NodeIterator<>(node);
        while (bufit != end_node) {
            bufit = find_if(bufit, end_node, requiresGlobalMemoryAccess);
            if (bufit != end_node) {
                KParam param;

                // TODO(umar): This is a hack. We need to clean up this API
                // so that the if statement is not necessary
                if (bufit->isBuffer()) {
                    param = static_cast<BufferNode &>(*bufit).getParam();
                } else {
                    param = static_cast<ShiftNode &>(*bufit).getParam();
                }

                auto it = find_if(begin(params), end(params),
                                  [&param](const KParam &p) {
                                      return equal_shape(param, p);
                                  });
                if (it == end(params)) {
                    params.push_back(param);
                    bufit->setParamIndex(params.size() - 1);
                } else {
                    bufit->setParamIndex(distance(begin(params), it));
                }
                ++bufit;
            }
        }
    }

    bool is_linear = true;
    for (const auto &node : full_nodes) {
        is_linear &= node->isLinear(outputs[0].info.dims);
    }

    Kernel ker =
        getKernel(output_nodes, output_ids, full_nodes, full_ids, is_linear);

    uint local_0   = 1;
    uint local_1   = 1;
    uint global_0  = 1;
    uint global_1  = 1;
    uint groups_0  = 1;
    uint groups_1  = 1;
    uint num_odims = 4;

    // CPUs seem to perform better with work group size 1024
    const int work_group_size =
        (getActiveDeviceType() == AFCL_DEVICE_TYPE_CPU) ? 1024 : 256;

    while (num_odims >= 1) {
        if (out_info.dims[num_odims - 1] == 1)
            num_odims--;
        else
            break;
    }

    if (is_linear) {
        local_0           = work_group_size;
        uint out_elements = out_info.dims[3] * out_info.strides[3];
        uint groups       = divup(out_elements, local_0);

        global_1 = divup(groups, 1000) * local_1;
        global_0 = divup(groups, global_1) * local_0;

    } else {
        local_1 = 4;
        local_0 = work_group_size / local_1;

        groups_0 = divup(out_info.dims[0], local_0);
        groups_1 = divup(out_info.dims[1], local_1);

        global_0 = groups_0 * local_0 * out_info.dims[2];
        global_1 = groups_1 * local_1 * out_info.dims[3];
    }

    NDRange local(local_0, local_1);
    NDRange global(global_0, global_1);

    int nargs = 0;
    for (const auto &node : full_nodes) {
        nargs = node->setArgs(nargs, is_linear,
                              [&](int id, const void *ptr, size_t arg_size) {
                                  ker.setArg(id, arg_size, ptr);
                              });
    }

    // Set output parameters
    for (auto output : outputs) {
        ker.setArg(nargs, *(output.data));
        ++nargs;
    }

    size_t smem_bytes = 0;
    int param_count   = 0;
    uptr dparam;
    if (!is_linear) {
        param_count = params.size();
        dparam      = memAlloc<uchar>(params.size() * sizeof(KParam));

        getQueue().enqueueWriteBuffer(*(dparam.get()), CL_FALSE, 0,
                                      params.size() * sizeof(KParam),
                                      params.data());

        ker.setArg(nargs++, *(dparam.get()));
        ker.setArg(nargs++, param_count);
        ker.setArg(nargs++, Local(sizeof(KParam) * params.size()));
        ker.setArg(nargs++, Local(sizeof(dim_t) * params.size()));
    }

    // Set dimensions
    // All outputs are asserted to be of same size
    // Just use the size from the first output
    ker.setArg(nargs + 0, out_info);
    ker.setArg(nargs + 1, groups_0);
    ker.setArg(nargs + 2, groups_1);
    ker.setArg(nargs + 3, num_odims);

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
