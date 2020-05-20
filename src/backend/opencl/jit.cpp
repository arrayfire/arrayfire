/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/compile_kernel.hpp>
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
#include <vector>

using common::compileKernel;
using common::getFuncName;
using common::Node;
using common::Node_ids;
using common::Node_map_t;

using cl::Kernel;
using cl::NDRange;
using cl::NullRange;

using std::string;
using std::stringstream;
using std::vector;

namespace opencl {

static string getKernelString(const string &funcName,
                              const vector<Node *> &full_nodes,
                              const vector<Node_ids> &full_ids,
                              const vector<int> &output_ids, bool is_linear) {
    // Common OpenCL code
    // This part of the code does not change with the kernel.

    static const char *kernelVoid = "__kernel void\n";
    static const char *dimParams =
        "KParam oInfo, uint groups_0, uint groups_1, uint num_odims";
    static const char *blockStart = "{\n\n";
    static const char *blockEnd   = "\n\n}";

    static const char *linearIndex = R"JIT(
        uint groupId  = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        uint threadId = get_local_id(0);
        int idx = groupId * get_local_size(0) * get_local_size(1) + threadId;
        if (idx >= oInfo.dims[3] * oInfo.strides[3]) return;
        )JIT";

    static const char *generalIndex = R"JIT(
        uint id0 = 0, id1 = 0, id2 = 0, id3 = 0;
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
        if (!cond) return;
        int idx = oInfo.strides[3] * id3 +
                  oInfo.strides[2] * id2 +
                  oInfo.strides[1] * id1 +
                  id0 + oInfo.offset;
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

static cl::Kernel getKernel(const vector<Node *> &output_nodes,
                            const vector<int> &output_ids,
                            const vector<Node *> &full_nodes,
                            const vector<Node_ids> &full_ids,
                            const bool is_linear) {
    string funcName =
        getFuncName(output_nodes, full_nodes, full_ids, is_linear);

    // Lookup kernel cache for a kernel with the key funcName.
    // A forward lookup helps avoid recompiling the jit source
    // generated from identical jit-trees. It also enables us
    // with a way to save jit kernels to disk only once
    auto entry = common::findKernel(getActiveDeviceId(), funcName);

    if (entry.getModule() == nullptr || entry.getKernel() == nullptr) {
        static const string jit(jit_cl, jit_cl_len);

        string jitKer = getKernelString(funcName, full_nodes, full_ids,
                                        output_ids, is_linear);
        int device    = getActiveDeviceId();
        vector<string> options;
        if (isDoubleSupported(device)) {
            options.emplace_back(DefineKey(USE_DOUBLE));
        }
        if (isHalfSupported(device)) {
            options.emplace_back(DefineKey(USE_HALF));
        }

        saveKernel(funcName, jitKer, ".cl");

        // getKernel compiles and caches the kernel handle for later lookup
        entry = common::getKernel(funcName, {jit, jitKer}, {}, options, true);
    }
    return *entry.getKernel();
}

void evalNodes(vector<Param> &outputs, const vector<Node *> &output_nodes) {
    if (outputs.empty()) { return; }

    // Assume all ouputs are of same size
    // FIXME: Add assert to check if all outputs are same size?
    KParam out_info = outputs[0].info;

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
    for (auto node : full_nodes) {
        is_linear &= node->isLinear(outputs[0].info.dims);
    }

    cl::Kernel ker =
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
        if (out_info.dims[num_odims - 1] == 1) {
            num_odims--;
        } else {
            break;
        }
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
