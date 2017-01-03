/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <map>
#include <vector>
#include <stdexcept>
#include <copy.hpp>
#include <JIT/Node.hpp>
#include <kernel_headers/jit.hpp>
#include <program.hpp>
#include <cache.hpp>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <functional>
#include <af/opencl.h>

namespace opencl
{

using JIT::Node;

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;
using std::stringstream;

static string getFuncName(std::vector<Node *> nodes, bool is_linear, bool *is_double)
{
    stringstream hashName;
    stringstream funcName;

    if (is_linear) {
        funcName << "L_";
    } else {
        funcName << "G_";
    }

    int id = 0;
    for (auto node :  nodes) {
        funcName << "[";
        id = node->setId(id);
        funcName << node->getNameStr();
        node->genKerName(funcName);
        funcName << "]";
    }

    string nameStr = funcName.str();
    string dblChars = "dDzZ";
    size_t loc = nameStr.find_first_of(dblChars);
    *is_double = (loc != std::string::npos);

    std::hash<std::string> hash_fn;
    hashName << "KER" << hash_fn(funcName.str());
    return hashName.str();
}

static string getKernelString(string funcName, std::vector<Node *> nodes, bool is_linear)
{

    // Common OpenCL code
    // This part of the code does not change with the kernel.

    static const char *kernelVoid =  "__kernel void\n";
    static const char *dimParams = "KParam oInfo, uint groups_0, uint groups_1, uint num_odims";
    static const char *blockStart = "{\n\n";
    static const char *blockEnd = "\n\n}";

    static const char *linearIndex = "\n"
        "uint groupId  = get_group_id(1) * get_num_groups(0) + get_group_id(0);\n"
        "uint threadId = get_local_id(0);\n"
        "int idx = groupId * get_local_size(0) * get_local_size(1) + threadId;\n"
        "if (idx >= oInfo.dims[3] * oInfo.strides[3]) return;\n";

    static const char *generalIndex = "\n"
        "uint id0 = 0, id1 = 0, id2 = 0, id3 = 0;\n"
        "if (num_odims > 2) {\n"
        "id2 = get_group_id(0) / groups_0;\n"
        "id0 = get_group_id(0) - id2 * groups_0;\n"
        "id0 = get_local_id(0) + id0 * get_local_size(0);\n"
        "if (num_odims > 3) {\n"
        "id3 = get_group_id(1) / groups_1;\n"
        "id1 = get_group_id(1) - id3 * groups_1;\n"
        "id1 = get_local_id(1) + id1 * get_local_size(1);\n"
        "} else {\n"
        "id1 = get_global_id(1);\n"
        "}\n"
        " } else {\n"
        "id3 = 0;\n"
        "id2 = 0;\n"
        "id1 = get_global_id(1);\n"
        "id0 = get_global_id(0);\n"
        "}\n"
        "bool cond = \n"
        "id0 < oInfo.dims[0] && \n"
        "id1 < oInfo.dims[1] && \n"
        "id2 < oInfo.dims[2] && \n"
        "id3 < oInfo.dims[3];\n\n"
        "if (!cond) return;\n\n"
        "int idx = "
        "oInfo.strides[3] * id3 + oInfo.strides[2] * id2 + "
        "oInfo.strides[1] * id1 + id0 + oInfo.offset;\n\n";


    stringstream inParamStream;
    stringstream outParamStream;
    stringstream outWriteStream;
    stringstream offsetsStream;
    stringstream opsStream;

    int count  = 0;

    for (auto node : nodes) {
        int id = node->getId();
        node->genParams(inParamStream, is_linear);
        outParamStream << "__global " << node->getTypeStr() << " *out" << id << ", \n";
        outWriteStream << "out" << id << "[idx] = " << "val" << id << ";\n";
        node->genOffsets(offsetsStream, is_linear);
        node->genFuncs(opsStream);
        opsStream << "//" << ++count << std::endl << std::endl;
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

static Kernel getKernel(std::vector<Node *> nodes, bool is_linear)
{

    bool is_dbl = false;
    string funcName = getFuncName(nodes, is_linear, &is_dbl);
    int device = getActiveDeviceId();

    kc_t::iterator idx = kernelCaches[device].find(funcName);
    kc_entry_t entry;

    if (idx == kernelCaches[device].end()) {
        string jit_ker = getKernelString(funcName, nodes, is_linear);

        const char *ker_strs[] = {jit_cl, jit_ker.c_str()};
        const int ker_lens[] = {jit_cl_len, (int)jit_ker.size()};
        cl::Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, is_dbl ? string(" -D USE_DOUBLE") :  string(""));
        entry.prog = new cl::Program(prog);
        entry.ker = new Kernel(*entry.prog, funcName.c_str());

        kernelCaches[device][funcName] = entry;
    } else {
        entry = idx->second;
    }

    return *entry.ker;
}

void evalNodes(std::vector<Param> &outputs, std::vector<Node *> nodes)
{
    if (outputs.size() == 0) return;

    // Assume all ouputs are of same size
    //FIXME: Add assert to check if all outputs are same size?
    KParam out_info = outputs[0].info;

    // Verify if all ASTs hold Linear Arrays
    bool is_linear = true;
    for (auto node : nodes) {
        is_linear &= node->isLinear(out_info.dims);
    }

    Kernel ker = getKernel(nodes, is_linear);

    uint local_0 = 1;
    uint local_1 = 1;
    uint global_0 = 1;
    uint global_1 = 1;
    uint groups_0 = 1;
    uint groups_1 = 1;
    uint num_odims = 4;

    // CPUs seem to perform better with work group size 1024
    const int work_group_size = (getActiveDeviceType() == AFCL_DEVICE_TYPE_CPU) ? 1024 : 256;

    while (num_odims >= 1) {
        if (out_info.dims[num_odims - 1] == 1) num_odims--;
        else break;
    }

    if (is_linear) {
        local_0 = work_group_size;
        uint out_elements = out_info.dims[3] * out_info.strides[3];
        uint groups = divup(out_elements, local_0);

        global_1 = divup(groups,     1000) * local_1;
        global_0 = divup(groups, global_1) * local_0;

    } else {
        local_1 =  4;
        local_0 = work_group_size / local_1;

        groups_0 = divup(out_info.dims[0], local_0);
        groups_1 = divup(out_info.dims[1], local_1);

        global_0 = groups_0 * local_0 * out_info.dims[2];
        global_1 = groups_1 * local_1 * out_info.dims[3];
    }

    NDRange local(local_0, local_1);
    NDRange global(global_0, global_1);

    int args = 0;
    for (auto node : nodes) {
        args = node->setArgs(ker, args, is_linear);
    }

    // Set output parameters
    for (auto output : outputs) {
        ker.setArg(args, *(output.data));
        ++args;
    }

    // Set dimensions
    // All outputs are asserted to be of same size
    // Just use the size from the first output
    ker.setArg(args + 0,  out_info);
    ker.setArg(args + 1,  groups_0);
    ker.setArg(args + 2,  groups_1);
    ker.setArg(args + 3,  num_odims);

    getQueue().enqueueNDRangeKernel(ker, cl::NullRange, global, local);

    for (auto node : nodes) {
        node->resetFlags();
    }
}

void evalNodes(Param &out, Node *node)
{
    std::vector<Param>  outputs{out};
    std::vector<Node *> nodes{node};
    return evalNodes(outputs, nodes);
}

}
