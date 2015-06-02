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
#include <stdexcept>
#include <copy.hpp>
#include <JIT/Node.hpp>
#include <kernel_headers/jit.hpp>
#include <program.hpp>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <functional>

namespace opencl
{

using JIT::Node;

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;
using std::stringstream;

static string getFuncName(Node *node, bool is_linear, bool *is_double)
{
    node->setId(0);

    stringstream hashName;
    stringstream funcName;

    if (is_linear) {
        funcName << "L_";
    } else {
        funcName << "G_";
    }

    std::string outName = node->getNameStr();
    funcName << outName;

    node->genKerName(funcName);

    string nameStr = funcName.str();
    funcName << nameStr;

    nameStr = nameStr + outName;
    string dblChars = "dDzZ";
    size_t loc = nameStr.find_first_of(dblChars);
    *is_double = (loc != std::string::npos);

    std::hash<std::string> hash_fn;
    hashName << "KER" << hash_fn(funcName.str());
    return hashName.str();
}

static string getKernelString(string funcName, Node *node, bool is_linear)
{
    stringstream kerStream;
    int id = node->getId();

    kerStream << "__kernel void" << "\n";

    kerStream << funcName;
    kerStream << "(" << "\n";

    node->genParams(kerStream);
    kerStream << "__global " << node->getTypeStr() << " *out, KParam oInfo," << "\n";
    kerStream << "uint groups_0, uint groups_1)" << "\n";

    kerStream << "{" << "\n" << "\n";

    if (!is_linear) {

        kerStream << "uint id2 = get_group_id(0) / groups_0;" << "\n";
        kerStream << "uint id3 = get_group_id(1) / groups_1;" << "\n";
        kerStream << "uint groupId_0 = get_group_id(0) - id2 * groups_0;" << "\n";
        kerStream << "uint groupId_1 = get_group_id(1) - id3 * groups_1;" << "\n";
        kerStream << "uint id1 = get_local_id(1) + groupId_1 * get_local_size(1);" << "\n";
        kerStream << "uint id0 = get_local_id(0) + groupId_0 * get_local_size(0);" << "\n";
        kerStream << "\n";

        kerStream << "bool cond = " << "\n";
        kerStream << "id0 < oInfo.dims[0] && " << "\n";
        kerStream << "id1 < oInfo.dims[1] && " << "\n";
        kerStream << "id2 < oInfo.dims[2] && " << "\n";
        kerStream << "id3 < oInfo.dims[3];" << "\n" << "\n";

        kerStream << "if (!cond) return;" << "\n" << "\n";

        kerStream << "int idx = ";
        kerStream << "oInfo.strides[3] * id3 + oInfo.strides[2] * id2 + ";
        kerStream << "oInfo.strides[1] * id1 + id0 + oInfo.offset;" << "\n" << "\n";

    } else {

        kerStream << "uint groupId  = get_group_id(1) * get_num_groups(0) + get_group_id(0);" << "\n";
        kerStream << "uint threadId = get_local_id(0);" << "\n";
        kerStream << "int idx = groupId * get_local_size(0) * get_local_size(1) + threadId;" << "\n";
        kerStream << "if (idx >= oInfo.dims[3] * oInfo.strides[3]) return;" << "\n";
    }

    node->genOffsets(kerStream, is_linear);
    node->genFuncs(kerStream);
    kerStream << "\n";

    kerStream << "out[idx] = val"
           << id << ";"  << "\n";

    kerStream << "}" << "\n";

    return kerStream.str();
}

static Kernel getKernel(Node *node, bool is_linear)
{

    bool is_dbl = false;
    string funcName = getFuncName(node, is_linear, &is_dbl);

    typedef struct {
        Program* prog;
        Kernel* ker;
    } kc_entry_t;

    typedef std::map<string, kc_entry_t> kc_t;
    static kc_t kernelCaches[DeviceManager::MAX_DEVICES];
    int device = getActiveDeviceId();

    kc_t::iterator idx = kernelCaches[device].find(funcName);
    kc_entry_t entry;

    if (idx == kernelCaches[device].end()) {
        string jit_ker = getKernelString(funcName, node, is_linear);

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

void evalNodes(Param &out, Node *node)
{
    try {

        bool is_linear = node->isLinear(out.info.dims);
        Kernel ker = getKernel(node, is_linear);

        uint local_0 = 1;
        uint local_1 = 1;
        uint global_0 = 1;
        uint global_1 = 1;
        uint groups_0 = 1;
        uint groups_1 = 1;

        if (is_linear) {
            local_0 = 256;
            uint out_elements = out.info.dims[3] * out.info.strides[3];
            uint groups = divup(out_elements, local_0);

            global_1 = divup(groups,     1000) * local_1;
            global_0 = divup(groups, global_1) * local_0;

        } else {
            local_0 = 32;
            local_1 = 8;

            groups_0 = divup(out.info.dims[0], local_0);
            groups_1 = divup(out.info.dims[1], local_1);

            global_0 = groups_0 * local_0 * out.info.dims[2];
            global_1 = groups_1 * local_1 * out.info.dims[3];
        }

        NDRange local(local_0, local_1);
        NDRange global(global_0, global_1);

        int args = node->setArgs(ker, 0);
        ker.setArg(args + 0, *out.data);
        ker.setArg(args + 1,  out.info);
        ker.setArg(args + 2,  groups_0);
        ker.setArg(args + 3,  groups_1);

        getQueue().enqueueNDRangeKernel(ker, cl::NullRange, global, local);

    } catch (const cl::Error &ex) {
        CL_TO_AF_ERROR(ex);
    }

}

}
