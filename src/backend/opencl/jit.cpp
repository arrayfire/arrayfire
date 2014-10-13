#include <af/dim4.hpp>
#include <Array.hpp>
#include <map>
#include <iostream>
#include <stdexcept>
#include <copy.hpp>
#include <JIT/Node.hpp>
#include <kernel_headers/jit.hpp>
#include <program.hpp>

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

static string getFuncName(Node *node)
{
    stringstream funcName;
    funcName << "K_";
    node->genKerName(funcName, false);
    funcName << "_";
    node->genKerName(funcName, true);
    return funcName.str();
}

static string getKernelString(string funcName, Node *node)
{
    stringstream kerStream;
    int id = node->setId(0) - 1;

    kerStream << "__kernel void" << std::endl;

    kerStream << funcName;
    kerStream << "(" << std::endl;

    node->genParams(kerStream);
    kerStream << "__global " << node->getTypeStr() << " *out, KParam oInfo," << std::endl;
    kerStream << "uint groups_0, uint groups_1)" << std::endl;

    kerStream << "{" << std::endl << std::endl;

    kerStream << "uint id2 = get_group_id(0) / groups_0;" << std::endl;
    kerStream << "uint id3 = get_group_id(1) / groups_1;" << std::endl;
    kerStream << "uint groupId_0 = get_group_id(0) - id2 * groups_0;" << std::endl;
    kerStream << "uint groupId_1 = get_group_id(1) - id3 * groups_1;" << std::endl;
    kerStream << "uint id1 = get_local_id(1) + groupId_1 * get_local_size(1);" << std::endl;
    kerStream << "uint id0 = get_local_id(0) + groupId_0 * get_local_size(0);" << std::endl;
    kerStream << std::endl;

    kerStream << "bool cond = " << std::endl;
    kerStream << "id0 < oInfo.dims[0] && " << std::endl;
    kerStream << "id1 < oInfo.dims[1] && " << std::endl;
    kerStream << "id2 < oInfo.dims[2] && " << std::endl;
    kerStream << "id3 < oInfo.dims[3];" << std::endl << std::endl;

    kerStream << "if (!cond) return;" << std::endl << std::endl;

    node->genOffsets(kerStream);
    kerStream << "int idx = ";
    kerStream << "oInfo.strides[3] * id3 + oInfo.strides[2] * id2 + ";
    kerStream << "oInfo.strides[1] * id1 + id0 + oInfo.offset;" << std::endl << std::endl;

    node->genFuncs(kerStream);
    kerStream << std::endl;

    kerStream << "out[idx] = val"
           << id << ";"  << std::endl;

    kerStream << "}" << std::endl;

    return kerStream.str();
}

static Kernel getKernel(Node *node)
{

    string funcName = getFuncName(node);

    typedef struct {
        Program prog;
        Kernel ker;
    } kc_entry_t;

    typedef std::map<string, kc_entry_t> kc_t;
    static kc_t kernelCaches[DeviceManager::MAX_DEVICES];
    int device = getActiveDeviceId();

    kc_t::iterator idx = kernelCaches[device].find(funcName);
    kc_entry_t entry;

    if (idx == kernelCaches[device].end()) {
        string jit_ker = getKernelString(funcName, node);

        const char *ker_strs[] = {jit_cl, jit_ker.c_str()};
        const int ker_lens[] = {jit_cl_len, (int)jit_ker.size()};

        buildProgram(entry.prog, 2, ker_strs, ker_lens, string(""));
        entry.ker = Kernel(entry.prog, funcName.c_str());

        kernelCaches[device][funcName] = entry;
    } else {
        entry = idx->second;
    }

    return entry.ker;
}

void evalNodes(Param &out, Node *node)
{
    try {

        Kernel ker = getKernel(node);

        NDRange local(32, 8);

        uint groups_0 = divup(out.info.dims[0], local[0]);
        uint groups_1 = divup(out.info.dims[1], local[1]);

        NDRange global(groups_0 * local[0] * out.info.dims[2],
                       groups_1 * local[1] * out.info.dims[3]);

        int args = node->setArgs(ker, 0);
        ker.setArg(args + 0, out.data);
        ker.setArg(args + 1, out.info);
        ker.setArg(args + 2, groups_0);
        ker.setArg(args + 3, groups_1);

        getQueue().enqueueNDRangeKernel(ker, cl::NullRange, global, local);

    } catch (cl::Error ex) {
        CL_TO_AF_ERROR(ex);
    }

}

}
