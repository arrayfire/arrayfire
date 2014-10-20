#include <af/dim4.hpp>
#include <Array.hpp>
#include <map>
#include <iostream>
#include <stdexcept>
#include <copy.hpp>
#include <JIT/Node.hpp>
#include <ptx_headers/arith.hpp>
#include <ptx_headers/logic.hpp>
#include <ptx_headers/exp.hpp>
#include <ptx_headers/numeric.hpp>
#include <ptx_headers/trig.hpp>
#include <ptx_headers/hyper.hpp>
#include <kernel/elwise.hpp>
#include <platform.hpp>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <math.hpp>

namespace cuda
{

using JIT::Node;
using std::string;
using std::stringstream;

const char *layout64 = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\n\n\n";
const char *layout32 = "target datalayout = \"e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\n\n\n";

static string getFuncName(Node *node)
{
    stringstream funcName;
    funcName << "@K_";
    node->genKerName(funcName, false);
    funcName << "_";
    node->genKerName(funcName, true);
    return funcName.str();
}

static string getKernelString(string funcName, Node *node)
{
    stringstream kerStream;
    stringstream declStream;

    int id = node->setId(0) - 1;

    if (sizeof(void *) == 8) {
        kerStream << layout64;
    } else {
        kerStream << layout32;
    }

    kerStream << "define void " << funcName << " (" << std::endl;
    node->genParams(kerStream);
    kerStream << node->getTypeStr() <<"* %out,\n"
              << "i32 %ostr0, i32 %ostr1, i32 %ostr2, i32 %ostr3,\n"
              << "i32 %odim0, i32 %odim1, i32 %odim2, i32 %odim3,\n"
              << "i32 %blkx, i32 %blky) {"
              << "\n\n";

    kerStream << "entry:\n\n";
    kerStream << "%tidx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n";
    kerStream << "%tidy = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()\n";
    kerStream << "%bdmx = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n";
    kerStream << "%bdmy = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()\n";
    kerStream << "%bidx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n";
    kerStream << "%bidy = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()\n";
    kerStream << "\n\n";
    kerStream << "%id2 = sdiv i32 %bidx, %blkx\n";
    kerStream << "%id3 = sdiv i32 %bidy, %blky\n";
    kerStream << "%id2m = smul i32 %id2, %blkx\n";
    kerStream << "%id3m = smul i32 %id3, %blky\n";
    kerStream << "%blk_x = sub i32 %bidx, %id2m\n";
    kerStream << "%blk_y = sub i32 %bidy, %id3m\n";
    kerStream << "%id0m = smul i32 %blk_x, %bdmx\n";
    kerStream << "%id1m = smul i32 %blk_y, %bdmy\n";
    kerStream << "%id0 = add i32 %tidx, %id0m\n";
    kerStream << "%id1 = add i32 %tidy, %id1m\n";
    kerStream << "\n\n";

    node->genOffsets(kerStream);

    kerStream << "%off3o = mul i32 %id3, ostr3\n";
    kerStream << "%off2o = mul i32 %id2, ostr2\n";
    kerStream << "%off1o = mul i32 %id1, ostr1\n";
    kerStream << "%off23o = add i32 %off3o, %off2o\n";
    kerStream << "%idxa = add i32 %off23o, %off1o\n";
    kerStream << "%idx = sext i32 %idxa to i64\n";
    kerStream << "\n\n";

    node->genFuncs(kerStream, declStream);

    kerStream << "%outIdx = getelementptr inbounds i32* %out, i64 %idx\n";
    kerStream << "store "
              << node->getTypeStr()
              << " %val" << id << " "
              << node->getTypeStr()
              << "* %idx, align 4\n";

    kerStream << "\n}\n\n";

    kerStream << declStream.str() << "\n";

    kerStream
        << "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() nounwind readnone\n";

    kerStream << "\n";

    kerStream << "!nvvm.annotations = !{!1}\n"
              << "!1 = metadata !{void (i32*)*"
              << " " << funcName << " "
              << "metadata !\"kernel\", i32 1}\n";

    return kerStream.str();
}

typedef struct {
    CUmodule prog;
    CUfunction ker;
} kc_entry_t;

static CUfunction getKernel(Node *node)
{

    string funcName = getFuncName(node);

    typedef std::map<string, kc_entry_t> kc_t;
    static kc_t kernelCaches[DeviceManager::MAX_DEVICES];
    int device = getActiveDeviceId();

    kc_t::iterator idx = kernelCaches[device].find(funcName);
    kc_entry_t entry = {NULL, NULL};

    if (idx == kernelCaches[device].end()) {
        string jit_ker = getKernelString(funcName, node);
        kernelCaches[device][funcName] = entry;
    } else {
        entry = idx->second;
    }

    return entry.ker;
}

template<typename T>
void evalNodes(Param<T> &out, Node *node)
{
    getKernel(node);

    kernel::set((T *)out.ptr, scalar<T>(0), out.strides[3] * out.dims[3]);
}

template void evalNodes<float  >(Param<float  > &out, Node *node);
template void evalNodes<double >(Param<double > &out, Node *node);
template void evalNodes<cfloat >(Param<cfloat > &out, Node *node);
template void evalNodes<cdouble>(Param<cdouble> &out, Node *node);
template void evalNodes<int    >(Param<int    > &out, Node *node);
template void evalNodes<uint   >(Param<uint   > &out, Node *node);
template void evalNodes<char   >(Param<char   > &out, Node *node);
template void evalNodes<uchar  >(Param<uchar  > &out, Node *node);


}
