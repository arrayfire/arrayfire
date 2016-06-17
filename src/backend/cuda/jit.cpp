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
#include <ptx_headers/arith.hpp>
#include <ptx_headers/logic.hpp>
#include <ptx_headers/exp.hpp>
#include <ptx_headers/numeric.hpp>
#include <ptx_headers/trig.hpp>
#include <ptx_headers/hyper.hpp>
#include <ptx_headers/cast.hpp>
#include <platform.hpp>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <vector>
#include <nvvm.h>
#include <boost/functional/hash.hpp>
#include <boost/scoped_ptr.hpp>

using std::vector;
using boost::scoped_ptr;

namespace cuda
{

using JIT::Node;
using std::string;
using std::stringstream;
using JIT::str_map_t;
using JIT::str_map_iter;

const char *layout64 = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\n\n\n";
const char *layout32 = "target datalayout = \"e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\n\n\n";

const char *triple64 = "target triple = \"nvptx64-unknown-cuda\"\n\n";
const char *triple32 = "target triple = \"nvptx-unknown-cuda\"\n\n";

static string getFuncName(std::vector<Node *> nodes, bool is_linear)
{
    stringstream funcName;
    stringstream hashName;

    if (is_linear) funcName << "L_"; //Kernel Linear
    else           funcName << "G_"; //Kernel General

    int id = 0;

    for (int i = 0; i < (int)nodes.size(); i++) {
        funcName << "[";
        id = nodes[i]->setId(id);
        funcName << nodes[i]->getNameStr();
        nodes[i]->genKerName(funcName);
        funcName << "]";
    }

    boost::hash<std::string> hash_fn;

    hashName << "@KER";
    hashName << hash_fn(funcName.str());
    return hashName.str();
}

static string getKernelString(string funcName, std::vector<Node *> nodes, bool is_linear)
{
    static const char *defineVoid = "define void ";
    static const char *dimParams = "\n"
        "i32 %ostr0, i32 %ostr1, i32 %ostr2, i32 %ostr3,\n"
        "i32 %odim0, i32 %odim1, i32 %odim2, i32 %odim3,\n"
        "i32 %blkx, i32 %blky, i32 %ndims";

    static const char *blockStart = "\n{\n\n"
        "entry:\n\n";
    static const char *blockEnd = "\n\n"
        "ret void\n"
        "\n\n}\n";

    static const char *idAlias = "\n"
        "%tidx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
        "%bdmx = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n"
        "%bidx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
        "%bidy = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()\n"
        "%gdmx = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()\n"
        "\n\n";
    static const char *earlyExit = "\n"
        "end:\n\n"
        "ret void\n";
    static const char *core = "\n"
        "core:\n\n";

    static const char *generalIndex = "\n"
        "%tidy = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()\n"
        "%bdmy = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()\n"
        "%blk_x = alloca i32, align 4\n"
        "%blk_y = alloca i32, align 4\n"
        "%id_3 = alloca i32, align 4\n"
        "%id_2 = alloca i32, align 4\n"
        "store i32 %bidx, i32* %blk_x, align 4\n"
        "store i32 %bidy, i32* %blk_y, align 4\n"
        "store i32 0, i32* %id_2, align 4\n"
        "store i32 0, i32* %id_3, align 4\n"
        "%two = alloca i32, align 4\n"
        "store i32 2, i32* %two, align 4\n"
        "%twoval = load i32* %two, align 4\n"
        "%is34 = icmp sgt i32 %ndims, %twoval\n"
        "br i1 %is34, label %do34, label %do2\n"
        "\ndo34:\n"
        "%id2t = sdiv i32 %bidx, %blkx\n"
        "store i32 %id2t, i32* %id_2, align 4\n"
        "%id2m = mul i32 %id2t, %blkx\n"
        "%blk_xx = sub i32 %bidx, %id2m\n"
        "store i32 %blk_xx, i32* %blk_x, align 4\n"
        "%three = alloca i32, align 4\n"
        "store i32 3, i32* %three, align 4\n"
        "%threeval = load i32* %three, align 4\n"
        "%is4 = icmp sgt i32 %ndims, %threeval\n"
        "br i1 %is4, label %do4, label %do2\n"
        "\ndo4:\n"
        "%id3t = sdiv i32 %bidy, %blky\n"
        "store i32 %id3t, i32* %id_3, align 4\n"
        "%id3m = mul i32 %id3t, %blky\n"
        "%blk_yy = sub i32 %bidy, %id3m\n"
        "store i32 %blk_yy, i32* %blk_y, align 4\n"
        "br label %do2\n"
        "\ndo2:\n"
        "%id2 = load i32* %id_2, align 4\n"
        "%id3 = load i32* %id_3, align 4\n"
        "%tmp_x = load i32* %blk_x, align 4\n"
        "%id0m = mul i32 %tmp_x, %bdmx\n"
        "%id0 = add i32 %tidx, %id0m\n"
        "%tmp_y = load i32* %blk_y, align 4\n"
        "%id1m = mul i32 %tmp_y, %bdmy\n"
        "%id1 = add i32 %tidy, %id1m\n"
        "\n\n"
        "%off3o = mul i32 %id3, %ostr3\n"
        "%off2o = mul i32 %id2, %ostr2\n"
        "%off1o = mul i32 %id1, %ostr1\n"
        "%off23o = add i32 %off3o, %off2o\n"
        "%off123o = add i32 %off23o, %off1o\n"
        "%idxa = add i32 %off123o, %id0\n"
        "%idx = sext i32 %idxa to i64\n"
        "\n\n"
        "%cmp3 = icmp slt i32 %id3, %odim3\n"
        "%cmp2 = icmp slt i32 %id2, %odim2\n"
        "%cmp1 = icmp slt i32 %id1, %odim1\n"
        "%cmp0 = icmp slt i32 %id0, %odim0\n"
        "br i1 %cmp3, label %check2, label %end\n"
        "\ncheck2:\n"
        "br i1 %cmp2, label %check1, label %end\n"
        "\ncheck1:\n"
        "br i1 %cmp1, label %check0, label %end\n"
        "\ncheck0:\n"
        "br i1 %cmp0, label %core, label %end\n";

    static const char *linearIndex = "\n"
        "%boff = mul i32 %bidy, %gdmx\n"
        "%bid  = add i32 %boff, %bidx\n"
        "%goff = mul i32 %bid , %bdmx\n"
        "%gid  = add i32 %goff ,%tidx\n"
        "%idx  = sext i32 %gid to i64\n"
        "%el1  = mul i32 %odim0, %odim1\n"
        "%el2  = mul i32 %el1  , %odim2\n"
        "%el3  = mul i32 %el2  , %odim3\n"
        "%cmp0 = icmp slt i32 %gid, %el3\n"
        "br i1 %cmp0, label %core, label %end\n";

    static const char *functionLoad = "\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() nounwind readnone\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() nounwind readnone\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() nounwind readnone\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() nounwind readnone\n"
        "\n";


    stringstream kerStream;
    stringstream inAnnStream;
    stringstream outAnnStream;
    stringstream inParamStream;
    stringstream outParamStream;
    stringstream funcBodyStream;
    stringstream offsetsStream;
    stringstream outWriteStream;
    str_map_t declStrs;

    for (int i = 0; i < (int)nodes.size(); i++) {
        std::string outTypeStr = nodes[i]->getTypeStr();
        int id = nodes[i]->getId();

        nodes[i]->genParams(inParamStream, inAnnStream, is_linear);
        outParamStream << outTypeStr << "* %out" << id << ",\n";
        nodes[i]->genOffsets(offsetsStream, is_linear);
        nodes[i]->genFuncs(funcBodyStream, declStrs, is_linear);

        outWriteStream << "%outIdx" << id
                       << "= getelementptr inbounds "
                       << outTypeStr
                       << "* %out" << id
                       << ", i64 %idx\n";
        outWriteStream << "store "
                       << outTypeStr
                       << " %val" << id << ", "
                       << outTypeStr
                       << "* %outIdx" << id << "\n";
        outAnnStream << outTypeStr << "*,\n";
    }

    if (sizeof(void *) == 8) {
        kerStream << layout64;
        kerStream << triple64;
    } else {
        kerStream << layout32;
        kerStream << triple32;
    }

    const char *index = is_linear ? linearIndex : generalIndex;

    kerStream << defineVoid
              << funcName
              << " (\n"
              << inParamStream.str()
              << outParamStream.str()
              << dimParams
              << " )\n"
              << blockStart
              << idAlias
              << index
              << earlyExit
              << core
              << offsetsStream.str()
              << funcBodyStream.str()
              << outWriteStream.str()
              << blockEnd;

    for(str_map_iter iterator = declStrs.begin(); iterator != declStrs.end(); iterator++) {
        kerStream << iterator->first << "\n";
    }
    kerStream << functionLoad;

    kerStream << "!nvvm.annotations = !{!1}\n"
              << "!1 = metadata !{void (\n"
              << inAnnStream.str()
              << outAnnStream.str()
              << "i32, i32, i32, i32,\n"
              << "i32, i32, i32, i32,\n"
              << "i32, i32, i32\n"
              << ")* " << funcName << ",\n "
              << "metadata !\"kernel\", i32 1}\n";

    return kerStream.str();
}

#define NVVM_CHECK(fn, msg) do {                \
        nvvmResult res = fn;                    \
        if (res == NVVM_SUCCESS) break;         \
        char nvvm_err_msg[1024];                \
        snprintf(nvvm_err_msg,                    \
                 sizeof(nvvm_err_msg),          \
                 "NVVM Error (%d): %s\n",       \
                 (int)(res), msg);              \
        AF_ERROR(nvvm_err_msg,                  \
                 AF_ERR_INTERNAL);              \
                                                \
    } while(0)

static char *irToPtx(string IR, size_t *ptx_size)
{
    nvvmProgram prog;

    NVVM_CHECK(nvvmCreateProgram(&prog), "Failed to create program");

    NVVM_CHECK(nvvmAddModuleToProgram(prog, IR.c_str(), IR.size(), "generated kernel"),
               "Failed to add module");

//#ifdef NDEBUG
#if 0
    NVVM_CHECK(nvvmCompileProgram(prog, 0, NULL), "Failed to compile program");
#else
    nvvmResult comp_res = nvvmCompileProgram(prog, 0, NULL);
    if (comp_res != NVVM_SUCCESS) {
        size_t log_size = 0;
        nvvmGetProgramLogSize(prog, &log_size);
        printf("%ld, %zu\n", IR.size(), log_size);
        scoped_ptr<char> log(new char[log_size]);
        nvvmGetProgramLog(prog, log.get());
        printf("LOG:\n%s\n%s", log.get(), IR.c_str());
        NVVM_CHECK(comp_res, "Failed to compile program");
    }
#endif

    NVVM_CHECK(nvvmGetCompiledResultSize(prog, ptx_size), "Can not get ptx size");

    char *ptx = new char[*ptx_size];
    NVVM_CHECK(nvvmGetCompiledResult(prog, ptx), "Can not get ptx from NVVM IR");

    NVVM_CHECK(nvvmDestroyProgram(&prog), "Failed to destroy program");
    return ptx;
}

typedef struct {
    CUmodule prog;
    CUfunction ker;
} kc_entry_t;


const size_t size = 1024;
char linkInfo[size];
char linkError[size];

#ifndef NDEBUG
#define CU_CHECK(fn) do {                       \
        CUresult res = fn;                      \
        if (res == CUDA_SUCCESS) break;         \
        char cu_err_msg[1024];                  \
        snprintf(cu_err_msg,                    \
                 sizeof(cu_err_msg),            \
                 "CU Error (%d)\n%s\n",         \
                 (int)(res), linkError);        \
        AF_ERROR(cu_err_msg,                    \
                 AF_ERR_INTERNAL);              \
    } while(0)
#else
#define CU_CHECK(fn) do {                       \
        CUresult res = fn;                      \
        if (res == CUDA_SUCCESS) break;         \
        char cu_err_msg[1024];                  \
        snprintf(cu_err_msg,                    \
                 sizeof(cu_err_msg),            \
                 "CU Error (%d)\n",             \
                 (int)(res));                   \
        AF_ERROR(cu_err_msg,                    \
                 AF_ERR_INTERNAL);              \
    } while(0)
#endif

static kc_entry_t compileKernel(const char *ker_name, string jit_ker)
{
    size_t ptx_size;
    scoped_ptr<const char> ptx(irToPtx(jit_ker, &ptx_size));

    CUlinkState linkState;

    linkInfo[0] = 0;
    linkError[0] = 0;

    CUjit_option linkOptions[] = {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_LOG_VERBOSE
    };

    void *linkOptionValues[] = {
        linkInfo,
        reinterpret_cast<void*>(1024),
        linkError,
        reinterpret_cast<void*>(1024),
        reinterpret_cast<void*>(1)
    };

    CU_CHECK(cuLinkCreate(5, linkOptions, linkOptionValues, &linkState));
    CU_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx.get(),
                           ptx_size, ker_name, 0, NULL, NULL));

    CU_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)arith_ptx,
                           arith_ptx_len, "arith", 0, NULL, NULL));

    CU_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)cast_ptx,
                           cast_ptx_len, "cast", 0, NULL, NULL));

    CU_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)exp_ptx,
                           exp_ptx_len, "exp", 0, NULL, NULL));

    CU_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)hyper_ptx,
                           hyper_ptx_len, "hyper", 0, NULL, NULL));

    CU_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)logic_ptx,
                           logic_ptx_len, "logic", 0, NULL, NULL));

    CU_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)numeric_ptx,
                           numeric_ptx_len, "numeric", 0, NULL, NULL));

    CU_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)trig_ptx,
                           trig_ptx_len, "trig", 0, NULL, NULL));

    void *cubin;
    size_t cubinSize;

    CUmodule module;
    CUfunction kernel;

    CU_CHECK(cuLinkComplete(linkState, &cubin, &cubinSize));
    CU_CHECK(cuModuleLoadDataEx(&module, cubin, 0, 0, 0));
    CU_CHECK(cuModuleGetFunction(&kernel, module, ker_name + 1));

    kc_entry_t entry = {module, kernel};

    return entry;
}

static CUfunction getKernel(std::vector<Node *> nodes, bool is_linear)
{

    string funcName = getFuncName(nodes, is_linear);

    typedef std::map<string, kc_entry_t> kc_t;
    static kc_t kernelCaches[DeviceManager::MAX_DEVICES];
    int device = getActiveDeviceId();

    kc_t::iterator idx = kernelCaches[device].find(funcName);
    kc_entry_t entry = {NULL, NULL};

    if (idx == kernelCaches[device].end()) {
        string jit_ker = getKernelString(funcName, nodes, is_linear);
        entry = compileKernel(funcName.c_str(), jit_ker);
        kernelCaches[device][funcName] = entry;
    } else {
        entry = idx->second;
    }

    return entry.ker;
}

template<typename T>
void evalNodes(std::vector<Param<T> >&outputs, std::vector<Node *> nodes)
{
    int num_outputs = (int)outputs.size();

    if (num_outputs == 0) return;

    bool is_linear = true;

    for (int i = 0; i < num_outputs; i++) {
        is_linear &= nodes[i]->isLinear(outputs[0].dims);
    }

    CUfunction ker = getKernel(nodes, is_linear);

    int threads_x = 1, threads_y = 1;
    int blocks_x_ = 1, blocks_y_ = 1;
    int blocks_x  = 1, blocks_y = 1;

    int num_odims = 4;

    while (num_odims >= 1) {
        if (outputs[0].dims[num_odims - 1] == 1) num_odims--;
        else break;
    }

    if (is_linear) {

        threads_x = 256;
        threads_y =  1;

        int blocks = divup((outputs[0].dims[0] *
                            outputs[0].dims[1] *
                            outputs[0].dims[2] *
                            outputs[0].dims[3]), threads_x);

        blocks_y_ = divup(blocks, 65535);
        blocks_x_ = divup(blocks, blocks_y_);

        blocks_x = blocks_x_;
        blocks_y = blocks_y_;

    } else {

        threads_x = 32;
        threads_y =  8;

        blocks_x_ = divup(outputs[0].dims[0], threads_x);
        blocks_y_ = divup(outputs[0].dims[1], threads_y);

        blocks_x = blocks_x_ * outputs[0].dims[2];
        blocks_y = blocks_y_ * outputs[0].dims[3];
    }

    vector<void *> args;

    for (int i = 0; i < num_outputs; i++) {
        nodes[i]->setArgs(args, is_linear);
    }

    int strides[] = {(int)outputs[0].strides[0],
                     (int)outputs[0].strides[1],
                     (int)outputs[0].strides[2],
                     (int)outputs[0].strides[3]};

    int dims[] = {(int)outputs[0].dims[0],
                  (int)outputs[0].dims[1],
                  (int)outputs[0].dims[2],
                  (int)outputs[0].dims[3]};

    for (int i = 0; i < num_outputs; i++) {
        args.push_back(&outputs[i].ptr);
    }

    for (int i = 0; i < 4; i++) args.push_back((void *)(strides + i));
    for (int i = 0; i < 4; i++) args.push_back((void *)(dims + i));

    args.push_back((void *)&blocks_x_);
    args.push_back((void *)&blocks_y_);
    args.push_back((void *)&num_odims);

    CU_CHECK(cuLaunchKernel(ker,
                            blocks_x,
                            blocks_y,
                            1,
                            threads_x,
                            threads_y,
                            1,
                            0,
                            getStream(getActiveDeviceId()),
                            &args.front(),
                            NULL));
}

template<typename T>
void evalNodes(Param<T> &out, Node *node)
{
    std::vector<Param<T> > outputs;
    std::vector<Node *> nodes;

    outputs.push_back(out);
    nodes.push_back(node);
    evalNodes(outputs, nodes);
    return;
}

template void evalNodes<float  >(Param<float  > &out, Node *node);
template void evalNodes<double >(Param<double > &out, Node *node);
template void evalNodes<cfloat >(Param<cfloat > &out, Node *node);
template void evalNodes<cdouble>(Param<cdouble> &out, Node *node);
template void evalNodes<int    >(Param<int    > &out, Node *node);
template void evalNodes<uint   >(Param<uint   > &out, Node *node);
template void evalNodes<char   >(Param<char   > &out, Node *node);
template void evalNodes<uchar  >(Param<uchar  > &out, Node *node);
template void evalNodes<intl   >(Param<intl   > &out, Node *node);
template void evalNodes<uintl  >(Param<uintl  > &out, Node *node);
template void evalNodes<short  >(Param<short  > &out, Node *node);
template void evalNodes<ushort >(Param<ushort > &out, Node *node);

template void evalNodes<float  >(std::vector<Param<float  > > &out, std::vector<Node *> node);
template void evalNodes<double >(std::vector<Param<double > > &out, std::vector<Node *> node);
template void evalNodes<cfloat >(std::vector<Param<cfloat > > &out, std::vector<Node *> node);
template void evalNodes<cdouble>(std::vector<Param<cdouble> > &out, std::vector<Node *> node);
template void evalNodes<int    >(std::vector<Param<int    > > &out, std::vector<Node *> node);
template void evalNodes<uint   >(std::vector<Param<uint   > > &out, std::vector<Node *> node);
template void evalNodes<char   >(std::vector<Param<char   > > &out, std::vector<Node *> node);
template void evalNodes<uchar  >(std::vector<Param<uchar  > > &out, std::vector<Node *> node);
template void evalNodes<intl   >(std::vector<Param<intl   > > &out, std::vector<Node *> node);
template void evalNodes<uintl  >(std::vector<Param<uintl  > > &out, std::vector<Node *> node);
template void evalNodes<short  >(std::vector<Param<short  > > &out, std::vector<Node *> node);
template void evalNodes<ushort >(std::vector<Param<ushort > > &out, std::vector<Node *> node);



}
