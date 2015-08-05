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

static string getFuncName(Node *node, bool is_linear)
{
    node->setId(0);

    stringstream funcName;
    stringstream hashName;

    if (is_linear) funcName << "L_"; //Kernel Linear
    else           funcName << "G_"; //Kernel General

    funcName << node->getNameStr();
    node->genKerName(funcName);
    funcName.str();

    boost::hash<std::string> hash_fn;

    hashName << "@KER";
    hashName << hash_fn(funcName.str());
    return hashName.str();
}

static string getKernelString(string funcName, Node *node, bool is_linear)
{
    stringstream kerStream;
    stringstream annStream;
    str_map_t declStrs;

    int id = node->getId();

    if (sizeof(void *) == 8) {
        kerStream << layout64;
        kerStream << triple64;
    } else {
        kerStream << layout32;
        kerStream << triple32;
    }

    kerStream << "define void " << funcName << " (" << std::endl;
    node->genParams(kerStream, annStream, is_linear);
    kerStream << node->getTypeStr() <<"* %out,\n"
              << "i32 %ostr0, i32 %ostr1, i32 %ostr2, i32 %ostr3,\n"
              << "i32 %odim0, i32 %odim1, i32 %odim2, i32 %odim3,\n"
              << "i32 %blkx, i32 %blky, i32 %ndims) {"
              << "\n\n";

    kerStream << "entry:\n\n";
    kerStream << "%tidx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n";
    kerStream << "%bdmx = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n";
    kerStream << "%bidx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n";
    kerStream << "%bidy = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()\n";
    kerStream << "%gdmx = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()\n";
    kerStream << "\n\n";

    if (!is_linear) {

        kerStream << "%tidy = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()\n";
        kerStream << "%bdmy = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()\n";

        kerStream << "%blk_x = alloca i32, align 4\n";
        kerStream << "%blk_y = alloca i32, align 4\n";
        kerStream << "%id_3 = alloca i32, align 4\n";
        kerStream << "%id_2 = alloca i32, align 4\n";
        kerStream << "store i32 %bidx, i32* %blk_x, align 4\n";
        kerStream << "store i32 %bidy, i32* %blk_y, align 4\n";
        kerStream << "store i32 0, i32* %id_2, align 4\n";
        kerStream << "store i32 0, i32* %id_3, align 4\n";

        kerStream << "%two = alloca i32, align 4\n";
        kerStream << "store i32 2, i32* %two, align 4\n";
        kerStream << "%twoval = load i32* %two, align 4\n";
        kerStream << "%is34 = icmp sgt i32 %ndims, %twoval\n";
        kerStream << "br i1 %is34, label %do34, label %do2\n";

        kerStream << "\ndo34:\n";

        kerStream << "%id2t = sdiv i32 %bidx, %blkx\n";
        kerStream << "store i32 %id2t, i32* %id_2, align 4\n";
        kerStream << "%id2m = mul i32 %id2t, %blkx\n";
        kerStream << "%blk_xx = sub i32 %bidx, %id2m\n";
        kerStream << "store i32 %blk_xx, i32* %blk_x, align 4\n";

        kerStream << "%three = alloca i32, align 4\n";
        kerStream << "store i32 3, i32* %three, align 4\n";
        kerStream << "%threeval = load i32* %three, align 4\n";
        kerStream << "%is4 = icmp sgt i32 %ndims, %threeval\n";
        kerStream << "br i1 %is4, label %do4, label %do2\n";

        kerStream << "\ndo4:\n";
        kerStream << "%id3t = sdiv i32 %bidy, %blky\n";
        kerStream << "store i32 %id3t, i32* %id_3, align 4\n";
        kerStream << "%id3m = mul i32 %id3t, %blky\n";
        kerStream << "%blk_yy = sub i32 %bidy, %id3m\n";
        kerStream << "store i32 %blk_yy, i32* %blk_y, align 4\n";
        kerStream << "br label %do2\n";

        kerStream << "\ndo2:\n";
        kerStream << "%id2 = load i32* %id_2, align 4\n";
        kerStream << "%id3 = load i32* %id_3, align 4\n";

        kerStream << "%tmp_x = load i32* %blk_x, align 4\n";
        kerStream << "%id0m = mul i32 %tmp_x, %bdmx\n";
        kerStream << "%id0 = add i32 %tidx, %id0m\n";

        kerStream << "%tmp_y = load i32* %blk_y, align 4\n";
        kerStream << "%id1m = mul i32 %tmp_y, %bdmy\n";
        kerStream << "%id1 = add i32 %tidy, %id1m\n";
        kerStream << "\n\n";

        kerStream << "%off3o = mul i32 %id3, %ostr3\n";
        kerStream << "%off2o = mul i32 %id2, %ostr2\n";
        kerStream << "%off1o = mul i32 %id1, %ostr1\n";
        kerStream << "%off23o = add i32 %off3o, %off2o\n";
        kerStream << "%off123o = add i32 %off23o, %off1o\n";
        kerStream << "%idxa = add i32 %off123o, %id0\n";
        kerStream << "%idx = sext i32 %idxa to i64\n";
        kerStream << "\n\n";

        kerStream << "%cmp3 = icmp slt i32 %id3, %odim3\n";
        kerStream << "%cmp2 = icmp slt i32 %id2, %odim2\n";
        kerStream << "%cmp1 = icmp slt i32 %id1, %odim1\n";
        kerStream << "%cmp0 = icmp slt i32 %id0, %odim0\n";

        kerStream << "br i1 %cmp3, label %check2, label %end\n";
        kerStream << "\ncheck2:\n";
        kerStream << "br i1 %cmp2, label %check1, label %end\n";
        kerStream << "\ncheck1:\n";
        kerStream << "br i1 %cmp1, label %check0, label %end\n";
        kerStream << "\ncheck0:\n";
        kerStream << "br i1 %cmp0, label %core, label %end\n";

    } else {

        kerStream << "%boff = mul i32 %bidy, %gdmx\n";
        kerStream << "%bid  = add i32 %boff, %bidx\n";
        kerStream << "%goff = mul i32 %bid , %bdmx\n";
        kerStream << "%gid  = add i32 %goff ,%tidx\n";
        kerStream << "%idx  = sext i32 %gid to i64\n";
        kerStream << "%el1  = mul i32 %odim0, %odim1\n";
        kerStream << "%el2  = mul i32 %el1  , %odim2\n";
        kerStream << "%el3  = mul i32 %el2  , %odim3\n";
        kerStream << "%cmp0 = icmp slt i32 %gid, %el3\n";
        kerStream << "br i1 %cmp0, label %core, label %end\n";
    }

    kerStream << "\n";
    kerStream << "end:\n\n";
    kerStream << "ret void\n";

    kerStream <<"\n";
    kerStream << "core:\n\n";
    node->genOffsets(kerStream, is_linear);

    node->genFuncs(kerStream, declStrs, is_linear);

    kerStream << "%outIdx = getelementptr inbounds " << node->getTypeStr() << "* %out, i64 %idx\n";
    kerStream << "store "
              << node->getTypeStr()
              << " %val" << id << ", "
              << node->getTypeStr()
              << "* %outIdx\n";

    kerStream << "\nret void\n";
    kerStream << "\n}\n\n";

    for(str_map_iter iterator = declStrs.begin(); iterator != declStrs.end(); iterator++) {
        kerStream << iterator->first << "\n";
    }

    kerStream
        << "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() nounwind readnone\n"
        << "declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() nounwind readnone\n";

    kerStream << "\n";

    kerStream << "!nvvm.annotations = !{!1}\n"
              << "!1 = metadata !{void (\n"
              << annStream.str()
              << node->getTypeStr() << "*,\n"
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

static CUfunction getKernel(Node *node, bool is_linear)
{

    string funcName = getFuncName(node, is_linear);

    typedef std::map<string, kc_entry_t> kc_t;
    static kc_t kernelCaches[DeviceManager::MAX_DEVICES];
    int device = getActiveDeviceId();

    kc_t::iterator idx = kernelCaches[device].find(funcName);
    kc_entry_t entry = {NULL, NULL};

    if (idx == kernelCaches[device].end()) {
        string jit_ker = getKernelString(funcName, node, is_linear);
        entry = compileKernel(funcName.c_str(), jit_ker);
        kernelCaches[device][funcName] = entry;
    } else {
        entry = idx->second;
    }

    return entry.ker;
}

template<typename T>
void evalNodes(Param<T> &out, Node *node)
{
    bool is_linear = node->isLinear(out.dims);
    CUfunction ker = getKernel(node, is_linear);
    vector<void *> args;
    node->setArgs(args, is_linear);

    void *ptr = (void *)out.ptr;
    int strides[] = {(int)out.strides[0],
                     (int)out.strides[1],
                     (int)out.strides[2],
                     (int)out.strides[3]};

    int dims[] = {(int)out.dims[0],
                  (int)out.dims[1],
                  (int)out.dims[2],
                  (int)out.dims[3]};

    args.push_back((void *)&ptr);
    for (int i = 0; i < 4; i++) args.push_back((void *)(strides + i));
    for (int i = 0; i < 4; i++) args.push_back((void *)(dims + i));

    int threads_x = 1, threads_y = 1;
    int blocks_x_ = 1, blocks_y_ = 1;
    int blocks_x  = 1, blocks_y = 1;

    int num_odims = 4;

    while (num_odims >= 1) {
        if (out.dims[num_odims - 1] == 1) num_odims--;
        else break;
    }

    if (is_linear) {

        threads_x = 256;
        threads_y =  1;

        int blocks = divup((out.dims[0] *
                            out.dims[1] *
                            out.dims[2] *
                            out.dims[3]), threads_x);

        blocks_y_ = divup(blocks, 65535);
        blocks_x_ = divup(blocks, blocks_y_);

        blocks_x = blocks_x_;
        blocks_y = blocks_y_;

    } else {

        threads_x = 32;
        threads_y =  8;

        blocks_x_ = divup(out.dims[0], threads_x);
        blocks_y_ = divup(out.dims[1], threads_y);

        blocks_x = blocks_x_ * out.dims[2];
        blocks_y = blocks_y_ * out.dims[3];
    }

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


}
