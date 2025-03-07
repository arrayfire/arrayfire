/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/complex.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/sp_sp_arith_csr.hpp>
#include <kernel_headers/sparse_arith_common.hpp>
#include <kernel_headers/sparse_arith_coo.hpp>
#include <kernel_headers/sparse_arith_csr.hpp>
#include <kernel_headers/ssarith_calc_out_nnz.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

constexpr unsigned TX      = 32;
constexpr unsigned TY      = 8;
constexpr unsigned THREADS = TX * TY;

template<af_op_t op>
AF_CONSTEXPR const char *getOpString() {
    switch (op) {
        case af_add_t: return "ADD";
        case af_sub_t: return "SUB";
        case af_mul_t: return "MUL";
        case af_div_t: return "DIV";
        default: return "";  // kernel will fail to compile
    }
    return "";
}

template<typename T, af_op_t op>
auto fetchKernel(const std::string key, const common::Source &additionalSrc,
                 const std::vector<std::string> additionalOptions = {}) {
    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    std::array<TemplateArg, 2> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(op),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(OP, getOpString<op>()),
        DefineKeyValue(IS_CPLX, (IsComplex ? 1 : 0)),
    };
    options.emplace_back(getTypeBuildDefinition<T>());
    options.insert(std::end(options), std::begin(additionalOptions),
                   std::end(additionalOptions));
    return common::getKernel(key, {{sparse_arith_common_cl_src, additionalSrc}},
                             tmpltArgs, options);
}

template<typename T, af_op_t op>
void sparseArithOpCSR(Param out, const Param values, const Param rowIdx,
                      const Param colIdx, const Param rhs, const bool reverse) {
    auto sparseArithCSR =
        fetchKernel<T, op>("sparseArithCSR", sparse_arith_csr_cl_src);

    cl::NDRange local(TX, TY, 1);
    cl::NDRange global(divup(out.info.dims[0], TY) * TX, TY, 1);

    sparseArithCSR(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                   out.info, *values.data, *rowIdx.data, *colIdx.data,
                   static_cast<int>(values.info.dims[0]), *rhs.data, rhs.info,
                   static_cast<int>(reverse));
    CL_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
void sparseArithOpCOO(Param out, const Param values, const Param rowIdx,
                      const Param colIdx, const Param rhs, const bool reverse) {
    auto sparseArithCOO =
        fetchKernel<T, op>("sparseArithCOO", sparse_arith_coo_cl_src);

    cl::NDRange local(THREADS, 1, 1);
    cl::NDRange global(divup(values.info.dims[0], THREADS) * THREADS, 1, 1);

    sparseArithCOO(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                   out.info, *values.data, *rowIdx.data, *colIdx.data,
                   static_cast<int>(values.info.dims[0]), *rhs.data, rhs.info,
                   static_cast<int>(reverse));
    CL_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
void sparseArithOpCSR(Param values, Param rowIdx, Param colIdx, const Param rhs,
                      const bool reverse) {
    auto sparseArithCSR =
        fetchKernel<T, op>("sparseArithCSR2", sparse_arith_csr_cl_src);

    cl::NDRange local(TX, TY, 1);
    cl::NDRange global(divup(rhs.info.dims[0], TY) * TX, TY, 1);

    sparseArithCSR(cl::EnqueueArgs(getQueue(), global, local), *values.data,
                   *rowIdx.data, *colIdx.data,
                   static_cast<int>(values.info.dims[0]), *rhs.data, rhs.info,
                   static_cast<int>(reverse));
    CL_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
void sparseArithOpCOO(Param values, Param rowIdx, Param colIdx, const Param rhs,
                      const bool reverse) {
    auto sparseArithCOO =
        fetchKernel<T, op>("sparseArithCOO2", sparse_arith_coo_cl_src);

    cl::NDRange local(THREADS, 1, 1);
    cl::NDRange global(divup(values.info.dims[0], THREADS) * THREADS, 1, 1);

    sparseArithCOO(cl::EnqueueArgs(getQueue(), global, local), *values.data,
                   *rowIdx.data, *colIdx.data,
                   static_cast<int>(values.info.dims[0]), *rhs.data, rhs.info,
                   static_cast<int>(reverse));
    CL_DEBUG_FINISH(getQueue());
}

static void csrCalcOutNNZ(Param outRowIdx, unsigned &nnzC, const uint M,
                          const uint N, uint nnzA, const Param lrowIdx,
                          const Param lcolIdx, uint nnzB, const Param rrowIdx,
                          const Param rcolIdx) {
    UNUSED(N);
    UNUSED(nnzA);
    UNUSED(nnzB);

    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<uint>(),
    };

    auto calcNNZ = common::getKernel(
        "csr_calc_out_nnz", {{ssarith_calc_out_nnz_cl_src}}, tmpltArgs, {});

    cl::NDRange local(256, 1);
    cl::NDRange global(divup(M, local[0]) * local[0], 1, 1);

    nnzC     = 0;
    auto out = memAlloc<unsigned>(1);
    getQueue().enqueueFillBuffer(*out, nnzC, 0, sizeof(unsigned));

    calcNNZ(cl::EnqueueArgs(getQueue(), global, local), *out, *outRowIdx.data,
            M, *lrowIdx.data, *lcolIdx.data, *rrowIdx.data, *rcolIdx.data,
            cl::Local(local[0] * sizeof(unsigned int)));
    getQueue().enqueueReadBuffer(*out, CL_TRUE, 0, sizeof(unsigned), &nnzC);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
void ssArithCSR(Param oVals, Param oColIdx, const Param oRowIdx, const uint M,
                const uint N, unsigned nnzA, const Param lVals,
                const Param lRowIdx, const Param lColIdx, unsigned nnzB,
                const Param rVals, const Param rRowIdx, const Param rColIdx) {
    const T iden_val =
        (op == af_mul_t || op == af_div_t ? scalar<T>(1) : scalar<T>(0));

    auto arithOp = fetchKernel<T, op>(
        "ssarith_csr", sp_sp_arith_csr_cl_src,
        {DefineKeyValue(IDENTITY_VALUE, scalar_to_option(iden_val))});

    cl::NDRange local(256, 1);
    cl::NDRange global(divup(M, local[0]) * local[0], 1, 1);

    arithOp(cl::EnqueueArgs(getQueue(), global, local), *oVals.data,
            *oColIdx.data, *oRowIdx.data, M, N, nnzA, *lVals.data,
            *lRowIdx.data, *lColIdx.data, nnzB, *rVals.data, *rRowIdx.data,
            *rColIdx.data);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
