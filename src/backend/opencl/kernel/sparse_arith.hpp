/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/sparse_arith_csr.hpp>
#include <kernel_headers/sparse_arith_coo.hpp>
#include <kernel_headers/sparse_arith_common.hpp>
#include <kernel_headers/ssarith_calc_out_nnz.hpp>
#include <kernel_headers/sp_sp_arith_csr.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <common/complex.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <cache.hpp>
#include <type_util.hpp>
#include <types.hpp>
#include <math.hpp>

namespace opencl
{
    namespace kernel
    {
        static const unsigned TX = 32;
        static const unsigned TY = 8;
        static const unsigned THREADS = TX * TY;

        template<af_op_t op>
        std::string getOpString()
        {
            switch(op) {
                case af_add_t : return "ADD";
                case af_sub_t : return "SUB";
                case af_mul_t : return "MUL";
                case af_div_t : return "DIV";
                default       : return ""; // kernel will fail to compile
            }
            return "";
        }

        template<typename T, af_op_t op>
        void sparseArithOpCSR(Param out, const Param values, const Param rowIdx, const Param colIdx,
                const Param rhs, const bool reverse)
        {
            std::string ref_name =
                std::string("sparseArithOpCSR_") +
                getOpString<op>() + std::string("_") +
                std::string(dtype_traits<T>::getName());

            int device = getActiveDeviceId();
            kc_entry_t entry = kernelCache(device, ref_name);

            if (entry.prog==0 && entry.ker==0) {

                std::ostringstream options;
                options << " -D T="  << dtype_traits<T>::getName();
                options << " -D OP=" << getOpString<op>();

                if((af_dtype) dtype_traits<T>::af_type == c32 ||
                        (af_dtype) dtype_traits<T>::af_type == c64) {
                    options << " -D IS_CPLX=1";
                } else {
                    options << " -D IS_CPLX=0";
                }
                if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                const char *ker_strs[] = {sparse_arith_common_cl    , sparse_arith_csr_cl};
                const int   ker_lens[] = {sparse_arith_common_cl_len, sparse_arith_csr_cl_len};

                cl::Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new cl::Program(prog);
                entry.ker  = new cl::Kernel(*entry.prog, "sparse_arith_csr_kernel");

                addKernelToCache(device, ref_name, entry);
            }

            auto sparseArithCSROp = cl::KernelFunctor<cl::Buffer, const KParam,
                 const cl::Buffer, const cl::Buffer, const cl::Buffer,
                 const int,
                 const cl::Buffer, const KParam,
                 const int>(*entry.ker);

            cl::NDRange local(TX, TY, 1);
            cl::NDRange global(divup(out.info.dims[0], TY) * TX, TY, 1);

            sparseArithCSROp(cl::EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info,
                    *values.data, *rowIdx.data, *colIdx.data, values.info.dims[0],
                    *rhs.data, rhs.info, reverse);

            CL_DEBUG_FINISH(getQueue());
        }

        template<typename T, af_op_t op>
        void sparseArithOpCOO(Param out, const Param values, const Param rowIdx, const Param colIdx,
                const Param rhs, const bool reverse)
        {
            std::string ref_name =
                std::string("sparseArithOpCOO_") +
                getOpString<op>() + std::string("_") +
                std::string(dtype_traits<T>::getName());

            int device = getActiveDeviceId();
            kc_entry_t entry = kernelCache(device, ref_name);

            if (entry.prog==0 && entry.ker==0) {
                std::ostringstream options;
                options << " -D T="  << dtype_traits<T>::getName();
                options << " -D OP=" << getOpString<op>();

                if((af_dtype) dtype_traits<T>::af_type == c32 ||
                        (af_dtype) dtype_traits<T>::af_type == c64) {
                    options << " -D IS_CPLX=1";
                } else {
                    options << " -D IS_CPLX=0";
                }
                if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                const char *ker_strs[] = {sparse_arith_common_cl    , sparse_arith_coo_cl};
                const int   ker_lens[] = {sparse_arith_common_cl_len, sparse_arith_coo_cl_len};

                cl::Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new cl::Program(prog);
                entry.ker  = new cl::Kernel(*entry.prog, "sparse_arith_coo_kernel");

                addKernelToCache(device, ref_name, entry);
            }

            auto sparseArithCOOOp = cl::KernelFunctor<cl::Buffer, const KParam,
                 const cl::Buffer, const cl::Buffer, const cl::Buffer,
                 const int,
                 const cl::Buffer, const KParam,
                 const int>(*entry.ker);

            cl::NDRange local(THREADS, 1, 1);
            cl::NDRange global(divup(values.info.dims[0], THREADS) * THREADS, 1, 1);

            sparseArithCOOOp(cl::EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info,
                    *values.data, *rowIdx.data, *colIdx.data, values.info.dims[0],
                    *rhs.data, rhs.info, reverse);

            CL_DEBUG_FINISH(getQueue());
        }

        template<typename T, af_op_t op>
        void sparseArithOpCSR(Param values, Param rowIdx, Param colIdx,
                const Param rhs, const bool reverse)
        {
            std::string ref_name =
                std::string("sparseArithOpSCSR_") +
                getOpString<op>() + std::string("_") +
                std::string(dtype_traits<T>::getName());

            int device = getActiveDeviceId();
            kc_entry_t entry = kernelCache(device, ref_name);

            if (entry.prog==0 && entry.ker==0) {

                std::ostringstream options;
                options << " -D T="  << dtype_traits<T>::getName();
                options << " -D OP=" << getOpString<op>();

                if((af_dtype) dtype_traits<T>::af_type == c32 ||
                        (af_dtype) dtype_traits<T>::af_type == c64) {
                    options << " -D IS_CPLX=1";
                } else {
                    options << " -D IS_CPLX=0";
                }
                if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                const char *ker_strs[] = {sparse_arith_common_cl    , sparse_arith_csr_cl};
                const int   ker_lens[] = {sparse_arith_common_cl_len, sparse_arith_csr_cl_len};

                cl::Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new cl::Program(prog);
                entry.ker  = new cl::Kernel(*entry.prog, "sparse_arith_csr_kernel_S");

                addKernelToCache(device, ref_name, entry);
            }

            auto sparseArithCSROp = cl::KernelFunctor<const cl::Buffer, const cl::Buffer, const cl::Buffer,
                 const int,
                 const cl::Buffer, const KParam,
                 const int>(*entry.ker);

            cl::NDRange local(TX, TY, 1);
            cl::NDRange global(divup(rhs.info.dims[0], TY) * TX, TY, 1);

            sparseArithCSROp(cl::EnqueueArgs(getQueue(), global, local),
                    *values.data, *rowIdx.data, *colIdx.data, values.info.dims[0],
                    *rhs.data, rhs.info, reverse);

            CL_DEBUG_FINISH(getQueue());
        }

        template<typename T, af_op_t op>
        void sparseArithOpCOO(Param values, Param rowIdx, Param colIdx,
                const Param rhs, const bool reverse)
        {
            std::string ref_name =
                std::string("sparseArithOpSCOO_") +
                getOpString<op>() + std::string("_") +
                std::string(dtype_traits<T>::getName());

            int device = getActiveDeviceId();
            kc_entry_t entry = kernelCache(device, ref_name);

            if (entry.prog==0 && entry.ker==0) {
                std::ostringstream options;
                options << " -D T="  << dtype_traits<T>::getName();
                options << " -D OP=" << getOpString<op>();

                if((af_dtype) dtype_traits<T>::af_type == c32 ||
                        (af_dtype) dtype_traits<T>::af_type == c64) {
                    options << " -D IS_CPLX=1";
                } else {
                    options << " -D IS_CPLX=0";
                }
                if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                const char *ker_strs[] = {sparse_arith_common_cl    , sparse_arith_coo_cl};
                const int   ker_lens[] = {sparse_arith_common_cl_len, sparse_arith_coo_cl_len};

                cl::Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new cl::Program(prog);
                entry.ker  = new cl::Kernel(*entry.prog, "sparse_arith_coo_kernel_S");

                addKernelToCache(device, ref_name, entry);
            }

            auto sparseArithCOOOp = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,
                 const int,
                 const cl::Buffer, const KParam,
                 const int>(*entry.ker);

            cl::NDRange local(THREADS, 1, 1);
            cl::NDRange global(divup(values.info.dims[0], THREADS) * THREADS, 1, 1);

            sparseArithCOOOp(cl::EnqueueArgs(getQueue(), global, local),
                    *values.data, *rowIdx.data, *colIdx.data, values.info.dims[0],
                    *rhs.data, rhs.info, reverse);

            CL_DEBUG_FINISH(getQueue());
        }

        static
        void csrCalcOutNNZ(Param outRowIdx, unsigned &nnzC,
                const uint M, const uint N,
                uint nnzA, const Param lrowIdx, const Param lcolIdx,
                uint nnzB, const Param rrowIdx, const Param rcolIdx)
        {
            std::string refName = std::string("csr_calc_output_NNZ");
            int device = getActiveDeviceId();
            kc_entry_t entry = kernelCache(device, refName);

            if (entry.prog==0 && entry.ker==0) {
                const char *kerStrs[] = { ssarith_calc_out_nnz_cl };
                const int kerLens[] = { ssarith_calc_out_nnz_cl_len };

                cl::Program prog;
                buildProgram(prog, 1, kerStrs, kerLens, std::string(""));
                entry.prog = new cl::Program(prog);
                entry.ker = new cl::Kernel(*entry.prog, "csr_calc_out_nnz");

                addKernelToCache(device, refName, entry);
            }
            auto calcNNZop = cl::KernelFunctor<cl::Buffer, cl::Buffer, unsigned,
                 const cl::Buffer, const cl::Buffer,
                 const cl::Buffer, const cl::Buffer,
                 cl::LocalSpaceArg>(*entry.ker);

            cl::NDRange local(256, 1);
            cl::NDRange global(divup(M, local[0])*local[0], 1, 1);

            nnzC = 0;
            cl::Buffer* out = bufferAlloc(sizeof(unsigned));
            getQueue().enqueueWriteBuffer(*out, CL_TRUE, 0, sizeof(unsigned), &nnzC);

            calcNNZop(cl::EnqueueArgs(getQueue(), global, local),
                      *out, *outRowIdx.data, M,
                      *lrowIdx.data, *lcolIdx.data,
                      *rrowIdx.data, *rcolIdx.data,
                      cl::Local(local[0]*sizeof(unsigned int)));
            getQueue().enqueueReadBuffer(*out, CL_TRUE, 0, sizeof(unsigned), &nnzC);

            CL_DEBUG_FINISH(getQueue());
        }

        template<typename T, af_op_t op>
        void ssArithCSR(Param oVals, Param oColIdx,
                const Param oRowIdx, const uint M, const uint N,
                unsigned nnzA, const Param lVals, const Param lRowIdx, const Param lColIdx,
                unsigned nnzB, const Param rVals, const Param rRowIdx, const Param rColIdx)
        {
            std::string refName = std::string("ss_arith_csr_") +
                                  getOpString<op>() + "_" +
                                  std::string(dtype_traits<T>::getName());
            int device = getActiveDeviceId();
            kc_entry_t entry = kernelCache(device, refName);

            if (entry.prog==0 && entry.ker==0) {
                const T iden_val = (op == af_mul_t || op == af_div_t ?
                                      scalar<T>(1) : scalar<T>(0));
                ToNumStr<T> toNumStr;
                std::ostringstream options;
                options << " -D T="  << dtype_traits<T>::getName()
                        << " -D OP=" << getOpString<op>()
                        << " -D IDENTITY_VALUE=(T)(" << af::scalar_to_option(iden_val) << ")";

                options << " -D IS_CPLX=" << common::is_complex<T>::value;
                if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                const char *kerStrs[] = { sparse_arith_common_cl, sp_sp_arith_csr_cl };
                const int kerLens[] = { sparse_arith_common_cl_len, sp_sp_arith_csr_cl_len };

                cl::Program prog;
                buildProgram(prog, 2, kerStrs, kerLens, options.str());
                entry.prog = new cl::Program(prog);
                entry.ker = new cl::Kernel(*entry.prog, "ssarith_csr_kernel");

                addKernelToCache(device, refName, entry);
            }
            auto arithOp = cl::KernelFunctor<cl::Buffer, cl::Buffer,
                 cl::Buffer, unsigned, unsigned,
                 unsigned, const cl::Buffer,
                 const cl::Buffer, const cl::Buffer,
                 unsigned, const cl::Buffer,
                 const cl::Buffer, const cl::Buffer>(*entry.ker);

            cl::NDRange local(256, 1);
            cl::NDRange global(divup(M, local[0])*local[0], 1, 1);

            arithOp(cl::EnqueueArgs(getQueue(), global, local),
                      *oVals.data, *oColIdx.data,
                      *oRowIdx.data, M, N,
                      nnzA, *lVals.data, *lRowIdx.data, *lColIdx.data,
                      nnzB, *rVals.data, *rRowIdx.data, *rColIdx.data);

            CL_DEBUG_FINISH(getQueue());
        }
    }
}
