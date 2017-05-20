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
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <cache.hpp>
#include <type_util.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

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

                Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new Program(prog);
                entry.ker  = new Kernel(*entry.prog, "sparse_arith_csr_kernel");

                addKernelToCache(device, ref_name, entry);
            }

            auto sparseArithCSROp = KernelFunctor<Buffer, const KParam,
                 const Buffer, const Buffer, const Buffer,
                 const int,
                 const Buffer, const KParam,
                 const int>(*entry.ker);

            NDRange local(TX, TY, 1);
            NDRange global(divup(out.info.dims[0], TY) * TX, TY, 1);

            sparseArithCSROp(EnqueueArgs(getQueue(), global, local),
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

                Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new Program(prog);
                entry.ker  = new Kernel(*entry.prog, "sparse_arith_coo_kernel");

                addKernelToCache(device, ref_name, entry);
            }

            auto sparseArithCOOOp = KernelFunctor<Buffer, const KParam,
                 const Buffer, const Buffer, const Buffer,
                 const int,
                 const Buffer, const KParam,
                 const int>(*entry.ker);

            NDRange local(THREADS, 1, 1);
            NDRange global(divup(values.info.dims[0], THREADS) * THREADS, 1, 1);

            sparseArithCOOOp(EnqueueArgs(getQueue(), global, local),
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

                Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new Program(prog);
                entry.ker  = new Kernel(*entry.prog, "sparse_arith_csr_kernel_S");

                addKernelToCache(device, ref_name, entry);
            }

            auto sparseArithCSROp = KernelFunctor<const Buffer, const Buffer, const Buffer,
                 const int,
                 const Buffer, const KParam,
                 const int>(*entry.ker);

            NDRange local(TX, TY, 1);
            NDRange global(divup(rhs.info.dims[0], TY) * TX, TY, 1);

            sparseArithCSROp(EnqueueArgs(getQueue(), global, local),
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

                Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new Program(prog);
                entry.ker  = new Kernel(*entry.prog, "sparse_arith_coo_kernel_S");

                addKernelToCache(device, ref_name, entry);
            }

            auto sparseArithCOOOp = KernelFunctor<Buffer, Buffer, Buffer,
                 const int,
                 const Buffer, const KParam,
                 const int>(*entry.ker);

            NDRange local(THREADS, 1, 1);
            NDRange global(divup(values.info.dims[0], THREADS) * THREADS, 1, 1);

            sparseArithCOOOp(EnqueueArgs(getQueue(), global, local),
                    *values.data, *rowIdx.data, *colIdx.data, values.info.dims[0],
                    *rhs.data, rhs.info, reverse);

            CL_DEBUG_FINISH(getQueue());
        }
    }
}
