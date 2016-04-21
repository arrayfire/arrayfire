/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/sort_pair.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <utility>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

#include <boost/compute/function.hpp>

template<typename Tk, typename Tv, bool isAscending>
inline
boost::compute::function<bool(const std::pair<Tk, Tv>, const std::pair<Tk, Tv>)>
makeCompareFunction()
{
    // Cannot use isAscending in BOOST_COMPUTE_FUNCTION
    if(isAscending) {
        BOOST_COMPUTE_FUNCTION(bool, IPCompare, (std::pair<Tk, Tv> lhs, std::pair<Tk, Tv> rhs),
            {
                return lhs.first < rhs.first;
            }
        );
        return IPCompare;
    } else {
        BOOST_COMPUTE_FUNCTION(bool, IPCompare, (std::pair<Tk, Tv> lhs, std::pair<Tk, Tv> rhs),
            {
                return lhs.first > rhs.first;
            }
        );
        return IPCompare;
    }
}

template<typename Tk>
inline boost::compute::function<Tk(Tk)>
flipFunction()
{
    BOOST_COMPUTE_FUNCTION(Tk, negateFn, (const Tk x),
        {
            return -x;
        }
    );

    return negateFn;
}

#define INSTANTIATE_FLIP(TY, XMAX)                              \
template<> inline boost::compute::function<TY(TY)>              \
flipFunction<TY>()                                              \
{                                                               \
    BOOST_COMPUTE_FUNCTION(TY, negateFn, (const TY x),          \
       {                                                        \
            return XMAX - x;                                    \
        }                                                       \
    );                                                          \
                                                                \
    return negateFn;                                            \
}

INSTANTIATE_FLIP(unsigned, UINT_MAX)
INSTANTIATE_FLIP(unsigned short, USHRT_MAX)
INSTANTIATE_FLIP(unsigned char, UCHAR_MAX)
INSTANTIATE_FLIP(cl_ulong, ULONG_MAX)

#undef INSTANTIATE_FLIP

namespace opencl
{
    namespace kernel
    {
        using std::conditional;
        using std::is_same;

        // If type is cdouble, return std::complex<double>, else return T
        template<typename T>
        using ztype_t = typename conditional<is_same<T, cdouble>::value,
                                             std::complex<double>, T
                                            >::type;

        // If type is cfloat, return std::complex<float>, else return ztype_t
        template<typename T>
        using ctype_t = typename conditional<is_same<T, cfloat>::value,
                                             std::complex<float>, ztype_t<T>
                                            >::type;

        // If type is intl, return cl_long, else return ctype_t
        template<typename T>
        using ltype_t = typename conditional<is_same<T, intl>::value,
                                             cl_long, ctype_t<T>
                                            >::type;

        // If type is uintl, return cl_ulong, else return ltype_t
        template<typename T>
        using type_t = typename conditional<is_same<T, uintl>::value,
                                            cl_ulong, ltype_t<T>
                                           >::type;

        static const int copyPairIter = 4;

        template<typename Tk, typename Tv>
        void makePair(cl::Buffer *out, const cl::Buffer *first, const cl::Buffer *second, const unsigned N)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   sortPairProgs;
                static std::map<int, Kernel*>  sortPairKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D Tk="      << dtype_traits<Tk>::getName()
                            << " -D Tv="      << dtype_traits<Tv>::getName()
                            << " -D copyPairIter=" << copyPairIter;
                    if (std::is_same<Tk, double >::value ||
                        std::is_same<Tk, cdouble>::value ||
                        std::is_same<Tv, double >::value ||
                        std::is_same<Tv, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, sort_pair_cl, sort_pair_cl_len, options.str());
                    sortPairProgs[device]   = new Program(prog);
                    sortPairKernels[device] = new Kernel(*sortPairProgs[device], "make_pair_kernel");
                });

                auto makePairOp = make_kernel<Buffer, const Buffer, const Buffer, const unsigned>
                                          (*sortPairKernels[device]);

                NDRange local(256, 1, 1);
                NDRange global(local[0] * divup(N, local[0] * copyPairIter), 1, 1);

                makePairOp(EnqueueArgs(getQueue(), global, local), *out, *first, *second, N);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }

        template<typename Tk, typename Tv>
        void splitPair(cl::Buffer *first, cl::Buffer *second, const cl::Buffer *in, const unsigned N)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   sortPairProgs;
                static std::map<int, Kernel*>  sortPairKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D Tk="      << dtype_traits<Tk>::getName()
                            << " -D Tv="      << dtype_traits<Tv>::getName()
                            << " -D copyPairIter=" << copyPairIter;
                    if (std::is_same<Tk, double >::value ||
                        std::is_same<Tk, cdouble>::value ||
                        std::is_same<Tv, double >::value ||
                        std::is_same<Tv, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, sort_pair_cl, sort_pair_cl_len, options.str());
                    sortPairProgs[device]   = new Program(prog);
                    sortPairKernels[device] = new Kernel(*sortPairProgs[device], "split_pair_kernel");
                });

                auto splitPairOp = make_kernel<Buffer, Buffer, const Buffer, const unsigned>
                                          (*sortPairKernels[device]);

                NDRange local(256, 1, 1);
                NDRange global(local[0] * divup(N, local[0] * copyPairIter), 1, 1);

                splitPairOp(EnqueueArgs(getQueue(), global, local), *first, *second, *in, N);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}
