/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/dispatch.hpp>
#include <platform.hpp>
#include <af/defines.h>

namespace opencl {
// OVERALL USAGE (With looping):
// ...                                                      // OWN CODE
// threadsMgt<T> th(...);                                   // backend.hpp
// cl::Kernel KER{GETKERNEL(..., th.loop0, th.loop1,
//                               th.loop3)};                // OWN CODE
// const cl::NDRange local{th.genLocal(KER)};               // backend.hpp
// const cl::NDRange global{th.genGlobal(local)};           // backend.hpp
// KER(local,global,...);                                   // OWN CODE
// ...                                                      // OWN CODE
//
// OVERALL USAGE (without looping):
// ...                                                      // OWN CODE
// threadsMgt<T> th(...);                                   // backend.hpp
// cl::Kernel KER{GETKERNEL(...)};                          // OWN CODE
// const cl::NDRange local{th.genLocal(KER)};               // backend.hpp
// const cl::NDRange global{th.genGlobalFull(local)};       // backend.hpp
// KER(local,global,...);                                   // OWN CODE
// ...                                                      // OWN CODE
template<typename T>
class threadsMgt {
   public:
    bool loop0, loop1, loop3;

   private:
    const unsigned d0, d1, d2, d3;
    const T ndims;
    const size_t totalSize;
    const cl::Device dev;
    const unsigned maxParallelThreads;
    const unsigned maxThreads;
    unsigned largeVolDivider;

   public:
    // INPUT dims = dims of output array
    // INPUT ndims = ndims of output array
    // INPUT nrInputs = number of buffers read by kernel in parallel
    // INPUT nrOutputs = number of buffer written by kernel in parallel
    // INPUT totalSize = size of all input & output arrays
    // INPUT sizeofT = size of 1 element to be written
    // OUTPUT this.loop0, this.loop1, this.loop3 are ready to create the kernel
    threadsMgt(const T dims[4], const T ndims, const unsigned nrInputs,
               const unsigned nrOutputs, const size_t totalSize,
               const size_t sizeofT);

    // The generated local is only best for independent element operations,
    //  as are: copying, scaling, math on independent elements,
    // ... Since vector dimensions can be returned, it is NOT USABLE FOR
    // BLOCK OPERATIONS, as are: matmul, etc.
    inline cl::NDRange genLocal(const cl::Kernel& ker) const;

    // INPUT local generated by genLocal()
    // OUTPUT global, supposing that each element results in 1 thread
    inline cl::NDRange genGlobalFull(const cl::NDRange& local) const;

    // INPUT local generated by genLocal()
    // OUTPUT global, assuming the the previous calculated looping will be
    // executed in the kernel
    inline cl::NDRange genGlobal(const cl::NDRange& local) const;
};

// INPUT dims = dims of output array
// INPUT ndims = ndims of output array
// INPUT nrInputs = number of buffers read by kernel in parallel
// INPUT nrOutputs = number of buffer written by kernel in parallel
// INPUT totalSize = size of all input & output arrays
// INPUT sizeofT = size of 1 element to be written
// OUTPUT this.loop0, this.loop1, this.loop3 are ready to create the kernel
template<typename T>
threadsMgt<T>::threadsMgt(const T dims[4], const T ndims,
                          const unsigned nrInputs, const unsigned nrOutputs,
                          const size_t totalSize, const size_t sizeofT)
    : loop0(false)
    , loop1(false)
    , loop3(false)
    , d0(static_cast<unsigned>(dims[0]))
    , d1(static_cast<unsigned>(dims[1]))
    , d2(static_cast<unsigned>(dims[2]))
    , d3(static_cast<unsigned>(dims[3]))
    , ndims(ndims)
    , totalSize(totalSize)
    , dev(opencl::getDevice())
    , maxParallelThreads(getMaxParallelThreads(dev))
    , maxThreads(maxParallelThreads *
                 (sizeofT * nrInputs * nrInputs > 8 ? 1 : 2))
    , largeVolDivider(1) {
    const unsigned cacheLine{getMemoryBusWidth(dev)};
    const size_t L2CacheSize{getL2CacheSize(dev)};
    // The bottleneck of anykernel is dependent on the type of memory
    // used.
    // a) For very small arrays (elements < maxParallelThreads), each
    //  element receives it individual thread
    // b) For arrays (in+out) smaller
    //  than 3/2 L2cache, memory access no longer is the bottleneck,
    //  because enough L2cache is available at any time. Threads are
    //  limited to reduce scheduling overhead.
    // c) For very large arrays and type sizes
    //  (<long double), 1 thread will not generate enough data to keep
    //  the memory sync mechanism saturated, so we start loooping inside
    //  each thread.
    //
    if (ndims == 1) {
        if (d0 > maxThreads) {
            loop0 = true;
            if (totalSize * 2 > L2CacheSize * 3) {
                // General formula to calculate best #loops
                // Dedicated GPUs:
                //  32/sizeof(T)**2/#outBuffers*(3/4)**(#inBuffers-1)
                // Integrated GPUs:
                //  4/sizeof(T)/#outBuffers*(3/4)**(#inBuffers-1)
                largeVolDivider = cacheLine == 64 ? sizeofT == 1   ? 4
                                                    : sizeofT == 2 ? 2
                                                                   : 1
                                                  : (sizeofT == 1   ? 32
                                                     : sizeofT == 2 ? 8
                                                                    : 1) /
                                                        nrOutputs;
                for (unsigned i = 1; i < nrInputs; ++i)
                    largeVolDivider = largeVolDivider * 3 / 4;
                loop0 = largeVolDivider > 1;
            }
        }
    } else {
        loop3 = d3 != 1;
        if ((d1 > 1) & (d0 * d1 * d2 > maxThreads)) {
            loop1 = true;
            if ((d0 * sizeofT * 8 > cacheLine * getComputeUnits(dev)) &
                (totalSize * 2 > L2CacheSize * 3)) {
                // General formula to calculate best #loops
                // Dedicated GPUs:
                //  32/sizeof(T)**2/#outBuffers*(3/4)**(#inBuffers-1)
                // Integrated GPUs:
                //  4/sizeof(T)/#outBuffers*(3/4)**(#inBuffers-1)
                //
                // dims[3] already loops, so the remaining #loops needs
                // to be divided
                largeVolDivider = cacheLine == 64 ? sizeofT == 1   ? 4
                                                    : sizeofT == 2 ? 2
                                                                   : 1
                                                  : (sizeofT == 1   ? 32
                                                     : sizeofT == 2 ? 8
                                                     : sizeofT == 4 ? 2
                                                                    : 1) /
                                                        (d3 * nrOutputs);
                for (unsigned i{1}; i < nrInputs; ++i)
                    largeVolDivider = largeVolDivider * 3 / 4;
                loop1 = largeVolDivider > 1;
            }
        }
    }
};

// The generated local is only best for independent element operations,
//  as are: copying, scaling, math on independent elements,
// ... Since vector dimensions can be returned, it is NOT USABLE FOR
// BLOCK OPERATIONS, as are: matmul, etc.
template<typename T>
inline cl::NDRange threadsMgt<T>::genLocal(const cl::Kernel& ker) const {
    // Performance is mainly dependend on:
    //    - reducing memory latency, by preferring a sequential read of
    //    cachelines (principally dim0)
    //    - more parallel threads --> higher occupation of available
    //    threads
    //    - more I/O operations per thread --> dims[3] indicates the #
    //    of I/Os handled by the kernel inside each thread, and outside
    //    the scope of the block scheduler
    // High performance is achievable with occupation rates as low as
    // 30%. Here we aim at 50%, to also cover older hardware with slower
    // cores.
    // https://stackoverflow.com/questions/7737772/improving-kernel-performance-by-increasing-occupancy
    // http://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf
    // https://www.cvg.ethz.ch/teaching/2011spring/gpgpu/GPU-Optimization.pdf
    // https://en.wikipedia.org/wiki/Graphics_Core_Next#SIMD_Vector_Unit

    // The performance for vectors is independent from array sizes.
    if ((d1 == 1) & (d2 == 1)) return cl::NDRange{128ULL};

    // TOTAL OCCUPATION = occup(dim0) * occup(dim1) * occup(dim2).
    // For linearized arrays, each linear block is allocated to a dim,
    // resulting in large numbers for dim0 & dim1.
    // - For dim2, we only return exact dividers of the array dim[3], so
    // occup(dim2)=100%
    // - For dim0 & dim1, we aim somewhere between 30% and 50%
    //      * Having 2 blocks filled + 1 thread in block 3 --> occup >
    //      2/3=66%
    //      * Having 3 blocks filled + 1 thread in block 4 --> occup >
    //      3/4=75%
    //      * Having 4 blocks filled + 1 thread in block 5 --> occup >
    //      4/5=80%
    constexpr unsigned OCCUPANCY_FACTOR{2U};  // at least 2 blocks filled

    // NVIDIA:
    //  WG multiple      = 32
    //  possible blocks  = [32, 64, 96, 128, 160, 192, 224, 256, .. 1024]
    //  best performance = [32, 64, 96, 128]
    //  optimal perf     = 128; any combination
    //   NIVIDA always processes full wavefronts.  Allocating partial WG
    //   (<32) reduces throughput.  Performance reaches a plateau from
    //   128 with a slightly slowing for very large sizes.
    // AMD:
    //  WG multiple      = 64
    //  possible block   = [16, 32, 48, 64, 128, 192, 256]
    //  best performance = [(32, low #threads) 64, 128, 256]
    //  optimal perf     = (128,2,1); max 128 for 1 dimension
    //   AMD can process partial wavefronts (multiple of 16), although
    //   all threads of a full WG are allocated, only the active ones
    //   are executed, so the same number of WGs will fit a CU. When we
    //   have insufficent threads to occupy all the CU's, partial
    //   wavefronts (<64) are usefull to distribute all threads over the
    //   available CU's iso all concentrating on the 1st CU.
    // For algorithm below:
    //  parallelThreads  = [32, 64, (96 for NIVIDA), 128, (256 for AMD)]
    constexpr unsigned minThreads{32};
    const unsigned relevantElements{d0 * d1 * d2};
    const unsigned WG{static_cast<unsigned>(
        ker.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
            dev))};

    // For small array's, we reduce the maximum threads in 1 block to
    // improve parallelisme.  In worst case the scheduler can have 1
    // block per CU, even when only partly loaded. Range for block is:
    //   [minThreads ... 4 * WG multiple]
    //   * NVIDIA: [4*32=128 threads]
    //   * AMD:    [4*64=256 threads]
    // At 4 * WG multiple, full wavefronts (queue of 4 partial
    // wavefronts) are all occupied.

    // We need at least maxParallelThreads to occupy all the CU's.
    const unsigned parallelThreads{
        relevantElements <= maxParallelThreads
            ? minThreads
            : std::min(4U, relevantElements / maxParallelThreads) * WG};

    // Priority 1: keep cachelines filled.  Aparrantly sharing
    // cachelines between CU's has a cost. Testing confirmed that the
    // occupation is mostly > 50%
    const unsigned threads0{d0 == 1 ? 1
                            : d0 <= minThreads
                                ? minThreads  // better distribution
                                : std::min(128U, (divup(d0, WG) * WG))};

    // Priority 2: Fill the block, while respecting the occupation limit
    // (>66%) (through parallelThreads limit)
    const unsigned threads1{
        (threads0 * 64U <= parallelThreads) &&
                (!(d1 & (64U - 1U)) || (d1 > OCCUPANCY_FACTOR * 64U))
            ? 64U
        : (threads0 * 32U <= parallelThreads) &&
                (!(d1 & (32U - 1U)) || (d1 > OCCUPANCY_FACTOR * 32U))
            ? 32U
        : (threads0 * 16U <= parallelThreads) &&
                (!(d1 & (16U - 1U)) || (d1 > OCCUPANCY_FACTOR * 16U))
            ? 16U
        : (threads0 * 8U <= parallelThreads) &&
                (!(d1 & (8U - 1U)) || (d1 > OCCUPANCY_FACTOR * 8U))
            ? 8U
        : (threads0 * 4U <= parallelThreads) &&
                (!(d1 & (4U - 1U)) || (d1 > OCCUPANCY_FACTOR * 4U))
            ? 4U
        : (threads0 * 2U <= parallelThreads) &&
                (!(d1 & (2U - 1U)) || (d1 > OCCUPANCY_FACTOR * 2U))
            ? 2U
            : 1U};

    const unsigned threads01{threads0 * threads1};
    if ((d2 == 1) | (threads01 * 2 > parallelThreads))
        return cl::NDRange(threads0, threads1);

    // Priority 3: Only exact dividers are used, so that
    //  - overflow checking is not needed in the kernel.
    //  - occupation rate never is reduced
    // Chances are low that threads2 will be different from 1.
    const unsigned threads2{
        (threads01 * 8 <= parallelThreads) && !(d2 & (8U - 1U))   ? 8U
        : (threads01 * 4 <= parallelThreads) && !(d2 & (4U - 1U)) ? 4U
        : (threads01 * 2 <= parallelThreads) && !(d2 & (2U - 1U)) ? 2U
                                                                  : 1U};
    return cl::NDRange(threads0, threads1, threads2);
};

// INPUT local generated by genLocal()
// OUTPUT global, supposing that each element results in 1 thread
template<typename T>
inline cl::NDRange threadsMgt<T>::genGlobalFull(
    const cl::NDRange& local) const {
    return cl::NDRange(divup(d0, local[0]) * local[0],
                       divup(d1, local[1]) * local[1],
                       divup(d2, local[2]) * local[2]);
};

// INPUT local generated by genLocal()
// OUTPUT global, assuming the the previous calculated looping will be
// executed in the kernel
template<typename T>
inline cl::NDRange threadsMgt<T>::genGlobal(const cl::NDRange& local) const {
    if (loop0) {
        const size_t blocks0{largeVolDivider > 1
                                 ? d0 / (largeVolDivider * local[0])
                                 : maxThreads / local[0]};
        return cl::NDRange(blocks0 == 0 ? local[0] : blocks0 * local[0]);
    } else if (loop1) {
        const size_t global0{divup(d0, local[0]) * local[0]};
        const size_t global2{divup(d2, local[2]) * local[2]};
        const size_t blocks1{largeVolDivider > 1
                                 ? d1 / (largeVolDivider * local[1])
                                 : maxThreads / (global0 * local[1] * global2)};
        return cl::NDRange(
            global0, blocks1 == 0 ? local[1] : blocks1 * local[1], global2);
    } else {
        return genGlobalFull(local);
    }
};
}  // namespace opencl