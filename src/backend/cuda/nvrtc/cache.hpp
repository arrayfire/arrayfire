/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <err_cuda.hpp>
#include <nvrtc/EnqueueArgs.hpp>
#include <traits.hpp>

#include <cstdio>
#include <string>
#include <vector>

#define CU_CHECK(fn)                                                      \
    do {                                                                  \
        CUresult res = fn;                                                \
        if (res == CUDA_SUCCESS) break;                                   \
        char cu_err_msg[1024];                                            \
        const char* cu_err_name;                                          \
        const char* cu_err_string;                                        \
        cuGetErrorName(res, &cu_err_name);                                \
        cuGetErrorString(res, &cu_err_string);                            \
        snprintf(cu_err_msg, sizeof(cu_err_msg), "CU Error %s(%d): %s\n", \
                 cu_err_name, (int)(res), cu_err_string);                 \
        AF_ERROR(cu_err_msg, AF_ERR_INTERNAL);                            \
    } while (0)

namespace cuda {

///
/// \brief Kernel Functor that wraps CUDA nvrtc constructs
///
/// This struct encapsulates CUmodule and CUfunction pointers that are required
/// to execution of CUDA C++ kernels compiled at runtime.
///
struct Kernel {
    CUmodule prog;   ///< CUmodule helps acquire kernel attributes
    CUfunction ker;  ///< CUfuntion is the actual kernel blob to run

    ///
    /// \brief Copy data to constant qualified global variable of kernel
    ///
    /// This function copies data of `bytes` size from the device pointer to a
    /// global(__constant__) variable declared inside the kernel.
    ///
    /// \param[in] name is the name of the global variable inside kernel
    /// \param[in] src is the device pointer from which data will be copied
    /// \param[in] bytes are the number of bytes of data to be copied
    ///
    void setConstant(const char* name, CUdeviceptr src, size_t bytes);

    ///
    /// \brief Copy scalar to device qualified global variable of kernel
    ///
    /// This function copies a single value of type T from host variable
    /// to a global(__device__) variable declared inside the kernel.
    ///
    /// \param[in] name is the name of the global variable inside kernel
    /// \param[in] value is the value of type T
    ///
    template<typename T>
    void setScalar(const char* name, T value);

    ///
    /// \brief Fetch scalar from device qualified global variable of kernel
    ///
    /// This function copies a single value of type T from a global(__device__)
    /// variable declared inside the kernel to host.
    ///
    /// \param[in] name is the name of the global variable inside kernel
    /// \param[in] value is the value of type T
    ///
    template<typename T>
    void getScalar(T& out, const char* name);

    ///
    /// \brief Enqueue Kernel per queueing criteria forwarding other parameters
    ///
    /// This operator overload enables Kernel object to work as functor that
    /// internally executes the CUDA kernel stored inside the Kernel object.
    /// All parameters that are passed in after the EnqueueArgs object are
    /// essentially forwarded to cuLaunchKernel driver API call.
    ///
    /// \param[in] qArgs is an object of struct \ref EnqueueArgs
    /// \param[in] args is the placeholder for variadic arguments
    ///
    template<typename... Args>
    void operator()(const EnqueueArgs& qArgs, Args... args) {
        void* params[] = {reinterpret_cast<void*>(&args)...};
        for (auto& event : qArgs.mEvents) {
            CU_CHECK(cuStreamWaitEvent(qArgs.mStream, event, 0));
        }
        CU_CHECK(cuLaunchKernel(
            ker, qArgs.mBlocks.x, qArgs.mBlocks.y, qArgs.mBlocks.z,
            qArgs.mThreads.x, qArgs.mThreads.y, qArgs.mThreads.z,
            qArgs.mSharedMemSize, qArgs.mStream, params, NULL));
    }
};

// TODO(pradeep): remove this in API and merge JIT and nvrtc caches
Kernel buildKernel(const int device, const std::string& nameExpr,
                   const std::string& jit_ker,
                   const std::vector<std::string>& opts = {},
                   const bool isJIT                     = false);


Kernel loadKernel(const int device, const std::string &nameExpr);

template<typename T>
std::string toString(T val);

struct TemplateArg {
    std::string _tparam;

    TemplateArg(std::string str) : _tparam(str) {}

    template<typename T>
    constexpr TemplateArg(T value) noexcept : _tparam(toString(value)) {}
};

template<typename T>
struct TemplateTypename {
    operator TemplateArg() const noexcept {
        return {std::string(dtype_traits<T>::getName())};
    }
};

#define SPECIALIZE(TYPE, NAME)                      \
    template<>                                      \
    struct TemplateTypename<TYPE> {                 \
        operator TemplateArg() const noexcept {     \
            return TemplateArg(std::string(#NAME)); \
        }                                           \
    }

SPECIALIZE(unsigned char, cuda::uchar);
SPECIALIZE(unsigned int, cuda::uint);
SPECIALIZE(unsigned short, cuda::ushort);
SPECIALIZE(long long, long long);
SPECIALIZE(unsigned long long, unsigned long long);

#undef SPECIALIZE

#define DefineKey(arg) "-D " #arg
#define DefineValue(arg) "-D " #arg "=" + toString(arg)
#define DefineKeyValue(key, arg) "-D " #key "=" + toString(arg)

///
/// \brief Find/Create-Cache a Kernel that fits the given criteria
///
/// This function takes in two vectors of strings apart from the main Kernel
/// name, match criteria, to find a suitable kernel in the Kernel cache. It
/// builds and caches a new Kernel object if one isn't found in the cache.
///
/// The paramter \p key has to be the unique name for a given CUDA kernel.
/// The key has to be present in one of the entries of KernelMap defined in
/// the header EnqueueArgs.hpp.
///
/// The parameter \p templateArgs is a list of stringified template arguments of
/// the CUDA kernel. These strings are used to generate the template
/// instantiation expression of the CUDA kernel during compilation stage. It is
/// critical that these strings are provided in correct format.
///
/// The paramter \p compileOpts is a list of strings that lets you add
/// definitions such as `-D<NAME>` or `-D<NAME>=<VALUE>` to the compiler. To
/// enable easy stringification of variables into their definition equation,
/// three helper macros are provided: TemplateArg, DefineKey and DefineValue.
///
/// Example Usage: transpose
///
/// \code
/// static const std::string src(transpose_cuh, transpose_cuh_len);
/// auto transpose = getKernel("cuda::transpose", src,
///         {
///           TemplateTypename<T>(),
///           TemplateArg(conjugate),
///           TemplateArg(is32multiple)
///         },
///         {
///           DefineValue(TILE_DIM), // Results in a definition
///                                  // "-D TILE_DIME=<Value of TILE_DIM>"
///           DefineValue(THREADS_Y) // Results in a definition
///                                  // "-D THREADS_Y=<Value of THREADS_Y>"
///           DefineKeyValue(DIMY, threads_y)  // Results in a definition
///                                            // "-D DIMY=<Value of threads_y>"
///         }
///         );
/// \endcode
///
/// \param[in] nameExpr is the of name expressions to be instantiated while
///            compiling the kernel.
/// \param[in] source is the kernel source code string
/// \param[in] templateArgs is a vector of strings containing stringified names
///            of the template arguments of CUDA kernel to be compiled.
/// \param[in] compileOpts is a vector of strings that enables the user to
///            add definitions such as `-D<NAME>` or `-D<NAME>=<VALUE>` for
///            the kernel compilation.
///
Kernel getKernel(const std::string& nameExpr, const std::string& source,
                 const std::vector<TemplateArg>& templateArgs,
                 const std::vector<std::string>& compileOpts = {});
}  // namespace cuda
