/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <backend.hpp>
#include <common/kernel_util.hpp>
#include <kernel_enqueuer.hpp>

#include <string>
#include <vector>

namespace common {

/// Kernel stores kernel module/program along with handle for compiled kernel
struct Kernel {
    detail::ModuleType prog;
    detail::KernelType kern;

    /// \brief Get device pointer associated with name(label)
    ///
    /// This function is only useful with CUDA NVRTC based compilation
    /// at the moment, calling this function for OpenCL backend build
    /// will return a null pointer.
    detail::DevPtrType get(const char* name);

    /// \brief Copy data from device memory to read-only memory
    ///
    /// This function copies data of `bytes` size from the device pointer to a
    /// read-only memory.
    ///
    /// \param[in] dst is the device pointer to which data will be copied
    /// \param[in] src is the device pointer from which data will be copied
    /// \param[in] bytes are the number of bytes of data to be copied
    ///
    void copyToReadOnly(detail::DevPtrType dst, detail::DevPtrType src,
                        size_t bytes);

    /// \brief Copy a single scalar to device memory
    ///
    /// This function copies a single value of type T from host variable
    /// to the device memory pointed by `dst`
    ///
    /// \param[in] dst is the device pointer to which data will be copied
    /// \param[in] value is the value of type T
    ///
    template<typename T>
    void setScalar(detail::DevPtrType dst, T value);

    /// \brief Fetch a scalar from device memory
    ///
    /// This function copies a single value of type T from device memory
    ///
    /// \param[inout] value is the value of type T
    /// \param[in] src is the device pointer from which data will be copied
    ///
    template<typename T>
    void getScalar(T& out, detail::DevPtrType src);

    /// \brief Enqueue Kernel per queueing criteria forwarding other parameters
    ///
    /// This operator overload enables Kernel object to work as functor that
    /// internally executes the kernel stored in the Kernel object.
    /// All parameters that are passed in after the EnqueueArgs object are
    /// essentially forwarded to kenel launch API
    ///
    /// \param[in] qArgs is an object of struct \ref cl::EnqueueArgs
    /// \param[in] args is the placeholder for variadic arguments
    template<typename... Args>
    void operator()(const detail::EnqueueArgs& qArgs, Args... args) {
        detail::Enqueuer launch;
        launch(kern, qArgs, std::forward<Args>(args)...);
    }
};

/// \brief Backend specific kernel compilation implementation
///
/// This function has to be implemented separately in each backend
void compileKernel(common::Kernel& out, const std::string& kernelName,
                   const std::string& templateInstance,
                   const std::vector<std::string>& sources,
                   const std::vector<std::string>& compileOpts,
                   const bool isJIT = false);

/// \brief Find/Create-Cache a Kernel that fits the given criteria
///
/// This function takes in two vectors of strings apart from the main Kernel
/// name, match criteria, to find a suitable kernel in the Kernel cache. It
/// builds and caches a new Kernel object if one isn't found in the cache.
///
/// The paramter \p key has to be the unique name for a given kernel.
/// The key has to be present in one of the entries of KernelMap defined in
/// the header EnqueueArgs.hpp.
///
/// The parameter \p templateArgs is a list of stringified template arguments of
/// the kernel. These strings are used to generate the template instantiation
/// expression of the kernel during compilation stage. This string is used as
/// key to kernel cache map. At some point in future, the idea is to use these
/// instantiation strings to generate template instatiations in online compiler.
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
/// auto transpose = getKernel("cuda::transpose", {src},
///         {
///           TemplateTypename<T>(),
///           TemplateArg(conjugate),
///           TemplateArg(is32multiple)
///         },
///         {
///           DefineValue(THREADS_Y) // Results in a definition
///                                  // "-D THREADS_Y=<Value of THREADS_Y>"
///           DefineKeyValue(DIMY, threads_y)  // Results in a definition
///                                            // "-D DIMY=<Value of threads_y>"
///         }
///         );
/// \endcode
///
/// \param[in] kernelName is the name of the kernel qualified as kernel in code
/// \param[in] sources is the list of source strings to be compiled if required
/// \param[in] templateArgs is a vector of strings containing stringified names
///            of the template arguments of kernel to be compiled.
/// \param[in] compileOpts is a vector of strings that enables the user to
///            add definitions such as `-D<NAME>` or `-D<NAME>=<VALUE>` for
///            the kernel compilation.
///
Kernel findKernel(const std::string& kernelName,
                  const std::vector<std::string>& sources,
                  const std::vector<TemplateArg>& templateArgs,
                  const std::vector<std::string>& compileOpts = {});

}  // namespace common
