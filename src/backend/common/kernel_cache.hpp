/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if !defined(AF_CPU)

#include <Kernel.hpp>
#include <Module.hpp>
#include <backend.hpp>
#include <common/Source.hpp>
#include <common/TemplateTypename.hpp>
#include <common/util.hpp>

#include <nonstd/span.hpp>
#include <string>
#include <vector>

namespace arrayfire {
namespace common {

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
/// The paramter \p options is a list of strings that lets you add
/// definitions such as `-D<NAME>` or `-D<NAME>=<VALUE>` to the compiler. To
/// enable easy stringification of variables into their definition equation,
/// three helper macros are provided: TemplateArg, DefineKey and DefineValue.
///
/// Example Usage: transpose
///
/// \code
/// auto transpose = getKernel("arrayfire::cuda::transpose",
/// {{transpase_cuh_src}},
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
/// \param[in] sources is the list of common::Source to be compiled if required
/// \param[in] templateArgs is a vector of strings containing stringified names
///            of the template arguments of kernel to be compiled.
/// \param[in] options is a vector of strings that enables the user to
///            add definitions such as `-D<NAME>` or `-D<NAME>=<VALUE>` for
///            the kernel compilation.
///
detail::Kernel getKernel(const std::string& kernelName,
                         const nonstd::span<const common::Source> sources,
                         const nonstd::span<const TemplateArg> templateArgs,
                         const nonstd::span<const std::string> options = {},
                         const bool sourceIsJIT                        = false);

/// \brief Lookup a Module that matches the given key
///
/// This function is intended to be used by JIT only. Usage in other
/// places will most likely result in Module{nullptr}. If by
/// chance you do get a match for non-jit usage, it is accidental and
/// such Module will not work as expected.
///
/// \param[in] device is index of device in given backend for which
///            the module look up has to be done
/// \param[in] key is hash generated from code + options + kernel_name
///            at caller scope
detail::Module findModule(const int device, const std::size_t& key);

/// \brief Get Kernel object for given name from given Module
///
/// This function is intended to be used by JIT and compileKernel only.
/// Usage in other places may have undefined behaviour.
///
/// \param[in] mod is cache entry from module map.
/// \param[in] name is actual kernel name or it's template instantiation
/// \param[in] sourceWasJIT is used to fetch mangled name for given module
///            associated with \p name
detail::Kernel getKernel(const detail::Module& mod, const std::string& name,
                         const bool sourceWasJIT);

}  // namespace common
}  // namespace arrayfire

#endif
