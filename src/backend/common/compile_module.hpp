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

#include <Module.hpp>
#include <backend.hpp>

#include <nonstd/span.hpp>
#include <string>
#include <vector>

namespace arrayfire {
namespace common {

/// \brief Backend specific source compilation implementation
///
/// This function has to be implemented separately in each backend
///
/// \p kInstances can take of the following two forms depending on backend.
/// - CUDA
///     - A template instantiation style string like transpose<float, true, 1>
///     - The \p kInstances is of size one in almost all cases. These strings
///       are used to generate template instantiations of CUDA kernels while
///       compiling the \p sources.
/// - OpenCL
///     - The \p kInstances parameter is not used.
///
/// \param[in] moduleKey is hash of code+options+instantiations. This is
///            provided by caller to avoid recomputation.
/// \param[in] sources is the list of source code to compile
/// \param[in] options is the list of preprocessor definitions to be passed
///            to the backend compilation function
/// \param[in] kInstances is the name list of kernels in the \p sources
/// \param[in] isJIT is identify if the module being compiled is not
///            hand-written kernel
///
/// \returns Backend specific binary module that contains associated kernel
detail::Module compileModule(const std::string& moduleKey,
                             const nonstd::span<const std::string> sources,
                             const nonstd::span<const std::string> options,
                             const nonstd::span<const std::string> kInstances,
                             const bool isJIT);

/// \brief Load module binary from disk cache
///
/// Note that, this is for internal use by functions that get called from
/// compileModule. The reason it is exposed here is that, it's implementation
/// is partly dependent on backend specifics like program binary loading etc.
/// Exposing this enables each backend to implement it's specifics.
///
/// \param[in] device is the device index
/// \param[in] moduleKey is hash of code+options+instantiations
detail::Module loadModuleFromDisk(const int device,
                                  const std::string& moduleKey,
                                  const bool isJIT);

}  // namespace common
}  // namespace arrayfire

#endif
