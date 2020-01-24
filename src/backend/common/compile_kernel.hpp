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
#include <backend.hpp>

#include <string>
#include <vector>

namespace common {

/// \brief Backend specific kernel compilation implementation
///
/// This function has to be implemented separately in each backend
detail::Kernel compileKernel(const std::string& kernelName,
                             const std::string& templateInstance,
                             const std::vector<std::string>& sources,
                             const std::vector<std::string>& compileOpts,
                             const bool isJIT = false);

/// \brief Load kernel from disk cache
///
/// Note that, this is for internal use by functions that get called from
/// compileKernel. The reason it is exposed here is that, it's implementation
/// is partly dependent on backend specifics like program binary loading etc.
///
/// \p kernelNameExpr can take following values depending on backend
/// -  namespace qualified kernel template instantiation for CUDA
/// -  simple kernel name for OpenCL
/// -  encoded string with KER prefix for JIT
///
/// \param[in] device is the device index
/// \param[in] kernelNameExpr is the name identifying the relevant kernel
/// \param[in] sources is the list of kernel and helper source files
detail::Kernel loadKernel(const int device, const std::string& kernelNameExpr,
                          const std::vector<std::string>& sources);

}  // namespace common

#endif
