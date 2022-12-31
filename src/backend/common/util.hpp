/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/// This file contains platform independent utility functions
#pragma once

#include <optypes.hpp>
#include <af/defines.h>

#include <string>

namespace arrayfire {
namespace common {
/// The environment variable that determines where the runtime kernels
/// will be stored on the file system
constexpr const char* JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME =
    "AF_JIT_KERNEL_CACHE_DIRECTORY";

std::string getEnvVar(const std::string& key);

std::string& ltrim(std::string& s);

// Dump the kernel sources only if the environment variable is defined
void saveKernel(const std::string& funcName, const std::string& jit_ker,
                const std::string& ext);

std::string& getCacheDirectory();

bool directoryExists(const std::string& path);

bool createDirectory(const std::string& path);

bool removeFile(const std::string& path);

bool renameFile(const std::string& sourcePath, const std::string& destPath);

bool isDirectoryWritable(const std::string& path);

/// Return a string suitable for naming a temporary file.
///
/// Every call to this function will generate a new string with a very low
/// probability of colliding with past or future outputs of this function,
/// including calls from other threads or processes. The string contains
/// no extension.
std::string makeTempFilename();

const char* getName(af_dtype type);

std::string getOpEnumStr(af_op_t val);

template<typename T>
std::string toString(T value);
}  // namespace common
}  // namespace arrayfire
