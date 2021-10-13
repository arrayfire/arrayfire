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

#include <af/defines.h>

#include <iosfwd>
#include <string>
#include <vector>

namespace common {
struct Source {
    const char* ptr;           // Pointer to the kernel source
    const std::size_t length;  // Length of the kernel source
    const std::size_t hash;    // hash value for the source *ptr;
};
}  // namespace common

/// The environment variable that determines where the runtime kernels
/// will be stored on the file system
constexpr const char* JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME =
    "AF_JIT_KERNEL_CACHE_DIRECTORY";

std::string getEnvVar(const std::string& key);

// Dump the kernel sources only if the environment variable is defined
void saveKernel(const std::string& funcName, const std::string& jit_ker,
                const std::string& ext);

std::string int_version_to_string(int version);

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

/// Return the FNV-1a hash of the provided bata.
///
/// \param[in] data Binary data to hash
/// \param[in] byteSize Size of the data in bytes
/// \param[in] optional prevHash Hash of previous parts when string is split
///
/// \returns An unsigned integer representing the hash of the data
constexpr std::size_t FNV1A_BASE_OFFSET = 0x811C9DC5;
constexpr std::size_t FNV1A_PRIME       = 0x01000193;
std::size_t deterministicHash(const void* data, std::size_t byteSize,
                              const std::size_t prevHash = FNV1A_BASE_OFFSET);

// This is just a wrapper around the above function.
std::size_t deterministicHash(const std::string& data,
                              const std::size_t prevHash = FNV1A_BASE_OFFSET);

// This concatenates strings in the vector and computes hash
std::size_t deterministicHash(const std::vector<std::string>& list,
                              const std::size_t prevHash = FNV1A_BASE_OFFSET);

// This concatenates hashes of multiple sources
std::size_t deterministicHash(const std::vector<common::Source>& list);
