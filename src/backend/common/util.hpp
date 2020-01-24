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

#include <iosfwd>
#include <string>

std::string getEnvVar(const std::string& key);

// Dump the kernel sources only if the environment variable is defined
void saveKernel(const std::string& funcName, const std::string& jit_ker,
                const std::string& ext);

std::string int_version_to_string(int version);

const std::string& getCacheDirectory();

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

/// Return the FNV-1a hash of the provided bata.
///
/// \param[in] data Binary data to hash
/// \param[in] byteSize Size of the data in bytes
///
/// \returns An unsigned integer representing the hash of the data
std::size_t deterministicHash(const void* data, std::size_t byteSize);

// This is just a wrapper around the above function.
std::size_t deterministicHash(const std::string& data);