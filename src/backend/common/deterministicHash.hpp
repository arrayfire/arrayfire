/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <nonstd/span.hpp>
#include <string>

#include <common/Source.hpp>

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
std::size_t deterministicHash(nonstd::span<const std::string> list,
                              const std::size_t prevHash = FNV1A_BASE_OFFSET);

// This concatenates hashes of multiple sources
std::size_t deterministicHash(
    nonstd::span<const arrayfire::common::Source> list);
