/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <string>

// Some compilers create these macros in the header. Causes
// some errors in the Version struct constructor
#ifdef major
#undef major
#endif
#ifdef minor
#undef minor
#endif

namespace arrayfire {
namespace common {
struct Version {
    int major = -1;
    int minor = -1;
    int patch = -1;

    /// Checks if the major version is defined before minor and minor is defined
    /// before patch
    constexpr static bool validate(int major_, int minor_,
                                   int patch_) noexcept {
        return !(major_ < 0 && (minor_ >= 0 || patch_ >= 0)) &&
               !(minor_ < 0 && patch_ >= 0);
    }

    constexpr Version(const int ver_major, const int ver_minor = -1,
                      const int ver_patch = -1) noexcept
        : major(ver_major), minor(ver_minor), patch(ver_patch) {}
};

constexpr bool operator==(const Version& lhs, const Version& rhs) {
    return lhs.major == rhs.major && lhs.minor == rhs.minor &&
           lhs.patch == rhs.patch;
}

constexpr bool operator!=(const Version& lhs, const Version& rhs) {
    return !(lhs == rhs);
}

constexpr static Version NullVersion{-1, -1, -1};

constexpr bool operator<(const Version& lhs, const Version& rhs) {
    if (lhs == NullVersion || rhs == NullVersion) return false;
    if (lhs.major != -1 && rhs.major != -1 && lhs.major < rhs.major)
        return true;
    if (lhs.minor != -1 && rhs.minor != -1 && lhs.minor < rhs.minor)
        return true;
    if (lhs.patch != -1 && rhs.patch != -1 && lhs.patch < rhs.patch)
        return true;
    return false;
}

inline Version fromCudaVersion(size_t version_int) {
    return {static_cast<int>(version_int / 1000),
            static_cast<int>(version_int % 1000) / 10,
            static_cast<int>(version_int % 10)};
}

inline std::string int_version_to_string(int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string(static_cast<int>((version % 1000) / 10.));
}

}  // namespace common
}  // namespace arrayfire
