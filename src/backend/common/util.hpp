/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/// This file contains platform independent utility functions

#include <iosfwd>
#include <string>

#pragma once

std::string getEnvVar(const std::string& key);

// Dump the kernel sources only if the environment variable is defined
void saveKernel(const std::string& funcName, const std::string& jit_ker,
                const std::string& ext);

std::string int_version_to_string(int version);
