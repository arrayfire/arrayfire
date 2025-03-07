/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/ModuleInterface.hpp>

#include <cl2hpp.hpp>

namespace arrayfire {
namespace opencl {

/// OpenCL backend wrapper for cl::Program object
class Module : public common::ModuleInterface<cl::Program> {
   public:
    using ModuleType = cl::Program;
    using BaseClass  = common::ModuleInterface<ModuleType>;

    /// \brief Create an uninitialized Module
    Module() = default;

    /// \brief Create a module given a cl::Program type
    Module(ModuleType mod) : BaseClass(mod) {}

    /// \brief Unload module
    operator bool() const final { return get()(); }

    /// Unload the module
    void unload() final { set(cl::Program()); }
};

}  // namespace opencl
}  // namespace arrayfire
