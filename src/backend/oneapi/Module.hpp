/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/ModuleInterface.hpp>

#include <sycl/sycl.hpp>

namespace arrayfire {
namespace oneapi {

/// oneapi backend wrapper for cl::Program object
class Module
    : public common::ModuleInterface<
          sycl::kernel_bundle<sycl::bundle_state::executable> *> {
   public:
    using ModuleType = sycl::kernel_bundle<sycl::bundle_state::executable> *;
    using BaseClass  = common::ModuleInterface<ModuleType>;

    /// \brief Create an uninitialized Module
    Module() = default;

    /// \brief Create a module given a sycl::program type
    Module(ModuleType mod) : BaseClass(mod) {}

    /// \brief Unload module
    operator bool() const final { return get()->empty(); }

    /// Unload the module
    void unload() final {
        // TODO(oneapi): Unload kernel/program
        ;
    }
};

}  // namespace oneapi
}  // namespace arrayfire
