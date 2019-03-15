/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/MemoryManager.hpp>

#include <iostream>

namespace common
{

int MemoryManagerCWrapper_getActiveDeviceId(af_memory_manager* impl_) {
  return ((MemoryManagerCWrapper*)impl_->wrapper_handle)->getActiveDeviceId();
}

int MemoryManagerCWrapper_getMaxMemorySize(af_memory_manager* impl_, int id) {
  return ((MemoryManagerCWrapper*)impl_->wrapper_handle)->getMaxMemorySize(id);
}

void* MemoryManagerCWrapper_nativeAlloc(af_memory_manager* impl_, size_t size) {
  return ((MemoryManagerCWrapper*)impl_->wrapper_handle)->nativeAlloc(size);
}

void MemoryManagerCWrapper_nativeFree(af_memory_manager* impl_, void* ptr) {
  return ((MemoryManagerCWrapper*)impl_->wrapper_handle)->nativeFree(ptr);
}

} // namespace common
