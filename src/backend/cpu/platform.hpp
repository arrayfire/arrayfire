/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <queue.hpp>
#include <string>

namespace graphics {
class ForgeManager;
}

namespace common {
namespace memory {
class MemoryManagerBase;
}
}  // namespace common

using common::memory::MemoryManagerBase;

namespace cpu {

int getBackend();

std::string getDeviceInfo() noexcept;

bool isDoubleSupported(int device);

bool isHalfSupported(int device);

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

unsigned getMaxJitSize();

int getDeviceCount();

unsigned getActiveDeviceId();

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

int setDevice(int device);

queue& getQueue(int device = 0);

void sync(int device);

bool& evalFlag();

MemoryManagerBase& memoryManager();

void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManager();

// Pinned memory not supported
void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManagerPinned();

graphics::ForgeManager& forgeManager();

}  // namespace cpu
