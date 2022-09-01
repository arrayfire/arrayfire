/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/graphics_common.hpp>

#include <GraphicsResourceManager.hpp>
#include <common/DefaultMemoryManager.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/Logger.hpp>
#include <common/defines.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <err_oneapi.hpp>
//#include <errorcodes.hpp>
#include <version.hpp>
//#include <af/oneapi.h>
#include <af/version.h>
#include <memory>

#ifdef OS_MAC
#include <OpenGL/CGLCurrent.h>
#endif

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using std::begin;
using std::end;
using std::find;
using std::make_unique;
using std::string;
using std::stringstream;
using std::unique_ptr;
using std::vector;
using sycl::device;

namespace oneapi {

bool checkExtnAvailability(const device& pDevice, const string& pName) {
    ONEAPI_NOT_SUPPORTED("");
    return false;
}

DeviceManager::DeviceManager()
    : logger(common::loggerFactory("platform"))
    , mUserDeviceOffset(0)
    , fgMngr(nullptr) {
}

spdlog::logger* DeviceManager::getLogger() { return logger.get(); }

DeviceManager& DeviceManager::getInstance() {
    ONEAPI_NOT_SUPPORTED("");
    static auto* my_instance = new DeviceManager();
    return *my_instance;
}

void DeviceManager::setMemoryManager(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    ONEAPI_NOT_SUPPORTED("");
}

void DeviceManager::resetMemoryManager() {
    ONEAPI_NOT_SUPPORTED("");
}

void DeviceManager::setMemoryManagerPinned(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    ONEAPI_NOT_SUPPORTED("");
}

void DeviceManager::resetMemoryManagerPinned() {
    ONEAPI_NOT_SUPPORTED("");
}

DeviceManager::~DeviceManager() {
    ONEAPI_NOT_SUPPORTED("");
}

void DeviceManager::markDeviceForInterop(const int device,
                                         const void* wHandle) {
    ONEAPI_NOT_SUPPORTED("");
}

}  // namespace oneapi
