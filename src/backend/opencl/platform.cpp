/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Include this before af/opencl.h
// Causes conflict between system cl.hpp and opencl/cl.hpp
#if defined(WITH_GRAPHICS)

#include <graphics_common.hpp>

#if defined(OS_MAC)
#include <OpenGL/OpenGL.h>
#include <libkern/OSAtomic.h>
#else
#include <GL/gl.h>
#endif // !__APPLE__

#endif

#include <af/version.h>
#include <af/opencl.h>
#include <defines.hpp>
#include <version.hpp>
#include <platform.hpp>
#include <util.hpp>
#include <functional>
#include <algorithm>
#include <cctype>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <map>
#include <errorcodes.hpp>
#include <err_opencl.hpp>
#include <util.hpp>
#include <host_memory.hpp>

using std::string;
using std::vector;
using std::ostringstream;
using std::runtime_error;

using cl::Platform;
using cl::Context;
using cl::CommandQueue;
using cl::Device;

namespace opencl
{

#if defined (OS_MAC)
static const std::string CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
static const std::string CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif

static const std::string get_system(void)
{
    std::string arch = (sizeof(void *) == 4) ? "32-bit " : "64-bit ";

    return arch +
#if defined(OS_LNX)
    "Linux";
#elif defined(OS_WIN)
    "Windows";
#elif defined(OS_MAC)
    "Mac OSX";
#endif
}

int getBackend()
{
    return AF_BACKEND_OPENCL;
}

DeviceManager& DeviceManager::getInstance()
{
    static DeviceManager my_instance;
    return my_instance;
}

DeviceManager::~DeviceManager()
{
    //TODO: FIXME:
    // OpenCL libs on Windows platforms
    // are crashing the application at program exit
    // most probably a reference counting issue based
    // on the investigation done so far. This problem
    // doesn't seem to happen on Linux or MacOSX.
    // So, clean up OpenCL resources on non-Windows platforms
#ifndef OS_WIN
    for (auto q: mQueues) delete q;
    for (auto d : mDevices) delete d;
    for (auto c : mContexts) delete c;
#endif
}

void DeviceManager::setContext(int device)
{
    mActiveQId = device;
    mActiveCtxId = device;
}

static inline bool verify_present(std::string pname, const char *ref)
{
    return pname.find(ref) != std::string::npos;
}

static inline bool compare_default(const Device *ldev, const Device *rdev)
{
    const cl_device_type device_types[] = {CL_DEVICE_TYPE_GPU,
                                           CL_DEVICE_TYPE_ACCELERATOR};

    auto l_dev_type = ldev->getInfo<CL_DEVICE_TYPE>();
    auto r_dev_type = rdev->getInfo<CL_DEVICE_TYPE>();

    // This ensures GPU > ACCELERATOR > CPU
    for (auto current_type : device_types) {
        auto is_l_curr_type = l_dev_type == current_type;
        auto is_r_curr_type = r_dev_type == current_type;

        if ( is_l_curr_type && !is_r_curr_type) return true;
        if (!is_l_curr_type &&  is_r_curr_type) return false;
    }

    // For GPUs, this ensures discreet > integrated
    auto is_l_integrared = ldev->getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();
    auto is_r_integrared = rdev->getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();

    if (!is_l_integrared &&  is_r_integrared) return true;
    if ( is_l_integrared && !is_r_integrared) return false;

    // At this point, the devices are of same type.
    // Sort based on emperical evidence of preferred platforms

    // Prefer AMD first
    std::string lPlatName = getPlatformName(*ldev);
    std::string rPlatName = getPlatformName(*rdev);

    if (l_dev_type == CL_DEVICE_TYPE_GPU &&
        r_dev_type == CL_DEVICE_TYPE_GPU ) {
        // If GPU, prefer AMD > NVIDIA > Beignet / Intel > APPLE
        const char *platforms[] = {"AMD", "NVIDIA", "APPLE", "INTEL", "BEIGNET"};

        for (auto ref_name : platforms) {
            if ( verify_present(lPlatName, ref_name) &&
                !verify_present(rPlatName, ref_name)) return true;

            if (!verify_present(lPlatName, ref_name) &&
                 verify_present(rPlatName, ref_name)) return false;
        }

        // Intel falls back to compare based on memory
    } else {
        // If CPU, prefer Intel > AMD > POCL > APPLE
        const char *platforms[] = {"INTEL", "AMD", "POCL", "APPLE"};

        for (auto ref_name : platforms) {
            if ( verify_present(lPlatName, ref_name) &&
                !verify_present(rPlatName, ref_name)) return true;

            if (!verify_present(lPlatName, ref_name) &&
                 verify_present(rPlatName, ref_name)) return false;
        }
    }


    // Compare device compute versions

    {
        // Check Device OpenCL Version
        auto lversion =  ldev->getInfo<CL_DEVICE_VERSION>();
        auto rversion =  rdev->getInfo<CL_DEVICE_VERSION>();

        bool lres = (lversion[7] > rversion[7]) ||
            ((lversion[7] == rversion[7]) && (lversion[9] > rversion[9]));

        bool rres = (lversion[7] < rversion[7]) ||
            ((lversion[7] == rversion[7]) && (lversion[9] < rversion[9]));

        if (lres) return true;
        if (rres) return false;
    }

    // Default crietria, sort based on memory
    // Sort based on memory
    auto l_mem = ldev->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    auto r_mem = rdev->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    return l_mem >= r_mem;
}

static afcl::deviceType getDeviceTypeEnum(cl::Device dev)
{
    return (afcl::deviceType)dev.getInfo<CL_DEVICE_TYPE>();
}


static afcl::platform getPlatformEnum(cl::Device dev)
{
    std::string pname = getPlatformName(dev);
    if (verify_present(pname, "AMD")) return AFCL_PLATFORM_AMD;
    if (verify_present(pname, "NVIDIA")) return AFCL_PLATFORM_NVIDIA;
    if (verify_present(pname, "INTEL")) return AFCL_PLATFORM_INTEL;
    if (verify_present(pname, "APPLE")) return AFCL_PLATFORM_APPLE;
    if (verify_present(pname, "BEIGNET")) return AFCL_PLATFORM_BEIGNET;
    if (verify_present(pname, "POCL")) return AFCL_PLATFORM_POCL;
    return AFCL_PLATFORM_UNKNOWN;
}


DeviceManager::DeviceManager()
    : mUserDeviceOffset(0), mActiveCtxId(0), mActiveQId(0)
{
    try {
        std::vector<cl::Platform>   platforms;
        Platform::get(&platforms);

        // This is all we need because the sort takes care of the order of devices
#ifdef OS_MAC
        cl_device_type DEVICE_TYPES = CL_DEVICE_TYPE_GPU;
#else
        cl_device_type DEVICE_TYPES = CL_DEVICE_TYPE_ALL;
#endif

        std::string deviceENV = getEnvVar("AF_OPENCL_DEVICE_TYPE");

        if (deviceENV.compare("GPU") == 0) {
            DEVICE_TYPES = CL_DEVICE_TYPE_GPU;
        } else if (deviceENV.compare("CPU") == 0) {
            DEVICE_TYPES = CL_DEVICE_TYPE_CPU;
        } else if (deviceENV.compare("ACC") >= 0) {
            DEVICE_TYPES = CL_DEVICE_TYPE_ACCELERATOR;
        }



        // Iterate through platforms, get all available devices and store them
        for (auto &platform : platforms) {
            std::vector<Device> current_devices;

            try {
                platform.getDevices(DEVICE_TYPES, &current_devices);
            } catch(const cl::Error &err) {
                if (err.err() != CL_DEVICE_NOT_FOUND) {
                    throw;
                }
            }

            for (auto dev : current_devices) {
                mDevices.push_back(new Device(dev));
            }
        }

        int nDevices = mDevices.size();

        if (nDevices == 0) AF_ERROR("No OpenCL devices found", AF_ERR_RUNTIME);

        // Sort OpenCL devices based on default criteria
        std::stable_sort(mDevices.begin(), mDevices.end(), compare_default);

        // Create contexts and queues once the sort is done
        for (int i = 0; i < nDevices; i++) {
            cl_platform_id device_platform = mDevices[i]->getInfo<CL_DEVICE_PLATFORM>();
            cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
                                            (cl_context_properties)(device_platform),
                                            0};

            Context *ctx = new Context(*mDevices[i], cps);
            CommandQueue *cq = new CommandQueue(*ctx, *mDevices[i]);
            mContexts.push_back(ctx);
            mQueues.push_back(cq);
            mIsGLSharingOn.push_back(false);
            mDeviceTypes.push_back(getDeviceTypeEnum(*mDevices[i]));
            mPlatforms.push_back(getPlatformEnum(*mDevices[i]));
        }

        bool default_device_set = false;
        deviceENV = getEnvVar("AF_OPENCL_DEFAULT_DEVICE");
        if(!deviceENV.empty()) {
            std::stringstream s(deviceENV);
            int def_device = -1;
            s >> def_device;
            if(def_device < 0 || def_device >= (int)nDevices) {
                printf("WARNING: AF_OPENCL_DEFAULT_DEVICE is out of range\n");
                printf("Setting default device as 0\n");
            } else {
                setContext(def_device);
                default_device_set = true;
            }
        }

        deviceENV = getEnvVar("AF_OPENCL_DEFAULT_DEVICE_TYPE");
        if (!default_device_set && !deviceENV.empty())
        {
            cl_device_type default_device_type = CL_DEVICE_TYPE_GPU;
            if (deviceENV.compare("CPU") == 0) {
                default_device_type = CL_DEVICE_TYPE_CPU;
            } else if (deviceENV.compare("ACC") >= 0) {
                default_device_type = CL_DEVICE_TYPE_ACCELERATOR;
            }

            bool default_device_set = false;
            for (int i = 0; i < nDevices; i++) {
                if (mDevices[i]->getInfo<CL_DEVICE_TYPE>() == default_device_type) {
                    default_device_set = true;
                    setContext(i);
                    break;
                }
            }

            if (!default_device_set) {
                printf("WARNING: AF_OPENCL_DEFAULT_DEVICE_TYPE=%s is not available\n",
                       deviceENV.c_str());
                printf("Using default device as 0\n");
            }
        }

    } catch (const cl::Error &error) {
            CL_TO_AF_ERROR(error);
    }


#if defined(WITH_GRAPHICS)
    // Define AF_DISABLE_GRAPHICS with any value to disable initialization
    std::string noGraphicsENV = getEnvVar("AF_DISABLE_GRAPHICS");
    if(noGraphicsENV.empty()) { // If AF_DISABLE_GRAPHICS is not defined
        try {
            /* loop over devices and replace contexts with
             * OpenGL shared contexts whereever applicable */
            int devCount = mDevices.size();
            fg::Window* wHandle = graphics::ForgeManager::getInstance().getMainWindow();
            for(int i=0; i<devCount; ++i)
                markDeviceForInterop(i, wHandle);
        } catch (...) {
        }
    }
#endif
    mUserDeviceOffset = mDevices.size();
}


// http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring/217605#217605
// trim from start
static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                    std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

static std::string platformMap(std::string &platStr)
{
    static bool isFirst = true;

    typedef std::map<std::string, std::string> strmap_t;
    static strmap_t platMap;
    if (isFirst) {
        platMap["NVIDIA CUDA"]                         = "NVIDIA  ";
        platMap["Intel(R) OpenCL"]                     = "INTEL   ";
        platMap["AMD Accelerated Parallel Processing"] = "AMD     ";
        platMap["Intel Gen OCL Driver"]                = "BEIGNET ";
        platMap["Apple"]                               = "APPLE   ";
        platMap["Portable Computing Language"]         = "POCL    ";
        isFirst = false;
    }

    strmap_t::iterator idx = platMap.find(platStr);

    if (idx == platMap.end()) {
        return platStr;
    } else {
        return idx->second;
    }
}

std::string getDeviceInfo()
{
    ostringstream info;
    info << "ArrayFire v" << AF_VERSION
         << " (OpenCL, " << get_system() << ", build " << AF_REVISION << ")" << std::endl;

    unsigned nDevices = 0;
    for(auto &device: DeviceManager::getInstance().mDevices) {
        const Platform platform(device->getInfo<CL_DEVICE_PLATFORM>());

        string dstr = device->getInfo<CL_DEVICE_NAME>();

        // Remove null termination character from the strings
        dstr.pop_back();

        bool show_braces = ((unsigned)getActiveDeviceId() == nDevices);

        string id =
            (show_braces ? string("[") : "-") +
            std::to_string(nDevices) +
            (show_braces ? string("]") : "-");

        size_t msize = device->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
        info << id << " " << getPlatformName(*device) << ": " << ltrim(dstr)
             << ", " << msize / 1048576 << " MB";
#ifndef NDEBUG
        info << " -- ";
        string devVersion = device->getInfo<CL_DEVICE_VERSION>();
        string driVersion = device->getInfo<CL_DRIVER_VERSION>();
        devVersion.pop_back();
        driVersion.pop_back();
        info << devVersion;
        info << " -- Device driver " << driVersion;
        info << " -- FP64 Support: "
             << (device->getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>()>0 ? "True" : "False");
        info << " -- Unified Memory ("
             << (isHostUnifiedMemory(*device) ? "True" : "False")
             << ")";
#endif
        info << std::endl;

        nDevices++;
    }
    return info.str();
}

std::string getPlatformName(const cl::Device &device)
{
    const Platform platform(device.getInfo<CL_DEVICE_PLATFORM>());
    std::string platStr = platform.getInfo<CL_PLATFORM_NAME>();

    // BELOW NULL TERMINATION character removal was required with
    // cl.hpp header, however with cl2.hpp this is not needed anymore.
    //
    // Remove null termination character from the strings
    //platStr.pop_back();

    return platformMap(platStr);
}

int getDeviceCount()
{
    return DeviceManager::getInstance().mQueues.size();
}

int getActiveDeviceId()
{
    return DeviceManager::getInstance().mActiveQId;
}

int getDeviceIdFromNativeId(cl_device_id id)
{
    DeviceManager& devMngr = DeviceManager::getInstance();
    int nDevices = devMngr.mDevices.size();
    int devId = 0;
    for (devId=0; devId<nDevices; ++devId) {
        if (id == devMngr.mDevices[devId]->operator()())
            break;
    }
    return devId;
}

int getActiveDeviceType()
{
    DeviceManager &instance = DeviceManager::getInstance();
    return instance.mDeviceTypes[instance.mActiveQId];
}

int getActivePlatform()
{
    DeviceManager &instance = DeviceManager::getInstance();
    return instance.mPlatforms[instance.mActiveQId];
}
const Context& getContext()
{
    DeviceManager& devMngr = DeviceManager::getInstance();
    return *(devMngr.mContexts[devMngr.mActiveCtxId]);
}

CommandQueue& getQueue()
{
    DeviceManager& devMngr = DeviceManager::getInstance();
    return *(devMngr.mQueues[devMngr.mActiveQId]);
}

const cl::Device& getDevice(int id)
{
    DeviceManager& devMngr = DeviceManager::getInstance();
    if(id == -1) id = devMngr.mActiveQId;
    return *(devMngr.mDevices[id]);
}

size_t getDeviceMemorySize(int device)
{
    const cl::Device& dev = getDevice(device);
    size_t msize = dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    return msize;
}

size_t getHostMemorySize()
{
    return common::getHostMemorySize();
}

cl_device_type getDeviceType()
{
    cl::Device device = getDevice();
    cl_device_type type = device.getInfo<CL_DEVICE_TYPE>();
    return type;
}

bool isHostUnifiedMemory(const cl::Device &device)
{
    return device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();
}

bool OpenCLCPUOffload(bool forceOffloadOSX)
{
    static const bool offloadEnv = getEnvVar("AF_OPENCL_CPU_OFFLOAD") == "1";
    bool offload = false;
    if(offloadEnv) offload = isHostUnifiedMemory(getDevice());
#if OS_MAC
    // FORCED OFFLOAD FOR LAPACK FUNCTIONS ON OSX UNIFIED MEMORY DEVICES
    //
    // On OSX Unified Memory devices (Intel), always offload LAPACK but not GEMM
    // irrespective of the AF_OPENCL_CPU_OFFLOAD value
    // From GEMM, OpenCLCPUOffload(false) is called which will render the
    // variable inconsequential to the returned result.
    //
    // Issue https://github.com/arrayfire/arrayfire/issues/662
    offload = offload || forceOffloadOSX;
#endif
    return offload;
}

bool isGLSharingSupported()
{
    DeviceManager& devMngr = DeviceManager::getInstance();
    return devMngr.mIsGLSharingOn[devMngr.mActiveQId];
}

bool isDoubleSupported(int device)
{
    DeviceManager& devMngr = DeviceManager::getInstance();
    return (devMngr.mDevices[device]->getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>()>0);
}

void devprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
{
    unsigned nDevices = 0;
    unsigned currActiveDevId = (unsigned)getActiveDeviceId();
    bool devset = false;

    for (auto context : DeviceManager::getInstance().mContexts) {
        vector<Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();

        for (auto &device : devices) {
            const Platform platform(device.getInfo<CL_DEVICE_PLATFORM>());
            string platStr = platform.getInfo<CL_PLATFORM_NAME>();

            if (currActiveDevId == nDevices) {
                string dev_str;
                device.getInfo(CL_DEVICE_NAME, &dev_str);
                string com_str = device.getInfo<CL_DEVICE_VERSION>();
                com_str = com_str.substr(7, 3);

                // strip out whitespace from the device string:
                const std::string& whitespace = " \t";
                const auto strBegin = dev_str.find_first_not_of(whitespace);
                const auto strEnd = dev_str.find_last_not_of(whitespace);
                const auto strRange = strEnd - strBegin + 1;
                dev_str = dev_str.substr(strBegin, strRange);

                // copy to output
                snprintf(d_name, 64, "%s", dev_str.c_str());
                snprintf(d_platform, 10, "OpenCL");
                snprintf(d_toolkit, 64, "%s", platStr.c_str());
                snprintf(d_compute, 10, "%s", com_str.c_str());
                devset = true;
            }
            if(devset) break;
            nDevices++;
        }
        if(devset) break;
    }

    // Sanitize input
    for (int i = 0; i < 31; i++) {
        if (d_name[i] == ' ') {
            if (d_name[i + 1] == 0 || d_name[i + 1] == ' ') d_name[i] = 0;
            else d_name[i] = '_';
        }
    }
}

int setDevice(int device)
{
    DeviceManager& devMngr = DeviceManager::getInstance();

    if (device >= (int)devMngr.mQueues.size() ||
            device>= (int)DeviceManager::MAX_DEVICES) {
        //throw runtime_error("@setDevice: invalid device index");
        return -1;
    }
    else {
        int old = devMngr.mActiveQId;
        devMngr.setContext(device);
        return old;
    }
}

void sync(int device)
{
    try {
        int currDevice = getActiveDeviceId();
        setDevice(device);
        getQueue().finish();
        setDevice(currDevice);
    } catch (const cl::Error &ex) {
        CL_TO_AF_ERROR(ex);
    }
}

bool checkExtnAvailability(const Device &pDevice, std::string pName)
{
    bool ret_val = false;
    // find the extension required
    std::string exts = pDevice.getInfo<CL_DEVICE_EXTENSIONS>();
    std::stringstream ss(exts);
    std::string item;
    while (std::getline(ss,item,' ')) {
        if (item==pName) {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

#if defined(WITH_GRAPHICS)
void DeviceManager::markDeviceForInterop(const int device, const fg::Window* wHandle)
{
    try {
        if (device >= (int)mQueues.size() ||
                device>= (int)DeviceManager::MAX_DEVICES) {
            throw cl::Error(CL_INVALID_DEVICE, "Invalid device passed for CL-GL Interop");
        } else {
            mQueues[device]->finish();

            // check if the device has CL_GL sharing extension enabled
            bool temp = checkExtnAvailability(*mDevices[device], CL_GL_SHARING_EXT);
            if (!temp) {
                /* return silently if given device has not OpenGL sharing extension
                 * enabled so that regular queue is used for it */
                return;
            }

            // call forge to get OpenGL sharing context and details
            cl::Platform plat(mDevices[device]->getInfo<CL_DEVICE_PLATFORM>());

#ifdef OS_MAC
            CGLContextObj cgl_current_ctx = CGLGetCurrentContext();
            CGLShareGroupObj cgl_share_group = CGLGetShareGroup(cgl_current_ctx);

            cl_context_properties cps[] = {
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)cgl_share_group,
                0
            };
#else
            cl_context_properties cps[] = {
                CL_GL_CONTEXT_KHR, (cl_context_properties)wHandle->context(),
#if defined(_WIN32) || defined(_MSC_VER)
                CL_WGL_HDC_KHR, (cl_context_properties)wHandle->display(),
#else
                CL_GLX_DISPLAY_KHR, (cl_context_properties)wHandle->display(),
#endif
                CL_CONTEXT_PLATFORM, (cl_context_properties)plat(),
                0
            };

            // Check if current OpenCL device is belongs to the OpenGL context
            {
                cl_context_properties test_cps[] = {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)wHandle->context(),
                    CL_CONTEXT_PLATFORM, (cl_context_properties)plat(),
                    0
                };

                // Load the extension
                // If cl_khr_gl_sharing is available, this function should be present
                // This has been checked earlier, it comes to this point only if it is found
                auto func = (clGetGLContextInfoKHR_fn)
                    clGetExtensionFunctionAddressForPlatform(plat(), "clGetGLContextInfoKHR");

                // If the function doesn't load, bail early
                if (!func) return;

                // Get all devices associated with opengl context
                std::vector<cl_device_id> devices(16);
                size_t ret = 0;
                cl_int err = func(test_cps,
                                  CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                  devices.size() * sizeof(cl_device_id),
                                  &devices[0],
                                  &ret);
                if (err != CL_SUCCESS) return;
                int num = ret / sizeof(cl_device_id);
                devices.resize(num);

                // Check if current device is present in the associated devices
                cl_device_id current_device = (*mDevices[device])();
                auto res = std::find(std::begin(devices),
                                     std::end(devices),
                                     current_device);

                if (res == std::end(devices)) return;
            }
#endif

            // Change current device to use GL sharing
            Context * ctx = new Context(*mDevices[device], cps);
            CommandQueue * cq = new CommandQueue(*ctx, *mDevices[device]);

            // May be fixes the AMD GL issues we see on windows?
#if !defined(_WIN32) && !defined(_MSC_VER)
            delete mContexts[device];
            delete mQueues[device];
#endif

            mContexts[device] = ctx;
            mQueues[device] = cq;
            mIsGLSharingOn[device] = true;
        }
    } catch (const cl::Error &ex) {
        /* If replacing the original context with GL shared context
         * failes, don't throw an error and instead fall back to
         * original context and use copy via host to support graphics
         * on that particular OpenCL device. So mark it as no GL sharing */
    }
}
#endif

void addDeviceContext(cl_device_id dev, cl_context ctx, cl_command_queue que)
{
    try {

        clRetainDevice(dev);
        clRetainContext(ctx);
        clRetainCommandQueue(que);

        DeviceManager& devMngr   = DeviceManager::getInstance();
        cl::Device* tDevice      = new cl::Device(dev);
        cl::Context* tContext    = new cl::Context(ctx);
        cl::CommandQueue* tQueue = (que==NULL ?
                new cl::CommandQueue(*tContext, *tDevice) : new cl::CommandQueue(que));
        devMngr.mDevices.push_back(tDevice);
        devMngr.mContexts.push_back(tContext);
        devMngr.mQueues.push_back(tQueue);
        devMngr.mPlatforms.push_back(getPlatformEnum(*tDevice));
        // FIXME: add OpenGL Interop for user provided contexts later
        devMngr.mIsGLSharingOn.push_back(false);
    } catch (const cl::Error &ex) {
        CL_TO_AF_ERROR(ex);
    }
}

void setDeviceContext(cl_device_id dev, cl_context ctx)
{
    // FIXME: add OpenGL Interop for user provided contexts later
    try {
        DeviceManager& devMngr = DeviceManager::getInstance();
        const int dCount = devMngr.mDevices.size();
        for (int i=0; i<dCount; ++i) {
            if(devMngr.mDevices[i]->operator()()==dev &&
                    devMngr.mContexts[i]->operator()()==ctx) {
                setDevice(i);
                return;
            }
        }
    } catch (const cl::Error &ex) {
        CL_TO_AF_ERROR(ex);
    }
    AF_ERROR("No matching device found", AF_ERR_ARG);
}

void removeDeviceContext(cl_device_id dev, cl_context ctx)
{
    try {
        if (getDevice()() == dev && getContext()()==ctx) {
            AF_ERROR("Cannot pop the device currently in use", AF_ERR_ARG);
        }

        DeviceManager& devMngr = DeviceManager::getInstance();
        const int dCount = devMngr.mDevices.size();
        int deleteIdx = -1;
        for (int i = 0; i<dCount; ++i) {
            if(devMngr.mDevices[i]->operator()()==dev &&
                    devMngr.mContexts[i]->operator()()==ctx) {
                deleteIdx = i;
                break;
            }
        }
        if (deleteIdx < (int)devMngr.mUserDeviceOffset) {
            AF_ERROR("Cannot pop ArrayFire internal devices", AF_ERR_ARG);
        } else if (deleteIdx == -1) {
            AF_ERROR("No matching device found", AF_ERR_ARG);
        } else {

            clReleaseDevice((*devMngr.mDevices[deleteIdx])());
            clReleaseContext((*devMngr.mContexts[deleteIdx])());
            clReleaseCommandQueue((*devMngr.mQueues[deleteIdx])());

            // FIXME: this case can potentially cause issues due to the
            // modification of the device pool stl containers.

            // IF the current active device is enumerated at a position
            // that lies ahead of the device that has been requested
            // to be removed. We just pop the entries from pool since it
            // has no side effects.
            devMngr.mDevices.erase(devMngr.mDevices.begin()+deleteIdx);
            devMngr.mContexts.erase(devMngr.mContexts.begin()+deleteIdx);
            devMngr.mQueues.erase(devMngr.mQueues.begin()+deleteIdx);
            devMngr.mPlatforms.erase(devMngr.mPlatforms.begin()+deleteIdx);
            // FIXME: add OpenGL Interop for user provided contexts later
            devMngr.mIsGLSharingOn.erase(devMngr.mIsGLSharingOn.begin()+deleteIdx);
            // OTHERWISE, update(decrement) the `mActive*Id` variables
            if (deleteIdx < (int)devMngr.mActiveCtxId) {
                --devMngr.mActiveCtxId;
                --devMngr.mActiveQId;
            }
        }
    } catch (const cl::Error &ex) {
        CL_TO_AF_ERROR(ex);
    }
}

bool synchronize_calls() {
    static bool sync = getEnvVar("AF_SYNCHRONOUS_CALLS") == "1";
    return sync;
}


unsigned getMaxJitSize()
{
    const int MAX_JIT_LEN = 20;

    static int length = 0;
    if (length == 0) {
        std::string env_var = getEnvVar("AF_OPENCL_MAX_JIT_LEN");
        if (!env_var.empty()) {
            length = std::stoi(env_var);
        } else {
            length = MAX_JIT_LEN;
        }
    }
    return length;
}

bool& evalFlag()
{
    static bool flag = true;
    return flag;
}

}

using namespace opencl;

af_err afcl_get_device_type(afcl_device_type *res)
{
    try {
        *res = (afcl_device_type)getActiveDeviceType();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_get_platform(afcl_platform *res)
{
    try {
        *res = (afcl_platform)getActivePlatform();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_get_context(cl_context *ctx, const bool retain)
{
    try {
        *ctx = getContext()();
        if (retain) clRetainContext(*ctx);
    } CATCHALL;
    return AF_SUCCESS;
}


af_err afcl_get_queue(cl_command_queue *queue, const bool retain)
{
    try {
        *queue = getQueue()();
        if (retain) clRetainCommandQueue(*queue);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_get_device_id(cl_device_id *id)
{
    try {
        *id = getDevice()();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_set_device_id(cl_device_id id)
{
    try {
        setDevice(getDeviceIdFromNativeId(id));
    } CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_add_device_context(cl_device_id dev, cl_context ctx, cl_command_queue que)
{
    try {
        addDeviceContext(dev, ctx, que);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_set_device_context(cl_device_id dev, cl_context ctx)
{
    try {
        setDeviceContext(dev, ctx);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err afcl_delete_device_context(cl_device_id dev, cl_context ctx)
{
    try {
        removeDeviceContext(dev, ctx);
    } CATCHALL;
    return AF_SUCCESS;
}
