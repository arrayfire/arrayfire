/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/// This file contains platform independent utility functions
#if defined(OS_WIN)
#include <Windows.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

#include <common/defines.hpp>
#include <common/util.hpp>
#include <af/defines.h>

#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

string getEnvVar(const std::string& key) {
#if defined(OS_WIN)
    DWORD bufSize =
        32767;  // limit according to GetEnvironment Variable documentation
    string retVal;
    retVal.resize(bufSize);
    bufSize = GetEnvironmentVariable(key.c_str(), &retVal[0], bufSize);
    if (!bufSize) {
        return string("");
    } else {
        retVal.resize(bufSize);
        return retVal;
    }
#else
    char* str = getenv(key.c_str());
    return str == NULL ? string("") : string(str);
#endif
}

const char* getName(af_dtype type) {
    switch (type) {
        case f32: return "float";
        case f64: return "double";
        case c32: return "complex float";
        case c64: return "complex double";
        case u32: return "unsigned int";
        case s32: return "int";
        case u16: return "unsigned short";
        case s16: return "short";
        case u64: return "unsigned long long";
        case s64: return "long long";
        case u8: return "unsigned char";
        case b8: return "bool";
        default: return "unknown type";
    }
}

void saveKernel(const std::string& funcName, const std::string& jit_ker,
                const std::string& ext) {
    static constexpr const char* saveJitKernelsEnvVarName =
        "AF_JIT_KERNEL_TRACE";
    static const char* jitKernelsOutput = getenv(saveJitKernelsEnvVarName);
    if (!jitKernelsOutput) { return; }
    if (std::strcmp(jitKernelsOutput, "stdout") == 0) {
        fputs(jit_ker.c_str(), stdout);
        return;
    }
    if (std::strcmp(jitKernelsOutput, "stderr") == 0) {
        fputs(jit_ker.c_str(), stderr);
        return;
    }
    // Path to a folder
    const std::string ffp =
        std::string(jitKernelsOutput) + AF_PATH_SEPARATOR + funcName + ext;
    FILE* f = fopen(ffp.c_str(), "we");
    if (!f) {
        fprintf(stderr, "Cannot open file %s\n", ffp.c_str());
        return;
    }
    if (fputs(jit_ker.c_str(), f) == EOF) {
        fprintf(stderr, "Failed to write kernel to file %s\n", ffp.c_str());
    }
    fclose(f);
}

std::string int_version_to_string(int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string((int)((version % 1000) / 10.));
}

#if !defined(OS_WIN)
string getHomeDirectory() {
    string home = getEnvVar("XDG_CACHE_HOME");
    if (!home.empty()) return home;

    home = getEnvVar("HOME");
    if (!home.empty()) return home;

    home = getpwuid(getuid())->pw_dir;
}
#else
string getTemporaryDirectory() {
    DWORD bufSize = 261;  // limit according to GetTempPath documentation
    string retVal;
    retVal.resize(bufSize);
    bufSize = GetTempPathA(bufSize, &retVal[0]);
    retVal.resize(bufSize);
    return retVal;
}
#endif

bool folderExists(const string &path) {
#if !defined(OS_WIN)
    struct stat status;
    return stat(path.c_str(), &status) == 0 && (status.st_mode & S_IFDIR) != 0;
#else
    struct _stat status;
    return _stat(path.c_str(), &status) == 0 && (status.st_mode & S_IFDIR) != 0;
#endif
}

bool createFolder(const string &path) {
#if !defined(OS_WIN)
    return mkdir(path.c_str(), 0777) == 0;
#else
    return CreateDirectoryA(path.c_str(), NULL) != 0;
#endif
}

bool removeFile(const string &path) {
#if !defined(OS_WIN)
    return unlink(path.c_str()) == 0;
#else
    return DeleteFileA(path.c_str()) != 0;
#endif
}

bool isDirectoryWritable(const string &path) {
    if (!folderExists(path) && !createFolder(path)) return false;

    const string testPath = path + AF_PATH_SEPARATOR + "test";
    if (!std::ofstream(testPath).is_open()) return false;
    removeFile(testPath);

    return true;
}

const string &getCacheDirectory() {
    thread_local std::once_flag flag;
    thread_local string cacheDirectory;

    std::call_once(flag, []() {
        const vector<string> pathList = {
#if !defined(OS_WIN)
            "/var/lib/arrayfire",
            getHomeDirectory() + "/.arrayfire",
            "/tmp/arrayfire"
#else
            getTemporaryDirectory() + "\\ArrayFire"
#endif
        };

        for (const string &path : pathList) {
            if (isDirectoryWritable(path)) {
                cacheDirectory = path;
                break;
            }
        }
    });

    return cacheDirectory;
}
