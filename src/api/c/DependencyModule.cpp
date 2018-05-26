
#include <DependencyModule.hpp>
#include <algorithm>
#include <string>

using std::string;
namespace {

#if(OS_WIN)
    static const char* librarySuffix = ".dll";
    static const char* libraryPrefix = "";

    LibHandle loadLibrary(const char* library_name) {
        return LoadLibrary(library_name);
    }

    void unloadLibrary(LibHandle handle) {
        FreeLibrary(handle);
    }

    string getErrorMessage() {
        LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;
        DWORD dw = GetLastError();

        size_t characters_in_message;
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                      FORMAT_MESSAGE_FROM_SYSTEM |
                      FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL,
                      dw,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR) &lpMsgBuf,
                      0, NULL );
        string error_message(lpMsgBuf);
        return error_message;
    }

#else
#if(OS_MAC)
    static const char* librarySuffix = ".dylib";
#else
    static const char* librarySuffix = ".so";
#endif

    static const char* libraryPrefix = "lib";

    LibHandle loadLibrary(const char* library_name) {
        return dlopen(library_name, RTLD_LAZY);
    }
    void unloadLibrary(LibHandle handle) {
        dlclose(handle);
    }

    string getErrorMessage() {
        string error_message(dlerror());
        return error_message;
    }

#endif

    std::string libName(std::string name) {
        return libraryPrefix + name + librarySuffix;
    }
}

#if(OS_WIN)
void* DependencyModule::getFunctionPointer(LibHandle handle, const char* name) {
    return GetProcAddress(handle, symbolName);
}
#else
void* DependencyModule::getFunctionPointer(LibHandle handle, const char* name) {
    return dlsym(handle, name);
}
#endif

DependencyModule::DependencyModule(const char* plugin_file_name, const char** paths)
    : handle(nullptr) {
    // TODO(umar): Implement handling of non-standard paths
    if(plugin_file_name) {
        handle = loadLibrary(libName(plugin_file_name).c_str());
    }
}

DependencyModule::~DependencyModule() {
    if(handle) {
        unloadLibrary(handle);
    }
}

bool DependencyModule::isLoaded() {
    return (bool)handle;
}

bool DependencyModule::symbolsLoaded() {
    return all_of(begin(functions), end(functions), [](void* ptr){ return ptr != nullptr; });
}

string DependencyModule::getErrorMessage() {
    return ::getErrorMessage();
}
