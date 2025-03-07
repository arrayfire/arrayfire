/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 * Source:
 * http://nadeausoftware.com/sites/NadeauSoftware.com/files/getMemorySize.c
 */

#include "host_memory.hpp"

#if defined(_WIN32)
#include <Windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
#include <sys/param.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(BSD) && !defined(__gnu_hurd__)
#include <sys/sysctl.h>
#endif

#else
#define NOMEMORYSIZE
#endif

namespace arrayfire {
namespace common {

#ifdef NOMEMORYSIZE
size_t getHostMemorySize() {
    return 0L;  // Can't detect
}

#else

/**
 * Returns the size of physical memory (RAM) in bytes.
 */
size_t getHostMemorySize() {
#if defined(_WIN32) && (defined(__CYGWIN__) || defined(__CYGWIN32__))
    /* Cygwin under Windows. ------------------------------------ */
    /* New 64-bit MEMORYSTATUSEX isn't available.  Use old 32.bit */
    MEMORYSTATUS status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatus(&status);
    return (size_t)status.dwTotalPhys;

#elif defined(_WIN32)
    /* Windows. ------------------------------------------------- */
    /* Use new 64-bit MEMORYSTATUSEX, not old 32-bit MEMORYSTATUS */
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t)status.ullTotalPhys;

#elif defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
    /* UNIX variants. ------------------------------------------- */
    /* Prefer sysctl() over sysconf() except sysctl() HW_REALMEM and HW_PHYSMEM
     */

#if defined(CTL_HW) && (defined(HW_MEMSIZE) || defined(HW_PHYSMEM64))
    int mib[2];
    mib[0]       = CTL_HW;
#if defined(HW_MEMSIZE)
    mib[1]       = HW_MEMSIZE; /* OSX. --------------------- */
#elif defined(HW_PHYSMEM64)
    mib[1] = HW_PHYSMEM64; /* NetBSD, OpenBSD. --------- */
#endif
    int64_t size = 0;          /* 64-bit */
    size_t len   = sizeof(size);
    if (sysctl(mib, 2, &size, &len, NULL, 0) == 0) return (size_t)size;
    return 0L; /* Failed? */

#elif defined(_SC_AIX_REALMEM)
    /* AIX. ----------------------------------------------------- */
    return (size_t)sysconf(_SC_AIX_REALMEM) * (size_t)1024L;

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    /* FreeBSD, Linux, OpenBSD, and Solaris. -------------------- */
    return static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) *
           static_cast<size_t>(sysconf(_SC_PAGESIZE));

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    /* Legacy. -------------------------------------------------- */
    return (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);

#elif defined(CTL_HW) && (defined(HW_PHYSMEM) || defined(HW_REALMEM))
    /* DragonFly BSD, FreeBSD, NetBSD, OpenBSD, and OSX. -------- */
    int mib[2];
    mib[0]            = CTL_HW;
#if defined(HW_REALMEM)
    mib[1]            = HW_REALMEM; /* FreeBSD. ----------------- */
#elif defined(HW_PYSMEM)
    mib[1] = HW_PHYSMEM; /* Others. ------------------ */
#endif
    unsigned int size = 0;          /* 32-bit */
    size_t len        = sizeof(size);
    if (sysctl(mib, 2, &size, &len, NULL, 0) == 0) return (size_t)size;
    return 0L; /* Failed? */
#endif /* sysctl and sysconf variants */

#else
    return 0L; /* Unknown OS. */
#endif
}

#endif  // NOMEMORYSIZE
}  // namespace common
}  // namespace arrayfire
