/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdio.h>
#include <string.h>
#include <driver.h>

#ifdef OS_WIN
#include <windows.h>
#include <stdlib.h>
#define snprintf _snprintf

int nvDriverVersion(char *result, int len)
{
#ifndef OS_WIN
    LPCTSTR lptstrFilename = "nvcuda.dll";
    DWORD dwLen, dwHandle;
    LPVOID lpData = NULL;
    VS_FIXEDFILEINFO *lpBuffer;
    unsigned int buflen;
    DWORD version;
    float fversion;
    int rv;

    dwLen = GetFileVersionInfoSize(lptstrFilename, &dwHandle);
    if (dwLen == 0) return 0;

    lpData = malloc(dwLen);
    if (!lpData) return 0;

    rv = GetFileVersionInfo(lptstrFilename, 0, dwLen, lpData);
    if (!rv) return 0;

    rv = VerQueryValue(lpData, "\\", (LPVOID *)&lpBuffer, &buflen);
    if (!rv) return 0;

    version = (HIWORD(lpBuffer->dwFileVersionLS) - 10)*10000 +
               LOWORD(lpBuffer->dwFileVersionLS);
    fversion = version / 100.f;

    snprintf(result, len, "%.2f", fversion);

    free(lpData);
#else
    snprintf(result, len, "%.2f", 0.0);
#endif
    return 1;
}

#else

int nvDriverVersion(char *result, int len)
{
    int pos = 0, epos = 0, i = 0;
    char buffer[1024];
    FILE *f = NULL;

    if (NULL == (f = fopen("/proc/driver/nvidia/version", "r"))) {
        return 0;
    }
    if (fgets(buffer, 1024, f) == NULL) {
        if(f) fclose(f);
        return 0;
    }

    //just close it now since we've already read what we need
    if(f) fclose(f);

    for (i = 1; i < 8; i++) {

        while (buffer[pos] != ' ' && buffer[pos] != '\t')
            if (pos >= 1024 || buffer[pos] == '\0' || buffer[pos] == '\n')
                return 0;
            else
                pos++;
        while (buffer[pos] == ' ' || buffer[pos] == '\t')
            if (pos >= 1024 || buffer[pos] == '\0' || buffer[pos] == '\n')
                return 0;
            else
                pos++;
    }

    epos = pos;
    while (buffer[epos] != ' ' && buffer[epos] != '\t') {
        if (epos >= 1024 || buffer[epos] == '\0' || buffer[epos] == '\n')
            return 0;
        else
            epos++;
    }

    buffer[epos] = '\0';

    strncpy(result, buffer+pos, len);

    return 1;
}

#endif
