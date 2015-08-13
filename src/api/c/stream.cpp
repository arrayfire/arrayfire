/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include <af/array.h>
#include <ArrayInfo.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <err_common.hpp>
#include <type_util.hpp>

#include <af/index.h>

using namespace detail;

#define STREAM_FORMAT_VERSION 0x1
static const char sfv_char = STREAM_FORMAT_VERSION;

template<typename T>
static int save(const char *key, const af_array arr, const char *filename, const bool append = false)
{
    // (char     )   Version (Once)
    // (int      )   No. of Arrays (Once)
        // (int    )   Length of the key
        // (cstring)   Key
        // (intl   )   Offset bytes to next array (type + dims + data)
        // (char   )   Type
        // (intl   )   dim4 (x 4)
        // (T      )   data (x elements)

    // Setup all the data structures that need to be written to file
    ///////////////////////////////////////////////////////////////////////////
    std::string k(key);
    int klen = k.size();

    const ArrayInfo info = getInfo(arr);
    std::vector<T> data(info.elements());

    AF_CHECK(af_get_data_ptr(&data.front(), arr));

    char type = info.getType();

    intl odims[4];
    for(int i = 0; i < 4; i++) {
        odims[i] = info.dims()[i];
    }

    intl offset = sizeof(char) + 4 * sizeof(intl) + info.elements() * sizeof(T);
    ///////////////////////////////////////////////////////////////////////////

    std::fstream fs;
    int n_arrays = 0;

    if(append) {
        std::ifstream checkIfExists(filename);
        bool exists = checkIfExists.good();
        checkIfExists.close();
        if(exists) {
            fs.open(filename, std::fstream::in | std::fstream::out | std::fstream::binary);
        } else {
            fs.open(filename, std::fstream::out | std::fstream::binary);
        }

        // Throw exception if file is not open
        if(!fs.is_open()) AF_ERROR("File failed to open", AF_ERR_ARG);

        // Assert Version
        if(fs.peek() == std::fstream::traits_type::eof()) {
            // File is empty
            fs.clear();
        } else {
            char prev_version = 0;
            fs.read(&prev_version, sizeof(char));

            AF_ASSERT(prev_version == sfv_char, "ArrayFire data format has changed. Can't append to file");

            fs.read((char*)&n_arrays, sizeof(int));
        }
    } else {
        fs.open(filename, std::fstream::out | std::fstream::binary | std::fstream::trunc);

        // Throw exception if file is not open
        if(!fs.is_open()) AF_ERROR("File failed to open", AF_ERR_ARG);
    }

    n_arrays++;

    // Write version and n_arrays to top of file
    fs.seekp(0);
    fs.write(&sfv_char, 1);
    fs.write((char*)&n_arrays, sizeof(int));

    // Write array to end of file. Irrespective of new or append
    fs.seekp(0, std::ios_base::end);
    fs.write((char*)&klen, sizeof(int));
    fs.write(k.c_str(), klen);
    fs.write((char*)&offset, sizeof(intl));
    fs.write(&type, sizeof(char));
    fs.write((char*)&odims, sizeof(intl) * 4);
    fs.write((char*)&data.front(), sizeof(T) * data.size());
    fs.close();

    return n_arrays - 1;
}

af_err af_save_array(int *index, const char *key, const af_array arr, const char *filename, const bool append)
{
    try {
        ARG_ASSERT(0, key != NULL);
        ARG_ASSERT(2, filename != NULL);

        ArrayInfo info = getInfo(arr);
        af_dtype type = info.getType();
        int id = -1;
        switch(type) {
            case f32:   id = save<float>   (key, arr, filename, append);   break;
            case c32:   id = save<cfloat>  (key, arr, filename, append);   break;
            case f64:   id = save<double>  (key, arr, filename, append);   break;
            case c64:   id = save<cdouble> (key, arr, filename, append);   break;
            case b8:    id = save<char>    (key, arr, filename, append);   break;
            case s32:   id = save<int>     (key, arr, filename, append);   break;
            case u32:   id = save<unsigned>(key, arr, filename, append);   break;
            case u8:    id = save<uchar>   (key, arr, filename, append);   break;
            case s64:   id = save<intl>    (key, arr, filename, append);   break;
            case u64:   id = save<uintl>   (key, arr, filename, append);   break;
            default:    TYPE_ERROR(1, type);
        }
        std::swap(*index, id);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename T>
static af_array readDataToArray(std::fstream &fs)
{
    intl dims[4];
    fs.read((char*)&dims, 4 * sizeof(intl));

    dim4 d;
    for(int i = 0; i < 4; i++) {
        d[i] = dims[i];
    }

    intl size = d.elements();

    std::vector<T> data(size);
    fs.read((char*)&data.front(), size * sizeof(T));

    return getHandle(createHostDataArray<T>(d, &data.front()));
}

static af_array readArrayV1(const char *filename, const unsigned index)
{
    char version = 0;
    int n_arrays = 0;

    std::fstream fs(filename, std::fstream::in | std::fstream::binary);

    // Throw exception if file is not open
    if(!fs.is_open()) AF_ERROR("File failed to open", AF_ERR_ARG);

    if(fs.peek() == std::fstream::traits_type::eof()) {
        AF_ERROR("File is empty", AF_ERR_ARG);
    }

    fs.read(&version, sizeof(char));
    fs.read((char*)&n_arrays, sizeof(int));

    AF_ASSERT((int)index < n_arrays, "Index out of bounds");

    for(int i = 0; i < (int)index; i++) {
        // (int    )   Length of the key
        // (cstring)   Key
        // (intl   )   Offset bytes to next array (type + dims + data)
        // (char   )   Type
        // (intl   )   dim4 (x 4)
        // (T      )   data (x elements)
        int klen = -1;
        fs.read((char*)&klen, sizeof(int));

        //char* key = new char[klen];
        //fs.read((char*)&key, klen * sizeof(char));

        // Skip the array name tag
        fs.seekg(klen, std::ios_base::cur);

        // Read data offset
        intl offset = -1;
        fs.read((char*)&offset, sizeof(intl));

        // Skip data
        fs.seekg(offset, std::ios_base::cur);
    }

    int klen = -1;
    fs.read((char*)&klen, sizeof(int));

    //char* key = new char[klen];
    //fs.read((char*)&key, klen * sizeof(char));

    // Skip the array name tag
    fs.seekg(klen, std::ios_base::cur);

    // Read data offset
    intl offset = -1;
    fs.read((char*)&offset, sizeof(intl));

    // Read type and dims
    char type_ = -1;
    fs.read(&type_, sizeof(char));

    af_dtype type = (af_dtype)type_;

    af_array out;
    switch(type) {
        case f32 : out = readDataToArray<float>  (fs);  break;
        case c32 : out = readDataToArray<cfloat> (fs);  break;
        case f64 : out = readDataToArray<double> (fs);  break;
        case c64 : out = readDataToArray<cdouble>(fs);  break;
        case b8  : out = readDataToArray<char>   (fs);  break;
        case s32 : out = readDataToArray<int>    (fs);  break;
        case u32 : out = readDataToArray<uint>   (fs);  break;
        case u8  : out = readDataToArray<uchar>  (fs);  break;
        case s64 : out = readDataToArray<intl>   (fs);  break;
        case u64 : out = readDataToArray<uintl>  (fs);  break;
        default:    TYPE_ERROR(1, type);
    }
    fs.close();

    return out;
}

static af_array checkVersionAndRead(const char *filename, const unsigned index)
{
    char version = 0;

    std::fstream fs(filename, std::fstream::in | std::fstream::binary);
    // Throw exception if file is not open
    if(!fs.is_open()) AF_ERROR("File failed to open", AF_ERR_ARG);

    if(fs.peek() == std::fstream::traits_type::eof()) {
        AF_ERROR("File is empty", AF_ERR_ARG);
    } else {
        fs.read(&version, sizeof(char));
    }
    fs.close();

    switch(version) {
        case 1: return readArrayV1(filename, index);
        default: AF_ERROR("Invalid version", AF_ERR_ARG);
    }
}

int checkVersionAndFindIndex(const char *filename, const char *k)
{
    char version = 0;
    std::string key(k);

    std::ifstream fs(filename, std::ifstream::in | std::ifstream::binary);
    // Throw exception if file is not open
    if(!fs.is_open()) AF_ERROR("File failed to open", AF_ERR_ARG);

    if(fs.peek() == std::ifstream::traits_type::eof()) {
        AF_ERROR("File is empty", AF_ERR_ARG);
    } else {
        fs.read(&version, sizeof(char));
    }

    int index = -1;
    if(version == 1) {
        int n_arrays = -1;
        fs.read((char*)&n_arrays, sizeof(int));
        for(int i = 0; i < n_arrays; i++) {
            int klen = -1;
            fs.read((char*)&klen, sizeof(int));
            char *readKey = new char[klen + 1];
            fs.read(readKey, klen);
            readKey[klen] = '\0';

            if(key == readKey) {
                // Ket matches, break
                index = i;
                delete [] readKey;
                break;
            } else {
                // Key doesn't match. Skip the data
                intl offset = -1;
                fs.read((char*)&offset, sizeof(intl));
                fs.seekg(offset, std::ios_base::cur);
                delete [] readKey;
            }
        }
    } else {
        AF_ERROR("Invalid version", AF_ERR_ARG);
    }
    fs.close();

    return index;
}

af_err af_read_array_index(af_array *out, const char *filename, const unsigned index)
{
    try {
        AF_CHECK(af_init());

        ARG_ASSERT(1, filename != NULL);

        af_array output = checkVersionAndRead(filename, index);
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_read_array_key(af_array *out, const char *filename, const char *key)
{
    try {
        AF_CHECK(af_init());
        ARG_ASSERT(1, filename != NULL);
        ARG_ASSERT(2, key != NULL);

        // Find index of key. Then call read by index
        int index = checkVersionAndFindIndex(filename, key);

        if(index == -1)
            AF_ERROR("Key not found", AF_ERR_INVALID_ARRAY);

        af_array output = checkVersionAndRead(filename, index);
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_read_array_key_check(int *index, const char *filename, const char* key)
{
    try {
        ARG_ASSERT(1, filename != NULL);
        ARG_ASSERT(2, key != NULL);

        AF_CHECK(af_init());

        // Find index of key. Then call read by index
        int id = checkVersionAndFindIndex(filename, key);
        std::swap(*index, id);
    }
    CATCHALL;
    return AF_SUCCESS;
}
