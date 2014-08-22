#pragma once

unsigned size_of(af_dtype type);

#define CATCHALL                \
    catch(...) {                \
        return AF_ERR_INTERNAL; \
    }
