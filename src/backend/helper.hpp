#pragma once
#define CATCHALL                \
    catch(...) {                \
        return AF_ERR_INTERNAL; \
    }
