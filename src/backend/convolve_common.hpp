#pragma once

typedef enum {
    ONE2ONE = 1, /* one signal, one filter   */
    MANY2ONE,    /* many signal, one filter  */
    MANY2MANY,   /* many signal, many filter */
    ONE2MANY,    /* one signal, many filter  */
} ConvolveBatchKind;
