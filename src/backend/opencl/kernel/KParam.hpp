#ifndef __KPARAM_H
#define __KPARAM_H
typedef struct
{
    dim_type dims[4];
    dim_type strides[4];
    dim_type offset;
} KParam;
#endif
