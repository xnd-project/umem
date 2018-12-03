#ifndef UMEM_PORTABILITY_H
#define UMEM_PORTABILITY_H

#include <inttypes.h>

#if defined(_WIN32) || defined(__CYGWIN__)

#define strdup _strdup
#define strerror_r strerror_s
#define UMEM_EXPORT __declspec(dllexport)
#else

#define UMEM_EXPORT

#endif

#define UMEM_EXTERN extern UMEM_EXPORT

#endif
