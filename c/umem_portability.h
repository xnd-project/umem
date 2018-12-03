#ifndef UMEM_PORTABILITY_H
#define UMEM_PORTABILITY_H


#ifdef _MSC_VER
#define strdup _strdup
#define strerror_r strerror_s
#else

#endif

#endif
