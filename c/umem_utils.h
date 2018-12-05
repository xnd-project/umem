#ifndef UMEM_UTILS_H
#define UMEM_UTILS_H

#include <inttypes.h>
#include <stddef.h>

/*
  Portability macros
 */

#define UMEM_FUNDAMENTAL_FILE_ALIGN 1
#define UMEM_FUNDAMENTAL_CUDA_ALIGN 256 // 512 on newer cards?

#if defined(_WIN32) || defined(__CYGWIN__)

#define strdup _strdup
#define strerror_r strerror_s
#define UMEM_EXPORT __declspec(dllexport)
#define UMEM_FUNDAMENTAL_HOST_ALIGN 8 // 16 on 64-bit Windows?

#else

#define UMEM_EXPORT
#include <stdalign.h>

#define UMEM_FUNDAMENTAL_HOST_ALIGN (alignof(max_align_t))

#endif

#define UMEM_EXTERN extern UMEM_EXPORT



/*
  Utilities
*/

UMEM_EXTERN void hexDump(char *desc, void *addr, int len);
UMEM_EXTERN void binDump(char *desc, void *addr, int len);

#define TRY_RETURN(ME, CALL, ERROR, ERRRETURN, FMT, ...)        \
  do {								\
    if (CALL) {                                                 \
      char buf[256];                                            \
      snprintf(buf, sizeof(buf), FMT, __VA_ARGS__);             \
      umem_set_status(ME, ERROR, buf);				\
      ERRRETURN;						\
    }								\
  } while (0)

static inline int umem_ispowerof2(size_t x) {
  return x && !(x & (x - 1));
}

#endif
