#ifndef UMEM_CUDA_UTILS_H
#define UMEM_CUDA_UTILS_H

#include "umem.h"

#define CUDA_CALL(CTX, CALL, ERROR, ERRRETURN, FMT, ...)                \
  do {									\
    int old_errno = errno;						\
    cudaError_t error = CALL;						\
    if (error != cudaSuccess) {						\
      char buf[256];							\
      snprintf(buf, sizeof(buf), FMT " -> %s: %s", __VA_ARGS__,		\
	       cudaGetErrorName(error), cudaGetErrorString(error));	\
      umem_set_status(CTX, ERROR, buf);					\
      ERRRETURN;							\
    } else errno = old_errno;						\
  } while (0)

#define CU_CALL(CTX, CALL, ERROR, ERRRETURN, FMT, ...)                  \
  do {									\
    int old_errno = errno;						\
    CUresult error = CALL;						\
    if (error != CUDA_SUCCESS) {                                        \
      char buf[256];                                                    \
      const char* errname = NULL;                                             \
      const char* errstr = NULL;                                              \
      cuGetErrorName(error, &errname);                                  \
      cuGetErrorString(error, &errstr);                                 \
      snprintf(buf, sizeof(buf), FMT " -> %s: %s", __VA_ARGS__,		\
	       errname, errstr);                                        \
      umem_set_status(CTX, ERROR, buf);					\
      ERRRETURN;							\
    } else errno = old_errno;						\
  } while (0)

UMEM_START_EXTERN_C

#if defined(HAVE_CUDA_CONTEXT) || defined(HAVE_RMM_CONTEXT)

UMEM_EXPORT bool umemCudaPeerAccessEnabled(umemVirtual * const ctx, int src_device, int dest_device);


UMEM_EXPORT void umemCudaSet(umemVirtual * const ctx,
                             uintptr_t adr, int c, size_t nbytes,
                             bool async, uintptr_t stream);

UMEM_EXPORT void umemCudaHostCopyToHost(umemVirtual * const ctx,
                                        uintptr_t src_adr, uintptr_t dest_adr,
                                        size_t nbytes,
                                        bool async, uintptr_t stream);

UMEM_EXPORT void umemCudaHostCopyFromHost(umemVirtual * const ctx,
                                          uintptr_t dest_adr, uintptr_t src_adr,
                                          size_t nbytes,
                                          bool async, uintptr_t stream);

UMEM_EXPORT void umemCudaCopyToHost(umemVirtual * const ctx,
                                    uintptr_t src_adr, uintptr_t dest_adr,
                                    size_t nbytes,
                                    bool async, uintptr_t stream);

UMEM_EXPORT void umemCudaCopyFromHost(umemVirtual * const ctx,
                                      uintptr_t dest_adr, uintptr_t src_adr,
                                      size_t nbytes,
                                      bool async, uintptr_t stream);

UMEM_EXPORT void umemCudaCopyToCuda(umemVirtual * const ctx,
                        int src_device, uintptr_t src_adr,
                        int dest_device, uintptr_t dest_adr,
                        size_t nbytes,
                        bool async, uintptr_t stream);
#endif


UMEM_CLOSE_EXTERN_C

#endif
