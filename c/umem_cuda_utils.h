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



UMEM_START_EXTERN_C

#ifdef HAVE_CUDA_CONTEXT
UMEM_EXPORT void umemCuda_copy_to_Host(umemVirtual * const ctx,
                                       umemCuda * const src_ctx, uintptr_t src_adr,
                                       umemHost * const dest_ctx, uintptr_t dest_adr,
                                       size_t nbytes);
UMEM_EXPORT void umemCuda_copy_from_Host(umemVirtual * const ctx,
                                         umemCuda * const dest_ctx, uintptr_t dest_adr,
                                         umemHost * const src_ctx, uintptr_t src_adr,
                                         size_t nbytes);
UMEM_EXPORT void umemCuda_copy_to_Cuda(umemVirtual * const ctx,
                                       umemCuda * const src_ctx, uintptr_t src_adr,
                                       umemCuda * const dest_ctx, uintptr_t dest_adr,
                                       size_t nbytes);
#endif // HAVE_CUDA_CONTEXT

#ifdef HAVE_RMM_CONTEXT
UMEM_EXPORT void umemRMM_copy_to_RMM(umemVirtual * const ctx,
                                     umemRMM * const src_ctx_, uintptr_t src_adr,
                                     umemRMM * const dest_ctx_, uintptr_t dest_adr,
                                     size_t nbytes, bool async);
UMEM_EXPORT void umemRMM_copy_to_Host(umemVirtual * const ctx,
                                      umemRMM * const src_ctx_, uintptr_t src_adr,
                                      umemHost * const dest_ctx_, uintptr_t dest_adr,
                                      size_t nbytes, bool async);
UMEM_EXPORT void umemRMM_copy_from_Host(umemVirtual * const ctx,
                                        umemRMM * const dest_ctx, uintptr_t dest_adr,
                                        umemHost * const src_ctx, uintptr_t src_adr,
                                        size_t nbytes, bool async);
#ifdef HAVE_CUDA_CONTEXT
UMEM_EXPORT void umemRMM_copy_to_Cuda(umemVirtual * const ctx,
                                      umemRMM * const src_ctx_, uintptr_t src_adr,
                                      umemCuda * const dest_ctx_, uintptr_t dest_adr,
                                      size_t nbytes, bool async);
UMEM_EXPORT void umemRMM_copy_from_Cuda(umemVirtual * const ctx,
                                        umemRMM * const dest_ctx_, uintptr_t dest_adr,
                                        umemCuda * const src_ctx_, uintptr_t src_adr,
                                        size_t nbytes, bool async);
#endif // HAVE_CUDA_CONTEXT
#endif // HAVE_RMM_CONTEXT

UMEM_CLOSE_EXTERN_C

#endif
