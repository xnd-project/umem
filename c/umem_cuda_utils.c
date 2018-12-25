#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include "umem_cuda_utils.h"

#if defined(HAVE_CUDA_CONTEXT) || defined(HAVE_RMM_CONTEXT)

inline static bool umemCudaIsPageLocked(uintptr_t adr) {
  struct cudaPointerAttributes my_attr;
  if (cudaPointerGetAttributes(&my_attr, (void*)adr) == cudaErrorInvalidValue) {
    cudaGetLastError(); // clear out the previous API error
    return false;
    }
  return true;
}

inline static bool umemCudaEnsurePageLocked(umemVirtual * const ctx, uintptr_t adr, size_t nbytes) {
  if (umemCudaIsPageLocked(adr))
    return false;
  else
    CUDA_CALL(ctx, cudaHostRegister((void*)adr, nbytes, cudaHostRegisterPortable), umemMemoryError, return false,
              "umemCudaEnsurePageLocked: cudaHostRegister(%" PRIxPTR ", %zu, cudaHostRegisterPortable)",
              adr, nbytes
              );
  return true;
}

inline static void umemCudaReleasePageLock(umemVirtual * const ctx, uintptr_t adr, bool lock) {
  if (lock)
    CUDA_CALL(ctx, cudaHostUnregister((void*)adr), umemMemoryError, return,
              "umemCudaReleasePageLock: cudaHostUnregister(%" PRIxPTR ")", adr
              );
}

void umemCudaHostCopyToHost(umemVirtual * const ctx,
                            uintptr_t src_adr, uintptr_t dest_adr,
                            size_t nbytes,
                            bool async, uintptr_t stream) {
  if (async) {
    bool lock = umemCudaEnsurePageLocked(ctx, dest_adr, nbytes);
    CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                   nbytes, cudaMemcpyHostToHost,
                                   (cudaStream_t)stream), umemMemoryError, goto cleanup_lock,
              "umemCudaHostCopyToHost: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToHost, %" PRIxPTR ")",
              dest_adr, src_adr, nbytes, stream);
  cleanup_lock:
    umemCudaReleasePageLock(ctx, dest_adr, lock);
  } else
    CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                              nbytes, cudaMemcpyHostToHost), umemMemoryError, return,
              "umemCudaHostCopyToHost: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToHost)",
              dest_adr, src_adr, nbytes);
}

void umemCudaHostCopyFromHost(umemVirtual * const ctx,
                              uintptr_t dest_adr, uintptr_t src_adr,
                              size_t nbytes,
                              bool async, uintptr_t stream) {
  if (async) {
    bool lock = umemCudaEnsurePageLocked(ctx, src_adr, nbytes);
    CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                   nbytes, cudaMemcpyHostToHost,
                                   (cudaStream_t)stream), umemMemoryError, goto cleanup_lock,
              "umemCudaHostCopyFromHost: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToHost, %" PRIxPTR ")",
              dest_adr, src_adr, nbytes, stream);
  cleanup_lock:
    umemCudaReleasePageLock(ctx, src_adr, lock);
  } else
    CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                              nbytes, cudaMemcpyHostToHost), umemMemoryError, return,
              "umemCudaHostCopyFromHost: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToHost)",
              dest_adr, src_adr, nbytes);
}

void umemCudaSet(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes, bool async, uintptr_t stream) {
  if (async)
    CUDA_CALL(ctx, cudaMemsetAsync((void*)adr, c, nbytes, (cudaStream_t)stream), umemMemoryError,return,
              "umemCudaSet: cudaMemsetAsync(&%lxu, %d, %zu, %zu)", adr, c, nbytes, stream);
  else
    CUDA_CALL(ctx, cudaMemset((void*)adr, c, nbytes), umemMemoryError,return,
              "umemCudaSet: cudaMemset(&%lxu, %d, %zu)", adr, c, nbytes);
}

void umemCudaCopyToHost(umemVirtual * const ctx,
                        uintptr_t src_adr, uintptr_t dest_adr,
                        size_t nbytes,
                        bool async, uintptr_t stream) {
  if (async) {
    bool lock = umemCudaEnsurePageLocked(ctx, dest_adr, nbytes);
    CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                   nbytes, cudaMemcpyDeviceToHost,
                                   (cudaStream_t)stream), umemMemoryError, goto cleanup_lock,
              "umemCudaCopyToHost: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost, %" PRIxPTR ")",
              dest_adr, src_adr, nbytes, stream);
  cleanup_lock:
    umemCudaReleasePageLock(ctx, dest_adr, lock);
  } else
    CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                              nbytes, cudaMemcpyDeviceToHost), umemMemoryError, return,
              "umemCudaCopyToHost: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost)",
              dest_adr, src_adr, nbytes);
}

void umemCudaCopyFromHost(umemVirtual * const ctx,
                          uintptr_t dest_adr, uintptr_t src_adr,
                          size_t nbytes,
                          bool async, uintptr_t stream) {
  if (async) {
    bool lock = umemCudaEnsurePageLocked(ctx, src_adr, nbytes);
    CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                   nbytes, cudaMemcpyHostToDevice,
                                   (cudaStream_t)stream), umemMemoryError, goto cleanup_lock,
              "umemCudaCopyFromHost: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToDevice, %" PRIxPTR ")",
              dest_adr, src_adr, nbytes, stream);
  cleanup_lock:
    umemCudaReleasePageLock(ctx, src_adr, lock);
  } else
    CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                            nbytes, cudaMemcpyHostToDevice),
            umemMemoryError, return,
            "umemCudaCopyFromHost: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToDevice)",
            dest_adr, src_adr, nbytes);
  
}

void umemCudaCopyToCuda(umemVirtual * const ctx,
                        int src_device, uintptr_t src_adr,
                        int dest_device, uintptr_t dest_adr,
                        size_t nbytes,
                        bool async, uintptr_t stream) {
  if (async) {
    if (src_device == dest_device) {
      CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                     nbytes, cudaMemcpyDeviceToDevice,
                                     (cudaStream_t)stream), umemMemoryError, return,
                "umemCudaCopyToCuda: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice, %" PRIxPTR ")",
                dest_adr, src_adr, nbytes, stream);
    } else {
      CUDA_CALL(ctx, cudaMemcpyPeerAsync((void*)dest_adr, dest_device,
                                         (const void*)src_adr, src_device,
                                         nbytes,
                                         (cudaStream_t)stream), umemMemoryError, return,
                "umemCudaCopyToCuda: cudaMemcpyPeerAsync (%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu, %" PRIxPTR ")",
                dest_adr, dest_device, src_adr, src_device, nbytes, stream);
    }
  } else {
    if (src_device == dest_device) {
      CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                                nbytes, cudaMemcpyDeviceToDevice),
                umemMemoryError, return,
                "umemCudaCopyToCuda: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice)",
                dest_adr, src_adr, nbytes);
    } else {
      CUDA_CALL(ctx, cudaMemcpyPeer((void*)dest_adr, dest_device,
                                    (const void*)src_adr, src_device,
                                    nbytes), umemMemoryError, return,
                "umemCudaCopyToCuda: cudaMemcpyPeer(%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu)",
                dest_adr, dest_device, src_adr, src_device, nbytes);	
    }
  }
}

#endif
