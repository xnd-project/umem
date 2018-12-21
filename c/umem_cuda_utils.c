#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include "umem_cuda_utils.h"

#ifdef HAVE_CUDA_CONTEXT
void umemCuda_copy_to_Host(umemVirtual * const ctx,
                           umemCuda * const src_ctx, uintptr_t src_adr,
                           umemHost * const dest_ctx, uintptr_t dest_adr,
                           size_t nbytes) {
  CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                            nbytes, cudaMemcpyDeviceToHost), umemMemoryError, return,
            "umemCuda_copy_to_Host: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost)",
            dest_adr, src_adr, nbytes);
}

void umemCuda_copy_from_Host(umemVirtual * const ctx,
                             umemCuda * const dest_ctx, uintptr_t dest_adr,
                             umemHost * const src_ctx, uintptr_t src_adr,
                             size_t nbytes) {
  CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                            nbytes, cudaMemcpyHostToDevice),
            umemMemoryError, return,
            "umemCuda_copy_from_Host: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToDevice)",
            dest_adr, src_adr, nbytes);
}

void umemCuda_copy_to_Cuda(umemVirtual * const ctx,
                           umemCuda * const src_ctx, uintptr_t src_adr,
                           umemCuda * const dest_ctx, uintptr_t dest_adr,
                           size_t nbytes) {
  if (src_ctx->device == dest_ctx->device) {
    CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                              nbytes, cudaMemcpyDeviceToDevice),
              umemMemoryError, return,
              "umemCuda_copy_to_Cuda: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice)",
              dest_adr, src_adr, nbytes);
  } else {
    CUDA_CALL(ctx, cudaMemcpyPeer((void*)dest_adr, dest_ctx->device,
                                  (const void*)src_adr, src_ctx->device,
                                  nbytes), umemMemoryError, return,
              "umemCuda_copy_to_Cuda: cudaMemcpyPeer(%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu)",
              dest_adr, dest_ctx->device, src_adr, src_ctx->device, nbytes);	
  }
}
#endif

#ifdef HAVE_RMM_CONTEXT

void umemRMM_copy_to_Host(umemVirtual * const ctx,
                          umemRMM * const src_ctx, uintptr_t src_adr,
                          umemHost * const dest_ctx, uintptr_t dest_adr,
                          size_t nbytes, bool async) {
  if (async)
    /* Host memory must be page-locked. The caller is responsible for
       ensuring this requirement. Note that generally, host memory is
       not page-locked. */
    CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                   nbytes, cudaMemcpyDeviceToHost,
                                   (cudaStream_t)src_ctx->stream), umemMemoryError, return,
              "umemRMM_copy_to_Host: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost, %" PRIxPTR ")",
              dest_adr, src_adr, nbytes, src_ctx->stream);
  else
    CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                              nbytes, cudaMemcpyDeviceToHost), umemMemoryError, return,
              "umemRMM_copy_to_Host: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost)",
              dest_adr, src_adr, nbytes);
}

void umemRMM_copy_from_Host(umemVirtual * const ctx,
                            umemRMM * const dest_ctx, uintptr_t dest_adr,
                            umemHost * const src_ctx, uintptr_t src_adr,
                            size_t nbytes, bool async) {
  if (async)
    /* Host memory must be page-locked. The caller is responsible for
       ensuring this requirement. Note that generally, host memory is
       not page-locked. */
    CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                   nbytes, cudaMemcpyHostToDevice,
                                   (cudaStream_t)dest_ctx->stream), umemMemoryError, return,
              "umemRMM_copy_from_Host: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToDevice, %" PRIxPTR ")",
              dest_adr, src_adr, nbytes, dest_ctx->stream);
  else
    CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                              nbytes, cudaMemcpyHostToDevice),
              umemMemoryError, return,
              "umemRMM_copy_from_Host: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToDevice)",
              dest_adr, src_adr, nbytes);
}

#ifdef HAVE_CUDA_CONTEXT
void umemRMM_copy_to_Cuda(umemVirtual * const ctx,
                          umemRMM * const src_ctx, uintptr_t src_adr,
                          umemCuda * const dest_ctx, uintptr_t dest_adr,
                          size_t nbytes, bool async) {
  if (src_ctx->device == dest_ctx->device) {
    if (async)
      CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                     nbytes, cudaMemcpyDeviceToDevice,
                                     (cudaStream_t)src_ctx->stream), umemMemoryError, return,
                "umemRMM_copy_to_Cuda: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice, %" PRIxPTR ")",
                dest_adr, src_adr, nbytes, src_ctx->stream);
    else
      CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                                nbytes, cudaMemcpyDeviceToDevice), umemMemoryError, return,
                "umemRMM_copy_to_Cuda: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice)",
                dest_adr, src_adr, nbytes);
  } else {
    if (async)
      CUDA_CALL(ctx, cudaMemcpyPeerAsync((void*)dest_adr, dest_ctx->device,
                                         (const void*)src_adr, src_ctx->device,
                                         nbytes,
                                         (cudaStream_t)src_ctx->stream), umemMemoryError, return,
                "umemRMM_copy_to_Cuda: cudaMemcpyPeerAsync (%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu, %" PRIxPTR ")",
                dest_adr, dest_ctx->device, src_adr, src_ctx->device, nbytes, src_ctx->stream);
    else
      CUDA_CALL(ctx, cudaMemcpyPeer((void*)dest_adr, dest_ctx->device,
                                    (const void*)src_adr, src_ctx->device,
                                    nbytes), umemMemoryError, return,
                "umemRMM_copy_to_Cuda: cudaMemcpyPeer (%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu)",
                dest_adr, dest_ctx->device, src_adr, src_ctx->device, nbytes);
  }
}

void umemRMM_copy_from_Cuda(umemVirtual * const ctx,
                            umemRMM * const dest_ctx, uintptr_t dest_adr,
                            umemCuda * const src_ctx, uintptr_t src_adr,
                            size_t nbytes, bool async) {
  if (src_ctx->device == dest_ctx->device) {
    if (async)
      CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                     nbytes, cudaMemcpyDeviceToDevice,
                                     (cudaStream_t)dest_ctx->stream), umemMemoryError, return,
                "umemRMM_copy_from_Cuda: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice, %" PRIxPTR ")",
                dest_adr, src_adr, nbytes, dest_ctx->stream);
    else
      CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                                nbytes, cudaMemcpyDeviceToDevice), umemMemoryError, return,
                "umemRMM_copy_from_Cuda: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice)",
                dest_adr, src_adr, nbytes);
  } else {
    if (async)
      CUDA_CALL(ctx, cudaMemcpyPeerAsync((void*)dest_adr, dest_ctx->device,
                                         (const void*)src_adr, src_ctx->device,
                                         nbytes,
                                         (cudaStream_t)dest_ctx->stream), umemMemoryError, return,
                "umemRMM_copy_from_Cuda: cudaMemcpyPeerAsync (%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu, %" PRIxPTR ")",
                dest_adr, dest_ctx->device, src_adr, src_ctx->device, nbytes, dest_ctx->stream);
    else
      CUDA_CALL(ctx, cudaMemcpyPeer((void*)dest_adr, dest_ctx->device,
                                    (const void*)src_adr, src_ctx->device,
                                    nbytes), umemMemoryError, return,
                "umemRMM_copy_from_Cuda: cudaMemcpyPeer (%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu)",
                dest_adr, dest_ctx->device, src_adr, src_ctx->device, nbytes);
  }
}
#endif

void umemRMM_copy_to_RMM(umemVirtual * const ctx,
                         umemRMM * const src_ctx, uintptr_t src_adr,
                         umemRMM * const dest_ctx, uintptr_t dest_adr,
                         size_t nbytes, bool async) {
  if (src_ctx->device == dest_ctx->device) {
    if (async)
      CUDA_CALL(ctx, cudaMemcpyAsync((void*)dest_adr, (const void*)src_adr,
                                     nbytes, cudaMemcpyDeviceToDevice,
                                     (cudaStream_t)src_ctx->stream), umemMemoryError, return,
                "umemRMM_copy_to_RMM: cudaMemcpyAsync (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost, %" PRIxPTR ")",
                dest_adr, src_adr, nbytes, src_ctx->stream);
    else
      CUDA_CALL(ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                                nbytes, cudaMemcpyDeviceToDevice), umemMemoryError, return,
                "umemRMM_copy_to_RMM: cudaMemcpy (%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost)",
                dest_adr, src_adr, nbytes);
  } else {
    if (async)
      CUDA_CALL(ctx, cudaMemcpyPeerAsync((void*)dest_adr, dest_ctx->device,
                                         (const void*)src_adr, src_ctx->device,
                                         nbytes,
                                         (cudaStream_t)src_ctx->stream), umemMemoryError, return,
                "umemRMM_copy_to_RMM: cudaMemcpyPeerAsync (%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu, %" PRIxPTR ")",
                dest_adr, dest_ctx->device, src_adr, src_ctx->device, nbytes, src_ctx->stream);
    else
      CUDA_CALL(ctx, cudaMemcpyPeer((void*)dest_adr, dest_ctx->device,
                                    (const void*)src_adr, src_ctx->device,
                                    nbytes), umemMemoryError, return,
                "umemRMM_copy_to_RMM: cudaMemcpyPeer (%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu)",
                dest_adr, dest_ctx->device, src_adr, src_ctx->device, nbytes);
  }
}

#endif
