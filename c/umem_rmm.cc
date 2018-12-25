
#include <cstdint>
#include "rmm.h"
#include "umem.h"
#include "umem_cuda_utils.h"

#define RMM_CALL(CTX, CALL, ERROR, ERRRETURN, FMT, ...)                 \
  do {									\
    int old_errno = errno;						\
    rmmError_t error = (CALL);						\
    if (error != RMM_SUCCESS) {						\
      char buf[256];							\
      snprintf(buf, sizeof(buf), FMT " -> %s", __VA_ARGS__,		\
	       rmmGetErrorString(error));                               \
      umem_set_status(CTX, ERROR, buf);					\
      ERRRETURN;							\
    } else errno = old_errno;						\
  } while (0)


static bool umemRMM_is_accessible_from_(umemVirtual * const src_ctx, umemVirtual * const dest_ctx) {
  assert(src_ctx->type == umemRMMDevice);
  switch (dest_ctx->type) {
  case umemRMMDevice:
    {
      umemRMM * const src_ctx_ = (umemRMM * const)src_ctx;
      umemRMM * const dest_ctx_ = (umemRMM * const)dest_ctx;
      return (src_ctx_->device == dest_ctx_->device && src_ctx_->stream == dest_ctx_->stream); // TODO: is stream equality required??
    }
      // TODO: RMM addresses are most probably accessible from certain Cuda memory contexts as well
  default:
    break;
  }
  return false;
}

static uintptr_t umemRMM_alloc_(umemVirtual * const ctx, size_t nbytes) {
  assert(ctx->type == umemRMMDevice);
  umemRMM * const ctx_ = (umemRMM * const)ctx;
  uintptr_t adr;
  RMM_CALL(ctx, RMM_ALLOC((void**)&adr, nbytes, (cudaStream_t)ctx_->stream), umemMemoryError, return 0,
           "umemRMM_alloc_: RMM_ALLOC(&%lxu, %zu)",
           adr, nbytes);
  return adr;
}

static void umemRMM_free_(umemVirtual * const ctx, uintptr_t adr) {
  assert(ctx->type == umemRMMDevice);
  umemRMM * const ctx_ = (umemRMM * const)ctx;
  RMM_CALL(ctx, RMM_FREE((void*)adr,  (cudaStream_t)ctx_->stream), umemMemoryError, return,
           "umemRMM_free_: RMM_FREE(%lxu)", adr); 
}

static void umemRMM_set_(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes) {
  assert(ctx->type == umemRMMDevice);
  umemRMM * const ctx_ = (umemRMM * const)ctx;
  umemCudaSet(ctx, adr, c, nbytes, ctx_->async, ctx_->stream);
}


static void umemRMM_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
                             umemVirtual * const dest_ctx, uintptr_t dest_adr,
                             size_t nbytes) {
  assert(src_ctx->type == umemRMMDevice);
  umemRMM * const src_ctx_ = (umemRMM * const)src_ctx;
  switch(dest_ctx->type) {
  case umemHostDevice:
    umemCudaCopyToHost(src_ctx, src_adr, dest_adr, nbytes, src_ctx_->async, src_ctx_->stream);
    break;
#ifdef HAVE_CUDA_CONTEXT
  case umemCudaDevice:
    umemCudaCopyToCuda(src_ctx,
                       src_ctx_->device, src_adr,
                       ((umemCuda * const)dest_ctx)->device, dest_adr,
                       nbytes, src_ctx_->async, src_ctx_->stream);
    break;
#endif
  case umemRMMDevice:
    umemCudaCopyToCuda(src_ctx,
                       src_ctx_->device, src_adr,
                       ((umemRMM * const)dest_ctx)->device, dest_adr,
                       nbytes, src_ctx_->async, src_ctx_->stream);
    break;
  default:
    umem_copy_to_via_host(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  }
}

static void umemRMM_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
                               umemVirtual * const src_ctx, uintptr_t src_adr,
                               size_t nbytes) {
  assert(dest_ctx->type == umemRMMDevice);
  umemRMM * const dest_ctx_ = (umemRMM * const)dest_ctx;
  switch(src_ctx->type) {
  case umemHostDevice:
    umemCudaCopyFromHost(dest_ctx,
                         dest_adr, src_adr, nbytes,
                         dest_ctx_->async, dest_ctx_->stream);
    break;
#ifdef HAVE_CUDA_CONTEXT
  case umemCudaDevice:
    umemCudaCopyToCuda(dest_ctx,
                       ((umemCuda * const)src_ctx)->device, src_adr,
                       ((umemRMM * const)dest_ctx)->device, dest_adr,
                       nbytes, dest_ctx_->async, dest_ctx_->stream);
    break;
#endif
  case umemRMMDevice:
    umemCudaCopyToCuda(dest_ctx,
                       ((umemRMM * const)src_ctx)->device, src_adr,
                       ((umemRMM * const)dest_ctx)->device, dest_adr,
                       nbytes, dest_ctx_->async, dest_ctx_->stream);
    break;
  default:
    umem_copy_from_via_host(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
  }
}

void umemRMM_ctor(umemRMM * const ctx, int device, bool async, uintptr_t stream) {
  static struct umemVtbl const vtbl = {
    &umemVirtual_dtor,
    &umemRMM_is_accessible_from_,
    &umemRMM_alloc_,
    &umemVirtual_calloc,
    &umemRMM_free_,
    &umemVirtual_aligned_alloc,
    &umemVirtual_aligned_origin,
    &umemVirtual_aligned_free,
    &umemRMM_set_,
    &umemRMM_copy_to_,
    &umemRMM_copy_from_,
  };
  //assert(sizeof(CUdeviceptr) == sizeof(uintptr_t));
  umemHost_ctor(&ctx->host);
  umemVirtual_ctor(&ctx->super, &ctx->host);
  ctx->super.vptr = &vtbl;
  ctx->super.type = umemRMMDevice;
  ctx->device = device;
  ctx->async = async;
  ctx->stream = stream;
}
