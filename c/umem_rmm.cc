
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

extern "C" {

  static bool umemRMM_is_same_context_(umemVirtual * const one_ctx, umemVirtual * const other_ctx);
  static uintptr_t umemRMM_alloc_(umemVirtual * const ctx, size_t nbytes);

  
}
static bool umemRMM_is_same_context_(umemVirtual * const one_ctx, umemVirtual * const other_ctx) {
  umemRMM * const one_ctx_ = (umemRMM * const)one_ctx;
  umemRMM * const other_ctx_ = (umemRMM * const)other_ctx;
  return (one_ctx_->stream == other_ctx_->stream ? true : false);
}


static uintptr_t umemRMM_alloc_(umemVirtual * const ctx, size_t nbytes) {
  assert(ctx->type == umemRMMDevice);
  umemRMM * const ctx_ = (umemRMM * const)ctx;
  uintptr_t adr;
  RMM_CALL(ctx, RMM_ALLOC((void**)&adr, nbytes, (cudaStream_t)ctx_->stream), umemMemoryError, return 0,
           "umemRMM_alloc_: RMM_ALLOC(&%xu, %zu)",
           adr, nbytes);
  return adr;
}

static void umemRMM_free_(umemVirtual * const ctx, uintptr_t adr) {
  assert(ctx->type == umemRMMDevice);
  umemRMM * const ctx_ = (umemRMM * const)ctx;
  RMM_CALL(ctx, RMM_FREE((void*)adr,  (cudaStream_t)ctx_->stream), umemMemoryError, return,
           "umemRMM_free_: RMM_FREE(%xu)", adr); 
}

static void umemRMM_set_(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes) {
  //assert(ctx->type == umemRMMDevice);
  CUDA_CALL(ctx, cudaMemset((void*)adr, c, nbytes), umemMemoryError,return,
	    "umemRMM_set_: cudaMemset(&%xu, %d, %zu)", adr, c, nbytes);
}

static void umemRMM_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
			      umemVirtual * const dest_ctx, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(src_ctx->type == umemRMMDevice);
  umemRMM * const src_ctx_ = (umemRMM * const)src_ctx;
  switch(dest_ctx->type) {
  case umemHostDevice:
    {
      //umemHost * const dest_ctx_ = (umemHost * const)dest_ctx;
      CUDA_CALL(src_ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
                                    nbytes, cudaMemcpyDeviceToHost), umemMemoryError, return,
		"umemRMM_copy_to_: cudaMemcpy(%xu, %xu, %zu, cudaMemcpyDeviceToHost)",
		dest_adr, src_adr, nbytes);
    }
    break;
  default:
    umem_copy_to_via_host(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  }
}

static void umemRMM_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
				umemVirtual * const src_ctx, uintptr_t src_adr,
				size_t nbytes) {
  assert(dest_ctx->type == umemRMMDevice);
  switch(src_ctx->type) {
  case umemHostDevice:
    {
      //umemHost * const src_ctx_ = (umemHost * const)src_ctx;
      CUDA_CALL(dest_ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
			       nbytes, cudaMemcpyHostToDevice),
		umemMemoryError, return,
		"umemRMM_copy_from_: cudaMemcpy(%xu, %xu, %zu, cudaMemcpyHostToDevice)",
		dest_adr, src_adr, nbytes);
    }
    break;
  default:
    umem_copy_from_via_host(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
  }
}

void umemRMM_ctor(umemRMM * const ctx, uintptr_t stream) {
  static struct umemVtbl const vtbl = {
    &umemVirtual_dtor,
    &umemRMM_is_same_context_,
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
  ctx->stream = stream;
}
