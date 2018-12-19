
#include <cstdint>

#include "umem.h"
#include "rmm.h"


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
  assert(ctx->type == umemRMMDevice);
}

static void umemRMM_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
			      umemVirtual * const dest_ctx, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(src_ctx->type == umemRMMDevice);
  umemRMM * const src_ctx_ = (umemRMM * const)src_ctx;
}

static void umemRMM_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
				umemVirtual * const src_ctx, uintptr_t src_adr,
				size_t nbytes) {
  assert(dest_ctx->type == umemRMMDevice);
  umemRMM * const dest_ctx_ = (umemRMM * const)dest_ctx;
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
