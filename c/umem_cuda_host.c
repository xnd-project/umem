#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <string.h>
#include "umem.h"
#include "umem_cuda_utils.h"
#include "umem_host_utils.h"

/*
  Implementations of umemCuda methods.
*/
static void umemCudaHost_dtor_(umemVirtual * const ctx) {
  umemVirtual_dtor(ctx);
}

static uintptr_t umemCudaHost_alloc_(umemVirtual * const ctx, size_t nbytes) {
  assert(ctx->type == umemCudaHostDevice);
  umemCudaHost * const ctx_ = (umemCudaHost * const)ctx;
  uintptr_t adr;
  CUDA_CALL(ctx, cudaHostAlloc((void**)&adr, nbytes, ctx_->flags), umemMemoryError, return 0,
	    "umemCudaHost_alloc_: cudaHostAlloc(&%" PRIxPTR ", %zu, %ud)",
	    adr, nbytes, ctx_->flags);
  return adr;
}

static void umemCudaHost_free_(umemVirtual * const ctx, uintptr_t adr) {
  assert(ctx->type == umemCudaHostDevice);
  //umemCudaHost * const ctx_ = (umemCudaHost * const)ctx;
  CUDA_CALL(ctx, cudaFreeHost((void*)adr), umemMemoryError, return,
	    "umemCudaHost_free_: cudaFreeHost(%" PRIxPTR ")", adr); 
}

static void umemCudaHost_set_(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes) {
  assert(ctx->type == umemCudaHostDevice);
  HOST_CALL(ctx, memset((void*)adr, c, nbytes)==NULL, umemMemoryError, return,
	    "umemCudaHost_set_: memset(&%" PRIxPTR ", %d, %zu)->NULL", adr, c, nbytes);
}


static void umemCudaHost_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
                                  umemVirtual * const dest_ctx, uintptr_t dest_adr,
                                  size_t nbytes) {
  assert(src_ctx->type == umemCudaHostDevice);
  switch(dest_ctx->type) {
  case umemHostDevice:
  case umemCudaHostDevice:
#ifdef HAVE_CUDA_MANAGED_CONTEXT
  case umemCudaManagedDevice:
#endif
    umemCudaHostCopyToHost(src_ctx, src_adr, dest_adr, nbytes, false, 0);
    break;
#ifdef HAVE_CUDA_CONTEXT
  case umemCudaDevice:
#endif
#ifdef HAVE_RMM_CONTEXT
  case umemRMMDevice:
#endif
    umemCudaCopyFromHost(src_ctx, dest_adr, src_adr, nbytes, false, 0);
    break;
  default:
    umem_copy_to_via_host(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  }
}


static void umemCudaHost_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
                                    umemVirtual * const src_ctx, uintptr_t src_adr,
                                    size_t nbytes) {
  assert(dest_ctx->type == umemCudaHostDevice);
  switch(src_ctx->type) {
  case umemHostDevice:
  case umemCudaHostDevice:
#ifdef HAVE_CUDA_MANAGED_CONTEXT
  case umemCudaManagedDevice:
#endif
    umemCudaHostCopyFromHost(dest_ctx, dest_adr, src_adr, nbytes, false, 0);
    break;
#ifdef HAVE_RMM_CONTEXT
  case umemRMMDevice:
#endif
#ifdef HAVE_CUDA_CONTEXT
  case umemCudaDevice:
#endif
    umemCudaCopyToHost(dest_ctx, src_adr, dest_adr, nbytes, false, 0);
    break;
  default:
    umem_copy_from_via_host(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
  }
}

bool umemCudaHost_is_accessible_from_(umemVirtual * const src_ctx, umemVirtual * const dest_ctx) {
  assert(src_ctx->type == umemCudaHostDevice);
  switch(dest_ctx->type) {
  case umemHostDevice:
  case umemCudaHostDevice:
#ifdef HAVE_CUDA_MANAGED_CONTEXT
  case umemCudaManagedDevice:
#endif
#ifdef HAVE_CUDA_CONTEXT
  case umemCudaDevice:
#endif
#ifdef HAVE_RMM_CONTEXT
  case umemRMMDevice:
#endif
    return true;
  default:
    ;
  }
  return false;
}

/*
  umemCudaHost constructor.
*/
void umemCudaHost_ctor(umemCudaHost * const ctx, unsigned int flags) {
  static struct umemVtbl const vtbl = {
    &umemCudaHost_dtor_,
    &umemCudaHost_is_accessible_from_,
    &umemCudaHost_alloc_,
    &umemVirtual_calloc,
    &umemCudaHost_free_,
    &umemVirtual_aligned_alloc,
    &umemVirtual_aligned_origin,
    &umemVirtual_aligned_free,
    &umemCudaHost_set_,
    &umemCudaHost_copy_to_,
    &umemCudaHost_copy_from_,
  };
  assert(sizeof(CUdeviceptr) == sizeof(uintptr_t));
  umemHost_ctor(&ctx->host);
  umemVirtual_ctor(&ctx->super, &ctx->host);
  ctx->super.vptr = &vtbl;
  ctx->super.type = umemCudaHostDevice;
  ctx->flags = flags;
}

