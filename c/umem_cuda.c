#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include "umem.h"
#include "umem_cuda_utils.h"

/*
  Implementations of umemCuda methods.
*/
static void umemCuda_dtor_(umemVirtual * const ctx) {
  umemVirtual_dtor(ctx);
}

static uintptr_t umemCuda_alloc_(umemVirtual * const ctx, size_t nbytes) {
  assert(ctx->type == umemCudaDevice);
  umemCuda * const ctx_ = (umemCuda * const)ctx;
  CUDA_CALL(ctx, cudaSetDevice(ctx_->device), umemRuntimeError, return 0,
	    "umemCuda_alloc_: cudaSetDevice(%d)", ctx_->device);
  uintptr_t adr;
  CUDA_CALL(ctx, cudaMalloc((void**)&adr, nbytes), umemMemoryError, return 0,
	    "umemCuda_alloc_: cudaMalloc(&%" PRIxPTR ", %zu)",
	    adr, nbytes);
  // TODO: do we need to reset to old device?
  return adr;
}

static void umemCuda_free_(umemVirtual * const ctx, uintptr_t adr) {
  assert(ctx->type == umemCudaDevice);
  //umemCuda * const ctx_ = (umemCuda * const)ctx;
  CUDA_CALL(ctx, cudaFree((void*)adr), umemMemoryError, return,
	    "umemCuda_free_: cudaFree(%" PRIxPTR ")", adr); 
}

static void umemCuda_set_(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes) {
  assert(ctx->type == umemCudaDevice);
  // TODO: does set require current device to be ctx_->device??
  umemCudaSet(ctx, adr, c, nbytes, false, 0);
}


static void umemCuda_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
			      umemVirtual * const dest_ctx, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(src_ctx->type == umemCudaDevice);
  switch(dest_ctx->type) {
  case umemHostDevice:
    umemCudaCopyToHost(src_ctx, src_adr, dest_adr, nbytes, false, 0);
    break;
  case umemCudaDevice:
    umemCudaCopyToCuda(src_ctx, ((umemCuda * const)src_ctx)->device, src_adr,
                       ((umemCuda * const)dest_ctx)->device, dest_adr,
                       nbytes, false, 0);
    break;
#ifdef HAVE_RMM_CONTEXT
  case umemRMMDevice:
    umemCudaCopyToCuda(src_ctx, ((umemCuda * const)src_ctx)->device, src_adr,
                       ((umemRMM * const)dest_ctx)->device, dest_adr,
                       nbytes, false, 0);
    break;
#endif
  default:
    umem_copy_to_via_host(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  }
}

static void umemCuda_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
				umemVirtual * const src_ctx, uintptr_t src_adr,
				size_t nbytes) {
  assert(dest_ctx->type == umemCudaDevice);
  switch(src_ctx->type) {
  case umemHostDevice:
    umemCudaCopyFromHost(dest_ctx, dest_adr, src_adr, nbytes, false, 0);


    break;
  case umemCudaDevice:
    umemCudaCopyToCuda(dest_ctx, ((umemCuda * const)src_ctx)->device, src_adr,
                       ((umemCuda * const)dest_ctx)->device, dest_adr,
                       nbytes, false, 0);
    break;
#ifdef HAVE_RMM_CONTEXT
  case umemRMMDevice:
    umemCudaCopyToCuda(dest_ctx, ((umemRMM * const)src_ctx)->device, src_adr,
                       ((umemCuda * const)dest_ctx)->device, dest_adr,
                       nbytes, false, 0);
    break;
#endif
  default:
    umem_copy_from_via_host(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
  }
}

bool umemCuda_is_accessible_from_(umemVirtual * const src_ctx, umemVirtual * const dest_ctx) {
  assert(src_ctx->type == umemCudaDevice);
  switch(dest_ctx->type) {
  case umemHostDevice:
    return false;
  case umemCudaDevice: {
    umemCuda * const src_ctx_ = (umemCuda * const)src_ctx;
    umemCuda * const dest_ctx_ = (umemCuda * const)dest_ctx;
    if (src_ctx_->device == dest_ctx_->device)
      return true;
    int accessible = 0;
    CUDA_CALL(src_ctx, cudaDeviceCanAccessPeer( &accessible, src_ctx_->device, dest_ctx_->device ),
              umemRuntimeError, return false,
              "umemCuda_are_accessible_: cudaDeviceCanAccessPeer( &accessible, %d, %d )",
              src_ctx_->device, dest_ctx_->device);
    return umemCudaPeerAccessEnabled(src_ctx, src_ctx_->device, dest_ctx_->device);
  }
#ifdef HAVE_CUDA_MANAGED_CONTEXT
  case umemCudaManagedDevice:
    return true;
#endif
  default:
    ;
  }

  return false;
}

/*
  umemCuda constructor.
*/
void umemCuda_ctor(umemCuda * const ctx, int device) {
  static struct umemVtbl const vtbl = {
    &umemCuda_dtor_,
    &umemCuda_is_accessible_from_,
    &umemCuda_alloc_,
    &umemVirtual_calloc,
    &umemCuda_free_,
    &umemVirtual_aligned_alloc,
    &umemVirtual_aligned_origin,
    &umemVirtual_aligned_free,
    &umemCuda_set_,
    &umemCuda_copy_to_,
    &umemCuda_copy_from_,
  };
  assert(sizeof(CUdeviceptr) == sizeof(uintptr_t));
  umemHost_ctor(&ctx->host);
  umemVirtual_ctor(&ctx->super, &ctx->host);
  ctx->super.vptr = &vtbl;
  ctx->super.type = umemCudaDevice;
  ctx->device = device;

  int count;
  CUDA_CALL(&ctx->super, cudaGetDeviceCount(&count),
	    umemRuntimeError, return,
	    "umemCuda_ctor: cudaGetDeviceCount(&%d)",
	    count);
  if (!(device >=0 && device < count)) {
    char buf[256];
    snprintf(buf, sizeof(buf),
	     "umemCuda_ctor: invalid device number: %d. Must be less than %d",
	     device, count);
    umem_set_status(&ctx->super, umemValueError, buf);
    return;
  }
  // might be superfluous as alloc sets the device
  CUDA_CALL(&ctx->super, cudaSetDevice(device), umemRuntimeError, return,
	    "umemCuda_ctor: cudaSetDevice(%d)", device);
}

