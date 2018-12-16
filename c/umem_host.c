#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "umem.h"

#define HOST_CALL(CTX, CALL, ERROR, ERRRETURN, FMT, ...)       \
  do {								\
    if (CALL) {				\
      char buf[256];						\
      snprintf(buf, sizeof(buf), FMT, __VA_ARGS__);             \
      umem_set_status(CTX, ERROR, buf);                        \
      ERRRETURN;						\
    }								\
  } while (0)

/*
  Implementations of umemHostMemory methods.
*/

static void umemHost_dtor_(umemVirtual * const ctx) {
  ctx->host_ctx = NULL;
  umemVirtual_dtor(ctx);
}

static uintptr_t umemHost_alloc_(umemVirtual * const ctx, size_t nbytes) {
  assert(ctx->type == umemHostDevice);
  uintptr_t adr = 0;
  if (nbytes != 0)
    HOST_CALL(ctx, (adr = (uintptr_t)malloc(nbytes))==0,
	      umemMemoryError, return 0,
	      "umemHost_alloc_: malloc(%zu)->NULL", nbytes
	      );
  return adr;
}

static uintptr_t umemHost_calloc_(umemVirtual * const ctx, size_t nmemb, size_t size) {
  assert(ctx->type == umemHostDevice);
  uintptr_t adr = 0;
  if (size != 0)
    HOST_CALL(ctx, (adr = (uintptr_t)calloc(nmemb, size))==0,
	      umemMemoryError, return 0,
	      "umemHost_alloc_: calloc(%zu, %zu)->NULL", nmemb, size
	      );
  return adr;
}


static void umemHost_free_(umemVirtual * const ctx, uintptr_t adr) {
  assert(ctx->type == umemHostDevice);
  free((void*)adr);
}

static uintptr_t umemHost_aligned_alloc_(umemVirtual * const ctx, size_t alignment, size_t size) {
  uintptr_t adr = 0;
  size_t extra = (alignment - 1) + sizeof(uintptr_t);
  size_t req = extra + (size ? size: 1);
  adr = umemHost_calloc_(ctx, req, 1);
  if (!umem_is_ok(ctx))
    return 0;
  uintptr_t aligned = adr + extra;
  aligned = aligned - (aligned % alignment);
  *((uintptr_t *)aligned - 1) = adr;
  return aligned;
}

static uintptr_t umemHost_aligned_origin_(umemVirtual * const ctx, uintptr_t aligned_adr) {
  return (aligned_adr ? *((uintptr_t *)aligned_adr - 1) : 0);
}

static void umemHost_aligned_free_(umemVirtual * const ctx, uintptr_t aligned_adr) {
  umemHost_free_(ctx, umemHost_aligned_origin_(ctx, aligned_adr));
}

static void umemHost_set_(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes) {
  assert(ctx->type == umemHostDevice);
  HOST_CALL(ctx, memset((void*)adr, c, nbytes)==NULL, umemMemoryError, return,
	    "umemHost_set_: memset(&%" PRIxPTR ", %d, %zu)->NULL", adr, c, nbytes);
}

static void umemHost_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
			      umemVirtual * const dest_ctx, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(src_ctx->type == umemHostDevice);
  if (dest_ctx->type == umemHostDevice) {
    HOST_CALL(src_ctx, memcpy((void*)dest_adr, (void*)src_adr, nbytes)==NULL, umemMemoryError, return,
	      "umemHost_copy_to_: memcpy(%02" PRIxPTR ", %" PRIxPTR ", %zu)->NULL",
	      dest_adr, src_adr, nbytes);
  } else
    umem_copy_from(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
}

static void umemHost_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
				umemVirtual * const src_ctx, uintptr_t src_adr,
				size_t nbytes) {
  assert(dest_ctx->type == umemHostDevice);
  if (src_ctx->type == umemHostDevice)
    umemHost_copy_to_(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  else
    umem_copy_to(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
}

bool umemHost_is_same_context_(umemVirtual * const one_ctx, umemVirtual * const other_ctx) {
  return true;
}

/*
  umemHost constructor.
*/

void umemHost_ctor(umemHost * const ctx) {
  static struct umemVtbl const vtbl = {
    &umemHost_dtor_,
    &umemHost_is_same_context_,
    &umemHost_alloc_,
    &umemHost_calloc_,
    &umemHost_free_,
    &umemHost_aligned_alloc_,
    &umemHost_aligned_origin_,
    &umemHost_aligned_free_,
    &umemHost_set_,
    &umemHost_copy_to_,
    &umemHost_copy_from_,
  };
  umemVirtual_ctor(&ctx->super, ctx);
  ctx->super.type = umemHostDevice;
  ctx->super.vptr = &vtbl;
}


/*
  umemVirtual copy methods that use host memory as an intermediate
  buffer.
 */
void umem_copy_to_via_host(void * const src_ctx, uintptr_t src_adr,
			   void * const dest_ctx, uintptr_t dest_adr,
			   size_t nbytes) {
  umemHost host;
  umemHost_ctor(&host);
  if (umem_is_ok((void*)&host)) {
    uintptr_t host_adr = umem_alloc(&host, nbytes);
    if (umem_is_ok((void*)&host)) {
      umem_copy_to(src_ctx, src_adr, &host, host_adr, nbytes);
      if (umem_is_ok(src_ctx))
	umem_copy_from(dest_ctx, dest_adr, &host, host_adr, nbytes);
      umem_free(&host, host_adr);
    }
  }
  umem_dtor(&host);
}

void umem_copy_from_via_host(void * const dest_ctx, uintptr_t dest_adr,
			     void * const src_ctx, uintptr_t src_adr,
			     size_t nbytes) {
  umemHost host;
  umemHost_ctor(&host);
  if (umem_is_ok((void*)&host)) {
    uintptr_t host_adr = umem_alloc(&host, nbytes);
    if (umem_is_ok((void*)&host)) {
      umem_copy_to(src_ctx, src_adr, &host, host_adr, nbytes);
      if (umem_is_ok(src_ctx))
	umem_copy_from(dest_ctx, dest_adr, &host, host_adr, nbytes);
      umem_free(&host, host_adr);
    }
  }
  umem_dtor(&host);
}
