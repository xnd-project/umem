#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "umem.h"

#define HOST_CALL(THIS, CALL, ERROR, ERRRETURN, FMT, ...)       \
  do {								\
    if (CALL) {				\
      char buf[256];						\
      snprintf(buf, sizeof(buf), FMT, __VA_ARGS__);             \
      umem_set_status(THIS, ERROR, buf);                        \
      ERRRETURN;						\
    }								\
  } while (0)

/*
  Implementations of umemHostMemory methods.
*/

static void umemHost_dtor_(umemVirtual * const this) {
  this->host = NULL;
  umemVirtual_dtor(this);
}

static uintptr_t umemHost_alloc_(umemVirtual * const this, size_t nbytes) {
  assert(this->type == umemHostDevice);
  uintptr_t adr = 0;
  if (nbytes != 0)
    HOST_CALL(this, (adr = (uintptr_t)malloc(nbytes))==0,
	      umemMemoryError, return 0,
	      "umemHost_alloc_: malloc(%zu)->NULL", nbytes
	      );
  return adr;
}

static uintptr_t umemHost_calloc_(umemVirtual * const this, size_t nmemb, size_t size) {
  assert(this->type == umemHostDevice);
  uintptr_t adr = 0;
  if (size != 0)
    HOST_CALL(this, (adr = (uintptr_t)calloc(nmemb, size))==0,
	      umemMemoryError, return 0,
	      "umemHost_alloc_: calloc(%zu, %zu)->NULL", nmemb, size
	      );
  return adr;
}


static void umemHost_free_(umemVirtual * const this, uintptr_t adr) {
  assert(this->type == umemHostDevice);
  free((void*)adr);
}

static uintptr_t umemHost_aligned_alloc_(umemVirtual * const this, size_t alignment, size_t size) {
  uintptr_t adr = 0;
  size_t extra = (alignment - 1) + sizeof(uintptr_t);
  size_t req = extra + (size ? size: 1);
  adr = umemHost_calloc_(this, req, 1);
  if (!umem_is_ok(this))
    return 0;
  uintptr_t aligned = adr + extra;
  aligned = aligned - (aligned % alignment);
  *((uintptr_t *)aligned - 1) = adr;
  return aligned;
}

static uintptr_t umemHost_aligned_origin_(umemVirtual * const this, uintptr_t aligned_adr) {
  return (aligned_adr ? *((uintptr_t *)aligned_adr - 1) : 0);
}

static void umemHost_aligned_free_(umemVirtual * const this, uintptr_t aligned_adr) {
  umemHost_free_(this, umemHost_aligned_origin_(this, aligned_adr));
}

static void umemHost_set_(umemVirtual * const this, uintptr_t adr, int c, size_t nbytes) {
  assert(this->type == umemHostDevice);
  HOST_CALL(this, memset((void*)adr, c, nbytes)==NULL, umemMemoryError, return,
	    "umemHost_set_: memset(&%" PRIxPTR ", %d, %zu)->NULL", adr, c, nbytes);
}

static void umemHost_copy_to_(umemVirtual * const this, uintptr_t src_adr,
			      umemVirtual * const that, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(this->type == umemHostDevice);
  if (that->type == umemHostDevice) {
    HOST_CALL(this, memcpy((void*)dest_adr, (void*)src_adr, nbytes)==NULL, umemMemoryError, return,
	      "umemHost_copy_to_: memcpy(%02" PRIxPTR ", %" PRIxPTR ", %zu)->NULL",
	      dest_adr, src_adr, nbytes);
  } else
    umem_copy_from(that, dest_adr, this, src_adr, nbytes);
}

static void umemHost_copy_from_(umemVirtual * const this, uintptr_t dest_adr,
				umemVirtual * const that, uintptr_t src_adr,
				size_t nbytes) {
  assert(this->type == umemHostDevice);
  if (that->type == umemHostDevice)
    umemHost_copy_to_(that, src_adr, this, dest_adr, nbytes);
  else
    umem_copy_to(that, src_adr, this, dest_adr, nbytes);
}

bool umemHost_is_same_device_(umemVirtual * const this, umemVirtual * const that) {
  return true;
}

/*
  umemHost constructor.
*/

void umemHost_ctor(umemHost * const this) {
  static struct umemVtbl const vtbl = {
    &umemHost_dtor_,
    &umemHost_is_same_device_,
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
  umemVirtual_ctor(&this->super, this);
  this->super.type = umemHostDevice;
  this->super.vptr = &vtbl;
}


/*
  umemVirtual copy methods that use host memory as an intermediate buffer.
 */
void umem_copy_to_via_host(void * const this, uintptr_t src_adr,
			   void * const that, uintptr_t dest_adr,
			   size_t nbytes) {
  umemHost host;
  umemHost_ctor(&host);
  if (umem_is_ok((void*)&host)) {
    uintptr_t host_adr = umem_alloc(&host, nbytes);
    if (umem_is_ok((void*)&host)) {
      umem_copy_to(this, src_adr, &host, host_adr, nbytes);
      if (umem_is_ok(this))
	umem_copy_from(that, dest_adr, &host, host_adr, nbytes);
      umem_free(&host, host_adr);
    }
  }
  umem_dtor(&host);
}

void umem_copy_from_via_host(void * const this, uintptr_t dest_adr,
			     void * const that, uintptr_t src_adr,
			     size_t nbytes) {
  umemHost host;
  umemHost_ctor(&host);
  if (umem_is_ok((void*)&host)) {
    uintptr_t host_adr = umem_alloc(&host, nbytes);
    if (umem_is_ok((void*)&host)) {
      umem_copy_to(that, src_adr, &host, host_adr, nbytes);
      if (umem_is_ok(that))
	umem_copy_from(this, dest_adr, &host, host_adr, nbytes);
      umem_free(&host, host_adr);
    }
  }
  umem_dtor(&host);
}
