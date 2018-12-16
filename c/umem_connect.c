#include "umem.h"

uintptr_t umem_connect(void * const src, uintptr_t src_adr, size_t nbytes, void * const dest, size_t dest_alignment) {
  if (umem_is_same_context(src, dest) && ((dest_alignment==0) || (src_adr % dest_alignment == 0)))
    return src_adr;
  uintptr_t dest_adr = 0;
  if (dest_alignment)
    dest_adr = umem_aligned_alloc(dest, dest_alignment, nbytes);
  else
    dest_adr = umem_alloc(dest, nbytes);
  if (!umem_is_ok(dest))
    return 0;
  umem_copy_to(src, src_adr, dest, dest_adr, nbytes);
  if (!umem_is_ok(dest))
    return 0;
  return dest_adr;
}

void umem_sync_from(void * const dest, uintptr_t dest_adr, void * const src, uintptr_t src_adr, size_t nbytes) {
  if (umem_is_same_context(dest, src) && src_adr == dest_adr)
    return;
  umem_copy_from(dest, dest_adr, src, src_adr, nbytes);
}

void umem_sync_to(void * const src, uintptr_t src_adr, void * const dest, uintptr_t dest_adr, size_t nbytes) {
  if (umem_is_same_context(src, dest) && src_adr == dest_adr)
    return;
  umem_copy_to(src, src_adr, dest, dest_adr, nbytes);
}

void umem_sync_from_safe(void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                         void * const src_ctx, uintptr_t src_adr, size_t src_size,
                         size_t nbytes) {
  if (umem_is_same_context(dest_ctx, src_ctx) && src_adr == dest_adr)
    return;
  umem_copy_from_safe(dest_ctx, dest_adr, dest_size, src_ctx, src_adr, src_size, nbytes);
}

void umem_sync_to_safe(void * const src_ctx, uintptr_t src_adr, size_t src_size,
                       void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                       size_t nbytes) {
  if (umem_is_same_context(src_ctx, dest_ctx) && src_adr == dest_adr)
    return;
  umem_copy_to_safe(src_ctx, src_adr, src_size, dest_ctx, dest_adr, dest_size, nbytes);
}

void umem_disconnect(void * const src, uintptr_t src_adr, void * const dest, uintptr_t dest_adr, size_t dest_alignment) {
  if (umem_is_same_context(src, dest)) {
    if (src_adr == dest_adr)
      return;
    if (dest_alignment)
      umem_aligned_free(dest, dest_adr);
    else
      umem_free(dest, dest_adr);
  } else {
    umem_free(dest, dest_adr);
  }
}
