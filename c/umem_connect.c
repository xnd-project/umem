#include "umem.h"

inline static bool are_accessible(void * src, void * dest) {
  return umem_is_accessible_from(src, dest) && umem_is_ok(src) && umem_is_accessible_from(dest, src) && umem_is_ok(dest);
}

uintptr_t umem_connect(void * const src_ctx, uintptr_t src_adr, size_t nbytes, void * const dest_ctx, size_t dest_alignment) {
  if (are_accessible(src_ctx, dest_ctx) && ((dest_alignment==0) || (src_adr % dest_alignment == 0)))
    return src_adr;
  uintptr_t dest_adr = 0;
  if (dest_alignment)
    dest_adr = umem_aligned_alloc(dest_ctx, dest_alignment, nbytes);
  else
    dest_adr = umem_alloc(dest_ctx, nbytes);
  if (!umem_is_ok(dest_ctx))
    return 0;
  umem_copy_to(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  if (!umem_is_ok(dest_ctx)) {
    if (dest_alignment)
      umem_aligned_free(dest_ctx, dest_adr);
    else
      umem_free(dest_ctx, dest_adr);
    return 0;
  }
  return dest_adr;
}

void umem_sync_from(void * const dest_ctx, uintptr_t dest_adr, void * const src_ctx, uintptr_t src_adr, size_t nbytes) {
  if (src_adr == dest_adr && are_accessible(dest_ctx, src_ctx))
    return;
  umem_copy_from(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
}

void umem_sync_to(void * const src_ctx, uintptr_t src_adr, void * const dest_ctx, uintptr_t dest_adr, size_t nbytes) {
  if (src_adr == dest_adr && are_accessible(src_ctx, dest_ctx))
    return;
  umem_copy_to(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
}

void umem_sync_from_safe(void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                         void * const src_ctx, uintptr_t src_adr, size_t src_size,
                         size_t nbytes) {
  if (src_adr == dest_adr && are_accessible(dest_ctx, src_ctx))
    return;
  umem_copy_from_safe(dest_ctx, dest_adr, dest_size, src_ctx, src_adr, src_size, nbytes);
}

void umem_sync_to_safe(void * const src_ctx, uintptr_t src_adr, size_t src_size,
                       void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                       size_t nbytes) {
  if (src_adr == dest_adr && are_accessible(src_ctx, dest_ctx))
    return;
  umem_copy_to_safe(src_ctx, src_adr, src_size, dest_ctx, dest_adr, dest_size, nbytes);
}

void umem_disconnect(void * const src_ctx, uintptr_t src_adr, void * const dest_ctx, uintptr_t dest_adr, size_t dest_alignment) {
  if (are_accessible(src_ctx, dest_ctx)) {
    if (src_adr == dest_adr)
      return;
    if (dest_alignment)
      umem_aligned_free(dest_ctx, dest_adr);
    else
      umem_free(dest_ctx, dest_adr);
  } else {
    umem_free(dest_ctx, dest_adr);
  }
}
