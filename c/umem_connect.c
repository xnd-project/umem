#include "umem.h"

uintptr_t umem_connect(void * const src, uintptr_t src_adr, size_t nbytes, void * const dest, size_t dest_alignment) {
  if (umem_is_same_device(src, dest) && ((dest_alignment==0) || (src_adr % dest_alignment == 0)))
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

void umem_sync_from(void * const src, uintptr_t src_adr, void * const dest, uintptr_t dest_adr, size_t nbytes) {
  if (umem_is_same_device(src, dest) && src_adr == dest_adr)
    return;
  umem_copy_from(src, src_adr, dest, dest_adr, nbytes);
}

void umem_sync_to(void * const src, uintptr_t src_adr, void * const dest, uintptr_t dest_adr, size_t nbytes) {
  if (umem_is_same_device(src, dest) && src_adr == dest_adr)
    return;
  umem_copy_to(src, src_adr, dest, dest_adr, nbytes);
}

void umem_disconnect(void * const src, uintptr_t src_adr, void * const dest, uintptr_t dest_adr, size_t dest_alignment) {
  if (umem_is_same_device(src, dest)) {
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
