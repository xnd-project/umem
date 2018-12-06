#include "umem_testing.h"

#define CHECK_ALIGNED(ALIGNMENT, SIZE)                     \
  do {                                                     \
    adr = umem_aligned_alloc(&host, ALIGNMENT, SIZE);      \
    assert_int_eq(adr==0, 0);                              \
    assert_is_ok(host);                                    \
    int rem =  adr % ALIGNMENT;                            \
    assert_int_eq(rem, 0);                                 \
    umem_set(&host, adr, 255, SIZE);                       \
    assert_is_ok(host);                                    \
    uintptr_t oadr = umem_aligned_origin(&host, adr);      \
    assert_is_ok(host);                                    \
    assert_int_eq(oadr==0, 0);                             \
    /*binDump("adr", (void*)oadr, (SIZE+adr-oadr));*/      \
    umem_aligned_free(&host, adr);                         \
  } while(0)

#define CHECK_ALIGNED_FAIL(ALIGNMENT, SIZE, MESSAGE)          \
  do {                                                        \
    adr = umem_aligned_alloc(&host, ALIGNMENT, SIZE);         \
    assert_int_eq(adr==0, 1);                                    \
    assert_is_not_ok(host);                                   \
    assert_str_eq(umem_get_message(&host), MESSAGE);          \
    umem_clear_status(&host);                                 \
  } while(0)

int main() {
  umemHost host;
  umemHost_ctor(&host);
  uintptr_t adr = 0;
  CHECK_ALIGNED_FAIL(0, 10, "umemVirtual_aligned_alloc: alignment 0 must be power of 2");
  CHECK_ALIGNED(1, 10);
  CHECK_ALIGNED(2, 10);
  CHECK_ALIGNED(2, 0);
  CHECK_ALIGNED_FAIL(2, 1, "umemVirtual_aligned_alloc: size 1 must be multiple of alignment 2");
  CHECK_ALIGNED(2, 2);
  CHECK_ALIGNED_FAIL(3, 10, "umemVirtual_aligned_alloc: alignment 3 must be power of 2");
  CHECK_ALIGNED_FAIL(4, 10, "umemVirtual_aligned_alloc: size 10 must be multiple of alignment 4");
  CHECK_ALIGNED(4, 12);
  CHECK_ALIGNED(64, 512);
  CHECK_ALIGNED_FAIL(64, 1000, "umemVirtual_aligned_alloc: size 1000 must be multiple of alignment 64");
  assert_is_ok(host);
  umem_dtor(&host);
  RETURN_STATUS;
}
