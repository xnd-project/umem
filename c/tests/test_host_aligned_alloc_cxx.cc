#include "umem_testing.h"

#define CHECK_ALIGNED(ALIGNMENT, SIZE)                     \
  do {                                                     \
    umem::Address adr = host.aligned_alloc(ALIGNMENT, SIZE);     \
    assert_int_eq(adr==0, false);                              \
    assert_eq(host.is_ok(), true);                              \
    int rem =  adr.adr % ALIGNMENT;                                \
    assert_int_eq(rem, 0);                                 \
    adr.set(255, SIZE);                                   \
    assert_eq(host.is_ok(), true);                              \
    uintptr_t oadr = adr.origin().adr;                      \
    assert_eq(host.is_ok(), true);                              \
    assert_int_eq(oadr==0, false);                              \
    /*binDump("adr", (void*)oadr, (SIZE+adr-oadr));*/      \
  } while(0)

#define CHECK_ALIGNED_FAIL(ALIGNMENT, SIZE, MESSAGE)          \
  do {                                                        \
    umem::Address adr = host.aligned_alloc(ALIGNMENT, SIZE);      \
    assert_int_eq(adr==0, true);                                    \
    assert_eq(host.is_ok(), false);                                \
    assert_str_eq(host.get_message().c_str(), MESSAGE);            \
    host.clear_status();                                 \
  } while(0)

int main() {
  {
    umem::Host host;
    CHECK_ALIGNED_FAIL(0, 10, "umem_aligned_alloc: alignment 0 must be power of 2");
    CHECK_ALIGNED(1, 10);
    CHECK_ALIGNED(2, 10);
    CHECK_ALIGNED(2, 0);
    CHECK_ALIGNED_FAIL(2, 1, "umem_aligned_alloc: size 1 must be multiple of alignment 2");
    CHECK_ALIGNED(2, 2);
    CHECK_ALIGNED_FAIL(3, 10, "umem_aligned_alloc: alignment 3 must be power of 2");
    CHECK_ALIGNED_FAIL(4, 10, "umem_aligned_alloc: size 10 must be multiple of alignment 4");
    CHECK_ALIGNED(4, 12);
    CHECK_ALIGNED(64, 512);
    CHECK_ALIGNED_FAIL(64, 1000, "umem_aligned_alloc: size 1000 must be multiple of alignment 64");

    assert_eq(host.is_ok(), true);
  }
  RETURN_STATUS;
}
