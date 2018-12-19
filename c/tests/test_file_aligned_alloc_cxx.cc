#include "umem_testing.h"

#define CHECK_ALIGNED(ALIGNMENT, SIZE)                         \
  do {                                                         \
    umem::Address adr = file.aligned_alloc(ALIGNMENT, SIZE);   \
    assert_is_ok(file);                                        \
    assert_eq((uintptr_t)adr>0, true);                         \
    int rem =  adr % ALIGNMENT;                                \
    assert_int_eq(rem, 0);                                     \
    adr.set(255, SIZE);                                        \
    assert_is_ok(file);                                        \
  } while(0)

#define CHECK_ALIGNED_FAIL(ALIGNMENT, SIZE, MESSAGE)                    \
  do {                                                                  \
    umem::Address adr = file.aligned_alloc(ALIGNMENT, SIZE);            \
    assert_is_not_ok(file);                                             \
    assert_str_eq(file.get_message().c_str(), MESSAGE);                 \
    file.clear_status();                                                \
    assert_is_ok(file);                                                 \
  } while(0)

int main() {
  {
    umem::File file(TMPDIR "test_file_aligned_alloc_cxx.txt", "w+b");
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
  }
  RETURN_STATUS;
}
