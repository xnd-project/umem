#include "umem_testing.h"

#define CHECK_ALIGNED(ALIGNMENT, SIZE)                     \
  do {                                                     \
    adr = umem_aligned_alloc(&file, ALIGNMENT, SIZE);      \
    assert_int_eq(adr == 0, 0);                            \
    assert_is_ok(file);                                    \
    int rem =  adr % ALIGNMENT;                            \
    assert_int_eq(rem, 0);                                 \
    umem_set(&file, adr, 255, SIZE);                       \
    uintptr_t oadr = umem_aligned_origin(&file, adr);      \
    /*binDump("adr", (void*)oadr, (SIZE+adr-oadr));*/      \
    umem_aligned_free(&file, adr);                         \
  } while(0)

#define CHECK_ALIGNED_FAIL(ALIGNMENT, SIZE, MESSAGE)          \
  do {                                                        \
    adr = umem_aligned_alloc(&file, ALIGNMENT, SIZE);         \
    assert_is_not_ok(file);                                   \
    assert_str_eq(umem_get_message(&file), MESSAGE);          \
    umem_clear_status(&file);                                 \
  } while(0)

int main() {
  umemFile file;
  umemFile_ctor(&file, TMPDIR "/test_file_aligned_alloc.txt", "w+b");
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
  assert_is_ok(file);
  umem_dtor(&file);
  RETURN_STATUS;
}
