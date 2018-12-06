#include "umem_testing.h"

int main() {
  umemFile file;
  umemFile_ctor(&file, TMPDIR "test_file_calloc.txt", "w+b");
  uintptr_t addr = umem_calloc(&file, 16, 10);
  assert_is_ok(file);
  umem_free(&file, addr);
  assert_is_ok(file);
  umem_dtor(&file);
  assert_is_ok(file);
  RETURN_STATUS;
}
