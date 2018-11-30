#include "umem_testing.h"

int main() {
  umemFile file;
  umemFile_ctor(&file, TMPDIR "/test_file_alloc_free.txt", "w+b");
  uintptr_t addr = umem_alloc(&file, 10);
  assert_is_ok(file);
  umem_free(&file, addr);
  assert_is_ok(file);
  umemFile_dtor(&file);
  assert_is_ok(file);
  RETURN_STATUS;
}
