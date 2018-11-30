#include "umem_testing.h"

int main() {
  static char fn[] = TMPDIR "/test_file_init.txt";
  umemFile file;
  umemHost host;
  umemHost_ctor(&host);
  assert_is_ok(host);
  umemFile_ctor(&file, fn, "wb");
  assert_is_ok(file);
  uintptr_t addr = umem_alloc(&file, 0); // open the file for writing
  static char text[] = "abcdefghij";
  assert_is_ok(file);
  umem_copy_from(&file, addr, &host, (uintptr_t)text, 10);
  assert_is_ok(file);
  assert_is_ok(host);
  umem_free(&file, addr); // close the file
  assert_is_ok(file);
  umemFile_ctor(&file, fn, "rb");
  assert_is_ok(file);
  addr = umem_alloc(&file, 0); // open the file for reading
  assert_is_ok(file);
  uintptr_t addr2 = umem_alloc(&host, 11);
  assert_is_ok(file);
  ((char*)addr2)[10] = 0;
  umem_copy_to(&file, addr, &host, addr2, 10);
  assert_is_ok(file);
  assert_str_eq((char*)addr2, text);
  assert_is_ok(file);
  umem_free(&host, addr2);
  umemFile_dtor(&file); // this also closes the file
  umemHost_dtor(&host);
  assert_is_ok(file);
  assert_is_ok(host);
  RETURN_STATUS;
}
