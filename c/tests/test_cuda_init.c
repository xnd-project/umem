#include "umem_testing.h"

int main() {
  umemCuda cuda;
  umemHost host;
  umemCuda_ctor(&cuda, 0);
  umemHost_ctor(&host);
  assert_is_ok(cuda);
  static char text[] = "abcdefghij";
  uintptr_t addr = umem_alloc(&cuda, 10);
  assert_is_ok(cuda);
  umem_copy_from(&cuda, addr, &host, (uintptr_t)text, 10);
  assert_is_ok(cuda);
  assert_is_ok(host);
  uintptr_t addr2 = umem_alloc(&host, 10);
  assert_is_ok(host);
  umem_copy_to(&cuda, addr, &host, addr2, 10);
  assert_is_ok(host);
  assert_str_eq((char*)addr2, "abcdefghij");
  umem_free(&cuda, addr);
  assert_is_ok(cuda);
  umem_free(&host, addr2);
  assert_is_ok(host);
  umemCuda_dtor(&cuda);
  assert_is_ok(cuda);
  umemHost_dtor(&host);
  assert_is_ok(host);
  RETURN_STATUS;
}
