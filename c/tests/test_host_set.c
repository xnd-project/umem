#include "umem_testing.h"

int main() {
  umemHost host;
  umemHost_ctor(&host);
  assert_is_ok(host);
  uintptr_t addr = umem_alloc(&host, 11);
  assert_is_ok(host);
  umem_set(&host, addr, 97, 5);
  umem_set(&host, addr+5, 98, 5);
  ((char*)addr)[10] = 0;
  assert_is_ok(host);
  assert_str_eq((char*)addr, "aaaaabbbbb");
  umem_free(&host, addr);
  assert_is_ok(host);
  umem_dtor(&host);
  assert_is_ok(host);
  RETURN_STATUS;
}
