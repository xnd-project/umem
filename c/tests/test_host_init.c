#include "umem_testing.h"

int main() {
  umemHost host;
  umemHost_ctor(&host);
  assert_is_ok(host);
  uintptr_t addr = umem_alloc(&host, 11);
  assert_is_ok(host);
  for (char i=0; i<10; ++i) ((char*)addr)[i] = i+97;
  ((char*)addr)[10] = 0;
  assert_is_ok(host);
  assert_str_eq((char*)addr, "abcdefghij");
  umem_free(&host, addr);
  assert_is_ok(host);
  umemHost_dtor(&host);
  assert_is_ok(host);
  RETURN_STATUS;
}
