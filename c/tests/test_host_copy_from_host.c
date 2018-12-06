#include "umem_testing.h"

int main() {
  umemHost host;
  umemHost_ctor(&host);
  assert_is_ok(host);
  uintptr_t addr1 = umem_alloc(&host, 11);
  uintptr_t addr2 = umem_alloc(&host, 11);
  assert_is_ok(host);
  for (char i=0; i<10; ++i) ((char*)addr1)[(int)i] = i+97;
  ((char*)addr1)[10] = 0;
  assert_is_ok(host);
  assert_str_eq((char*)addr1, "abcdefghij");
  umem_copy_from(&host, addr2, &host, addr1, 11);
  assert_str_eq((char*)addr2, "abcdefghij");
  assert_str_eq((char*)addr2, (char*)addr1);
  umem_free(&host, addr2);
  umem_free(&host, addr1);
  assert_is_ok(host);
  umem_dtor(&host);
  assert_is_ok(host);
  RETURN_STATUS;
}
