#include "umem_testing.h"

int main() {
  umemRMM dev;
  umemHost host;
  umemRMM_ctor(&dev, 0, 0, false);
  umemHost_ctor(&host);
  assert_is_ok(dev);
  static char text[] = "abcdefghij";
  uintptr_t addr = umem_alloc(&dev, 10);
  assert_is_ok(dev);
  umem_copy_from(&dev, addr, &host, (uintptr_t)text, 10);
  assert_is_ok(dev);
  assert_is_ok(host);
  uintptr_t addr2 = umem_alloc(&host, 10);
  assert_is_ok(host);
  umem_copy_to(&dev, addr, &host, addr2, 10);
  assert_is_ok(host);
  assert_str_eq((char*)addr2, "abcdefghij");
  umem_free(&dev, addr);
  assert_is_ok(dev);
  umem_free(&host, addr2);
  assert_is_ok(host);
  umem_dtor(&dev);
  assert_is_ok(dev);
  umem_dtor(&host);
  assert_is_ok(host);
  RETURN_STATUS;
}
