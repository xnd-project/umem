#include "umem_testing.h"

int main() {
  umemCudaManaged ctx;
  umemHost host;
  umemCudaManaged_ctor(&ctx, 0, false, 0);
  umemHost_ctor(&host);
  assert_is_ok(ctx);
  static char text[] = "abcdefghij";
  uintptr_t addr = umem_alloc(&ctx, 10);
  assert_is_ok(ctx);
  umem_copy_from(&ctx, addr, &host, (uintptr_t)text, 10);
  assert_is_ok(ctx);
  assert_is_ok(host);
  uintptr_t addr2 = umem_alloc(&host, 10);
  assert_is_ok(host);
  umem_copy_to(&ctx, addr, &host, addr2, 10);
  assert_is_ok(host);
  assert_str_eq((char*)addr2, "abcdefghij");
  umem_free(&ctx, addr);
  assert_is_ok(ctx);
  umem_free(&host, addr2);
  assert_is_ok(host);
  umem_dtor(&ctx);
  assert_is_ok(ctx);
  umem_dtor(&host);
  assert_is_ok(host);
  RETURN_STATUS;
}
