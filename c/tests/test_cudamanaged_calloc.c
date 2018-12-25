#include "umem_testing.h"

int main() {
  umemCudaManaged ctx;
  umemCudaManaged_ctor(&ctx, 0, false, 0);
  assert_is_ok(ctx);
  uintptr_t addr = umem_calloc(&ctx, 16, 10);
  assert_is_ok(ctx);
  umem_free(&ctx, addr);
  assert_is_ok(ctx);
  umem_dtor(&ctx);
  assert_is_ok(ctx);
  RETURN_STATUS;
}
