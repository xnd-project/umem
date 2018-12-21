#include "umem_testing.h"

int main() {
  umemRMM rmm;
  umemRMM* ctx = &rmm;
  umemRMM_ctor(ctx, 0, 0, false);
  assert_is_ok(rmm);
  uintptr_t addr = umem_alloc(ctx, 10);
  assert_is_ok(rmm);
  umem_free(ctx, addr);
  assert_is_ok(rmm);
  umem_dtor(ctx);
  assert_is_ok(rmm);
  RETURN_STATUS;
}
