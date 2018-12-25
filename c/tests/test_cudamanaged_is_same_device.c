#include "umem_testing.h"

int main() {
  umemHost host;
  umemCudaManaged ctx1, ctx12;
  umemHost_ctor(&host);
  umemCudaManaged_ctor(&ctx1, 0, false, 0);
  umemCudaManaged_ctor(&ctx12, 0, false, 0);
  assert_is_ok(host);
  assert_is_ok(ctx1);
  assert_is_ok(ctx12);
  assert_eq(umem_is_same_context(&ctx1, &host), false);
  assert_eq(umem_is_same_context(&ctx1, &ctx1), true);
  assert_eq(umem_is_same_context(&ctx1, &ctx12), true);
  assert_is_ok(host);
  assert_is_ok(ctx1);
  assert_is_ok(ctx12);
  umem_dtor(&host);
  umem_dtor(&ctx1);
  umem_dtor(&ctx12);
  RETURN_STATUS;
}
