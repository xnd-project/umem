#include "umem_testing.h"

int main() {
  {
    umem::RMM ctx(0);
    assert_is_ok(ctx);
    {
      umem::Address addr = ctx.alloc(10);
      assert_is_ok(ctx);
    }
    assert_is_ok(ctx);
  }
  RETURN_STATUS;
}
