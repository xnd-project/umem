#include "umem_testing.h"

int main() {
  {
    umem::Cuda ctx(0);
    assert_eq(ctx.is_ok(), true);
    {
      umem::Address addr = ctx.alloc(10);
      assert_eq(ctx.is_ok(), true);
    }
    assert_eq(ctx.is_ok(), true);
  }
  RETURN_STATUS;
}
