#include "umem_testing.h"
#include "rmm.h"

int main() {
  {
    rmmOptions_t options;
    options.allocation_mode = PoolAllocation;
    //options.allocation_mode = CudaDefaultAllocation;
    options.initial_pool_size = 1024*100;
    options.enable_logging = false;
    rmmError_t status = rmmInitialize(&options);
    errno = 0; // rmmInitialize may result in errno=17 [File exists], so resetting
    assert_eq(status, RMM_SUCCESS);
    {
      umem::RMM ctx(0);
      assert_is_ok(ctx);
      {
        umem::Address addr = ctx.alloc(10);
        assert_is_ok(ctx);
      }
      assert_is_ok(ctx);
    }
    status = rmmFinalize();
    assert_eq(status, RMM_SUCCESS);
  }
  RETURN_STATUS;
}
