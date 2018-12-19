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
      umem::RMM dev(0);
      umem::Host host;
      assert_is_ok(dev);
      static char text[] = "abcdefghij";
      umem::Address addr = dev.alloc(10);
      assert_is_ok(dev);
      addr.copy_from(text, 10);
      assert_is_ok(dev);
      assert_is_ok(host);
      umem::Address addr2 = host.alloc(10);
      assert_is_ok(host);
      addr.copy_to(addr2, 10);
      assert_is_ok(host);
      assert_is_ok(dev);
      assert_str_eq((char*)addr2, "abcdefghij");
    }
    status = rmmFinalize();
    assert_eq(status, RMM_SUCCESS);
  }
  RETURN_STATUS;
}
