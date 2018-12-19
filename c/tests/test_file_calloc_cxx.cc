#include "umem_testing.h"

int main() {
  {
    umem::File file(TMPDIR "test_file_calloc_cxx.txt", "w+b");
    assert_is_ok(file);
    umem::Address addr = file.calloc(16, 10);
    assert_is_ok(file);
  }
  RETURN_STATUS;
}
