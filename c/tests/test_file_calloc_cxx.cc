#include "umem_testing.h"

int main() {
  {
    umem::File file(TMPDIR "test_file_calloc_cxx.txt", "w+b");
    assert_eq(file.is_ok(), true);
    umem::Address addr = file.calloc(16, 10);
    assert_eq(file.is_ok(), true);
  }
  RETURN_STATUS;
}
