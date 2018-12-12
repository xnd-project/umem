#include "umem_testing.h"

int main() {
  umemHost host, host2;
  umemHost_ctor(&host);
  umemHost_ctor(&host2);
  assert_is_ok(host);
  assert_is_ok(host2);
  assert_eq(umem_is_same_device(&host, &host), true);
  assert_eq(umem_is_same_device(&host, &host2), true);

  static char fn1[] = TMPDIR "test_is_same_device_1.txt";
  static char fn2[] = TMPDIR "test_is_same_device_2.txt";
  static char fn21[] = TMPDIR "test_is_same_device_2.txt";
  umemFile file1, file2, file21;
  umemFile_ctor(&file1, fn1, "wb");
  umemFile_ctor(&file2, fn2, "wb");
  umemFile_ctor(&file21, fn21, "wb");

  assert_eq(umem_is_same_device(&host, &file1), false);
  assert_eq(umem_is_same_device(&file1, &host), false);
  assert_eq(umem_is_same_device(&file1, &file1), true);
  assert_eq(umem_is_same_device(&file1, &file2), false);
  assert_eq(umem_is_same_device(&file2, &file21), true);

  assert_is_ok(host);
  assert_is_ok(host2);
  assert_is_ok(file1);
  assert_is_ok(file2);
  assert_is_ok(file21);
  
  umem_dtor(&file1);
  umem_dtor(&file2);
  umem_dtor(&file21);
  umem_dtor(&host2);
  umem_dtor(&host);
  RETURN_STATUS;
}
