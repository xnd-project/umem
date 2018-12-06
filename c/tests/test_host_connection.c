#include "umem_testing.h"

int main() {
  umemConnection con;
  umemHost host1, host2;
  umemHost_ctor(&host1);
  umemHost_ctor(&host2);
  size_t n = 10;
  uintptr_t adr1 = umem_alloc(&host1, n+1);
  ((char*)adr1)[n] = '\0';
  umem_set(&host1, adr1, 'A', n/2);
  umem_set(&host1, adr1+n/2, 'B', n/2);
  assert_str_eq((char*)adr1, "AAAAABBBBB");
  assert_is_ok(host1);
  assert_is_ok(host2);
  
  /* Same memory */
  uintptr_t adr2 = umem_connect(&host1, adr1, n, &host2);
  assert_is_ok(host1);
  assert_is_ok(host2);
  umem_sync_to(&host1, adr1, &host2, adr2, n);
  assert_is_ok(host1);
  assert_is_ok(host2);
  umem_set(&host2, adr2+n/4, 'C', n/2);
  umem_sync_from(&host1, adr1, &host2, adr2, n);
  assert_is_ok(host1);
  assert_is_ok(host2);
  assert_str_eq((char*)adr1, "AACCCCCBBB");
  umem_disconnect(&host1, &host2, adr2);
  assert_is_ok(host1);
  assert_is_ok(host2);

  /* Distinct memory */
  adr2 = umem_alloc(&host2, n + 1);
  ((char*)adr2)[n] = '\0';
  umem_set(&host2, adr2, 'D', n);
  assert_str_eq((char*)adr2, "DDDDDDDDDD");
  umem_sync_to(&host1, adr1, &host2, adr2, n);
  assert_str_eq((char*)adr2, "AACCCCCBBB");
  umem_set(&host2, adr2, 'E', n/2);
  assert_str_eq((char*)adr2, "EEEEECCBBB");
  assert_str_eq((char*)adr1, "AACCCCCBBB");
  umem_sync_from(&host1, adr1, &host2, adr2, n);
  umem_free(&host2, adr2);
  assert_str_eq((char*)adr1, "EEEEECCBBB");
  assert_is_ok(host1);
  assert_is_ok(host2);

  umem_free(&host1, adr1);
  umem_dtor(&host1);
  umem_dtor(&host2);
  RETURN_STATUS;
}
