#include "umem_testing.h"

int main() {
  umemHost host1;
  umemCuda cuda;
  umemHost_ctor(&host1);
  umemCuda_ctor(&cuda, 0);
  size_t n = 10;
  uintptr_t adr1 = umem_alloc(&host1, n+1);
  uintptr_t adr = umem_alloc(&host1, n+1);
  ((char*)adr1)[n] = '\0';
  umem_set(&host1, adr1, 'A', n/2);
  umem_set(&host1, adr1+n/2, 'B', n/2);
  assert_str_eq((char*)adr1, "AAAAABBBBB");
  assert_is_ok(host1);
  assert_is_ok(cuda);

  /* Default alignment */
  uintptr_t adr2 = umem_connect(&host1, adr1, n+1, &cuda, 0);
  umem_copy_to(&cuda, adr2, &host1, adr, n+1);
  assert_nstr_eq(n, (char*)adr, "AAAAABBBBB");
  umem_set(&cuda, adr2+n/4, 'C', n/2);
  umem_copy_to(&cuda, adr2, &host1, adr, n+1);
  assert_nstr_eq(n, (char*)adr, "AACCCCCBBB");
  assert_nstr_eq(n, (char*)adr1, "AAAAABBBBB");
  umem_sync_from(&host1, adr1, &cuda, adr2, n);
  assert_nstr_eq(n, (char*)adr1, "AACCCCCBBB");
  umem_set(&host1, adr1+3, 'D', 2);
  assert_nstr_eq(n, (char*)adr1, "AACDDCCBBB");
  umem_sync_to(&host1, adr1, &cuda, adr2, n);
  umem_copy_to(&cuda, adr2, &host1, adr, n+1);
  assert_nstr_eq(n, (char*)adr, "AACDDCCBBB");  
  umem_disconnect(&host1, adr1, &cuda, adr2, 0);
  umem_free(&host1, adr1);
  umem_free(&host1, adr);

  /* Aligned memory */
  n = 1024;
  adr1 = umem_alloc(&host1, n);
  adr = umem_alloc(&host1, n);
  umem_set(&host1, adr1, 'A', n);
  assert_nstr_eq(10, (char*)adr1, "AAAAAAAAAA");
  size_t alignment = 1024;
  adr2 = umem_connect(&host1, adr1, n, &cuda, alignment);
  assert_is_ok(host1);
  assert_is_ok(cuda);
  umem_copy_to(&cuda, adr2, &host1, adr, n+1);
  assert_nstr_eq(10, (char*)adr, "AAAAAAAAAA");
  
  umem_set(&cuda, adr2+2, 'C', 4);
  umem_copy_to(&cuda, adr2, &host1, adr, n+1);
  assert_nstr_eq(10, (char*)adr, "AACCCCAAAA");
  umem_sync_from(&host1, adr1, &cuda, adr2, n);
  assert_nstr_eq(10, (char*)adr1, "AACCCCAAAA");
  umem_set(&host1, adr1+3, 'D', 2);
  assert_nstr_eq(10, (char*)adr1, "AACDDCAAAA");
  umem_sync_to(&host1, adr1, &cuda, adr2, n);
  umem_copy_to(&cuda, adr2, &host1, adr, n+1);
  assert_nstr_eq(10, (char*)adr, "AACDDCAAAA");
  
  umem_disconnect(&host1, adr1, &cuda, adr2, alignment);
  umem_free(&host1, adr1);
  umem_free(&host1, adr);;
  
  umem_dtor(&host1);
  umem_dtor(&cuda);
  RETURN_STATUS;
}
