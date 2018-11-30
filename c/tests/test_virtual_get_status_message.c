#include "umem_testing.h"

int main() {
  umemVirtual virt;
  umemVirtual_ctor(&virt);
  assert(umem_get_status(&virt) == umemOK);
  assert_is_ok(virt);
  assert_str_eq(umem_get_message(&virt), "");
  umemVirtual_dtor(&virt);
  RETURN_STATUS;
}
