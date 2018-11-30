#include "umem_testing.h"

int main() {
  umemVirtual virt;
  umemVirtual_ctor(&virt);
  umem_set_status(&virt, umemNotImplementedError, "notimpl");
  umem_set_status(&virt, umemNotImplementedError, "notimpl2");
  assert(umem_get_status(&virt) == umemNotImplementedError);
  assert_str_eq(umem_get_message(&virt), "notimpl\nnotimpl2");
  assert_is_not_ok(virt);
  umemVirtual_dtor(&virt); // destructor also clears
  assert_is_ok(virt);
  assert_str_eq(umem_get_message(&virt), "");
  RETURN_STATUS;
}
