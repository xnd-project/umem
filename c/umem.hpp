#ifndef UMEM_HPP
#define UMEM_HPP
/*
  Author: Pearu Peterson
  Created: December 2018
*/

#include <memory>
#include <iostream>
#include <string>

namespace umem {

  typedef unsigned short alignment_t;
  
  class Address;
  
  class Context {
  public:
    
    inline Address alloc(size_t nbytes);
    inline Address calloc(size_t nmemb, size_t size);
    inline Address aligned_alloc(alignment_t alignment, size_t size);


    inline bool operator == (Context &other) { return umem_is_same_device(get_raw_context(), other.get_raw_context()); }
    inline bool operator != (Context &other) { return !(*this == other); }
    
    inline bool is_ok() { return umem_is_ok(get_raw_context()); };
    inline std::string get_message() { return umem_get_message(get_raw_context()); }
    inline umemStatusType get_status() { return umem_get_status(get_raw_context()); }
    inline void set_status(umemStatusType status, const char* message) { umem_set_status(get_raw_context(), status, message); }
    inline void set_status(umemStatusType status, std::string message) { umem_set_status(get_raw_context(), status, message.c_str()); }
    inline void clear_status() { umem_clear_status(get_raw_context()); }
    
    virtual void * const get_raw_context() = 0;
  };


  class Host : public Context {
  private:
    umemHost ctx;
  public:
    Host() {
      std::cout << "  Host::Host()\n";
      umemHost_ctor(&ctx);
    }
    ~Host() {
      std::cout << "  Host::~Host()\n";
      umem_dtor(&ctx);
    }

    inline void * const get_raw_context() { return &(this->ctx); }
  };


  class File : public Context {
  private:
    umemFile ctx;
  public:
    File(std::string filename, std::string mode="wb") {
      std::cout << "  File::File(\""<<filename<<"\", mode=\""<<mode<<"\")\n";
      umemFile_ctor(&ctx, filename.c_str(), mode.c_str());
    }
    ~File() {
      std::cout << "  File::~File()\n";
      umem_dtor(&ctx);
    }
    inline void * const get_raw_context() { return &(this->ctx); }
  };

  
  class Address {
  private:
    uintptr_t adr_;
    alignment_t alignment_;
    size_t size_;
    void * raw_ctx;
    bool own;
    Address* mate_; // TODO: use shared pointer?
  public:
    Address(uintptr_t adr, alignment_t alignment,  size_t size, void * raw_ctx, bool own = false,
            Address* mate = NULL):
      adr_(adr), alignment_(alignment), size_(size), raw_ctx(raw_ctx), own(own), mate_(mate) {
      std::cout << "  Address::Address("<<adr_<<", "<<alignment_<<", "<<size_<<", ctx="<<raw_ctx<<", own="<<own<<", mate="<<mate_<<")\n";
    }
    ~Address() {
      std::cout << "  Address::~Address()["<<adr_<<", "<<alignment_<<", "<<size_<<", ctx="<<raw_ctx<<", own="<<own<<", mate="<<mate_<<"]\n";
      if (own) {
        if (mate_ == NULL) {
          if (alignment_)
            umem_aligned_free(raw_ctx, adr_);
          else
            umem_free(raw_ctx, adr_);
        } else
          umem_disconnect(mate_->get_raw_context(), (uintptr_t)(*mate_), raw_ctx, (uintptr_t)adr_, alignment_);
      }
    }

    inline size_t size() const { return size_; }
    inline size_t alignment() const { return alignment_; }

    inline void copy_to(Address& dest, size_t nbytes) { umem_copy_to_safe(raw_ctx, adr_, size_, dest.get_raw_context(), (uintptr_t)dest, dest.size(), nbytes); }
    inline void copy_from(Address& src, size_t nbytes) { umem_copy_from_safe(raw_ctx, adr_, size_, src.get_raw_context(), (uintptr_t)src, src.size(), nbytes); }
    inline void copy_from(std::string src, size_t nbytes) { umem_copy_from_safe(raw_ctx, adr_, size_, ((umemVirtual*)raw_ctx)->host, (uintptr_t)src.c_str(), src.size(), nbytes); }
    inline Address connect(Context& dest, size_t dest_nbytes, alignment_t dest_alignment = 0) {
      return Address(umem_connect(raw_ctx, adr_, dest_nbytes, dest.get_raw_context(), dest_alignment), dest_alignment, dest_nbytes, dest.get_raw_context(), true, this);
    }
    inline void sync_to(Address& dest, size_t nbytes) { umem_sync_to_safe(raw_ctx, adr_, size_, dest.get_raw_context(), (uintptr_t)dest, dest.size(), nbytes); }
    inline void sync_from(Address& src, size_t nbytes) { umem_sync_from_safe(raw_ctx, adr_, size_, src.get_raw_context(), (uintptr_t)src, src.size(), nbytes); }
    inline void sync(size_t nbytes) { assert(mate_ != NULL); umem_sync_to_safe(raw_ctx, adr_, size_, mate_->get_raw_context(), (uintptr_t)(*mate_), mate_->size(), nbytes); }
    inline void update(size_t nbytes) { assert(mate_ != NULL);umem_sync_from_safe(raw_ctx, adr_, size_, mate_->get_raw_context(), (uintptr_t)(*mate_), mate_->size(), nbytes); }
    
    inline char& operator[] (size_t n) { assert(n<size_); return ((char*)adr_)[n]; }
    inline void set(int c, size_t nbytes) { umem_set_safe(raw_ctx, adr_, size_, c, nbytes); };

    inline operator uintptr_t() { return adr_; }
    inline operator void*() { return (void*)adr_; }
    inline operator bool*() { return (bool*)adr_; }
    inline operator char*() { return (char*)adr_; }
    inline operator short*() { return (short*)adr_; }
    inline operator int*() { return (int*)adr_; }
    inline operator long*() { return (long*)adr_; }
    inline operator long long*() { return (long long*)adr_; }
    inline operator float*() { return (float*)adr_; }
    inline operator double*() { return (double*)adr_; }
    inline operator unsigned char*() { return (unsigned char*)adr_; }
    inline operator unsigned short*() { return (unsigned short*)adr_; }
    inline operator unsigned int*() { return (unsigned int*)adr_; }
    inline operator unsigned long*() { return (unsigned long*)adr_; }
    inline operator unsigned long long*() { return (unsigned long long*)adr_; }
    /* 
       To cast to user-defined types, use
         mytype* ptr = (mytype*)(void*)adr
       where adr is Address object.
     */
    inline bool operator == (Address& other) const { return adr_ == (uintptr_t)other; }
    inline bool operator == (uintptr_t i) const { return adr_ == i; }
    inline bool operator == (int i) const { return adr_ == (unsigned int)i; }
    //inline bool operator == (size_t i) const { return adr_ == i; }
    //inline bool operator != (uintptr_t i) const { return adr_ != i; }
    inline Address operator + (int i) const { return Address(adr_ + i, alignment_, (i<(int)size_ ? size_ - i : 0), raw_ctx); }
    inline Address operator + (size_t i) const { return Address(adr_ + i, alignment_, (i<size_ ? size_ - i : 0), raw_ctx); }
    //inline Address operator - (int i) const { return Address(adr_ - i, alignment_, size + i, raw_ctx); }
    //inline Address operator - (size_t i) const { return Address(adr_ - i, alignment_, size + i, raw_ctx); }
    inline int operator % (int i) const { return adr_ % i; }
    inline size_t operator % (size_t i) const { return adr_ % i; }
    inline Address origin() const { return (alignment_ ? Address(umem_aligned_origin(raw_ctx, adr_), 0, size_, raw_ctx) : Address(adr_, alignment_, size_, raw_ctx)); }
    
  protected:
    inline void * const get_raw_context() { return raw_ctx; };
  private:
    Address& operator=(const Address&); // Disallow assignment operator.
    //Address(const Address&);            // Disallow copy constructor. MSVC requires copy constructor.
    
  };

  // Implementations:
  inline Address Context::alloc(size_t nbytes) { return Address(umem_alloc(get_raw_context(), nbytes), 0, nbytes, get_raw_context(), true); }
  inline Address Context::calloc(size_t nmemb, size_t size) {
    return Address(umem_calloc(get_raw_context(), nmemb, size), 0, nmemb * size, get_raw_context(), true); } // TODO: check overflow
  inline Address Context::aligned_alloc(alignment_t alignment, size_t size) {
    return Address(umem_aligned_alloc(get_raw_context(), alignment, size), alignment, size, get_raw_context(), true); }
  
}

#endif
