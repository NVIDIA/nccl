# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from libc.stdint cimport int32_t, int64_t, intptr_t
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libcpp cimport nullptr_t, nullptr
from libcpp.memory cimport unique_ptr

from ..cynccl cimport ncclUniqueId


cdef extern from * nogil:
    """
    template<typename T>
    class nullable_unique_ptr {
      public:
        nullable_unique_ptr() noexcept = default;

        nullable_unique_ptr(std::nullptr_t) noexcept = delete;

        explicit nullable_unique_ptr(T* data, bool own_data):
            own_data_(own_data)
        {
            if (own_data)
                manager_.reset(data);
            else
                raw_data_ = data;
        }

        nullable_unique_ptr(const nullable_unique_ptr&) = delete;

        nullable_unique_ptr& operator=(const nullable_unique_ptr&) = delete;

        nullable_unique_ptr(nullable_unique_ptr&& other) noexcept
        {
            own_data_ = other.own_data_;
            other.own_data_ = false;  // ownership is transferred
            if (own_data_)
            {
                manager_ = std::move(other.manager_);
                raw_data_ = nullptr;  // just in case
            }
            else
            {
                manager_.reset(nullptr);  // just in case
                raw_data_ = other.raw_data_;
            }
        }

        nullable_unique_ptr& operator=(nullable_unique_ptr&& other) noexcept
        {
            own_data_ = other.own_data_;
            other.own_data_ = false;  // ownership is transferred
            if (own_data_)
            {
                manager_ = std::move(other.manager_);
                raw_data_ = nullptr;  // just in case
            }
            else
            {
                manager_.reset(nullptr);  // just in case
                raw_data_ = other.raw_data_;
            }
            return *this;
        }

        ~nullable_unique_ptr() = default;

        void reset(T* data, bool own_data)
        {
            own_data_ = own_data;
            if (own_data_)
            {
                manager_.reset(data);
                raw_data_ = nullptr;
            }
            else
            {
                manager_.reset(nullptr);
                raw_data_ = data;
            }
        }

        void swap(nullable_unique_ptr& other) noexcept
        {
            std::swap(manager_, other.manager_);
            std::swap(raw_data_, other.raw_data_);
            std::swap(own_data_, other.own_data_);
        }

        /*
         * Get the pointer to the underlying object (this is different from data()!).
         */
        T* get() const noexcept
        {
            if (own_data_)
                return manager_.get();
            else
                return raw_data_;
        }

        /*
         * Get the pointer to the underlying buffer (this is different from get()!).
         */
        void* data() noexcept
        {
            if (own_data_)
                return manager_.get()->data();
            else
                return raw_data_;
        }

        T& operator*()
        {
            if (own_data_)
                return *manager_;
            else
                return *raw_data_;
        }

      private:
        std::unique_ptr<T> manager_{};
        T* raw_data_{nullptr};
        bool own_data_{false};
    };
    """
    # xref: cython/Cython/Includes/libcpp/memory.pxd
    cdef cppclass nullable_unique_ptr[T]:
        nullable_unique_ptr()
        nullable_unique_ptr(T*, cppbool)
        nullable_unique_ptr(nullable_unique_ptr[T]&)

        # Modifiers
        void reset(T*, cppbool)
        void swap(nullable_unique_ptr&)

        # Observers
        T* get()
        T& operator*()
        void* data()


ctypedef fused ResT:
    int
    int32_t
    int64_t
    char
    float
    double
    ncclUniqueId


ctypedef fused PtrT:
    void


cdef cppclass nested_resource[T]:
    nullable_unique_ptr[ vector[intptr_t] ] ptrs
    nullable_unique_ptr[ vector[vector[T]] ] nested_resource_ptr


# accepts the output pointer as input to use the return value for exception propagation
cdef int get_resource_ptr(nullable_unique_ptr[vector[ResT]] &in_out_ptr, object obj, ResT* __unused) except 1
cdef int get_resource_ptrs(nullable_unique_ptr[ vector[PtrT*] ] &in_out_ptr, object obj, PtrT* __unused) except 1
cdef int get_nested_resource_ptr(nested_resource[ResT] &in_out_ptr, object obj, ResT* __unused) except 1

cdef bint is_nested_sequence(data)
cdef void* get_buffer_pointer(buf, Py_ssize_t size, readonly=*) except*
