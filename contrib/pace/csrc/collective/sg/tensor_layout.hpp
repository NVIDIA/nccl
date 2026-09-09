#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace pace {
/**
 * TensorLayout - iterates over a possibly non-contiguous tensor memory layout
 *
 * Given the shape of a contiguous tensor, pin one dimension to form a
 * non-contiguous tensor view, and read data out of that view.
 */
class TensorLayout {
public:
    TensorLayout() = default;
    /**
     * Constructor
     * @param base_ptr   start address of the tensor data
     * @param shape      size of each dimension
     * @param elem_size  size of a single element (bytes)
     * @param pin_dim    the pinned dimension
     */
    TensorLayout(void* base_ptr,
                 const std::vector<size_t>& shape,
                 size_t elem_size, size_t pin_dim)
        : base_ptr_(static_cast<uint8_t*>(base_ptr)), x(0), z(0) {
        /**
         * [X, p/Y, Z]
         * X is the large block, Z the small block in bytes; Z bytes are contiguous.
         */
        size_t d_idx = 0;
        X = 1;
        Y = 0;
        Z = elem_size;
        for (size_t dim : shape) {
            if (d_idx == pin_dim) {
                Y = dim;
            } else if (d_idx < pin_dim) {
                X *= dim;
            } else {
                // d_idx > pin_dim
                Z *= dim;
            }
            d_idx += 1;
        }
        total_bytes = X * Y * Z;
    }

    /**
     * Get the next contiguous memory block
     * @param n  requested number of bytes
     * @param pin_dim_value value of the pinned dimension
     * @return   pair<pointer, byte count>: how many bytes are contiguous from
     *           that pointer. When the returned count < n, the following bytes
     *           are non-contiguous and this must be called again. When the
     *           returned count == 0, all data has been traversed.
     *           Does not actually advance the pointer.
     */
    std::pair<void*, size_t> try_get_slice(size_t pin_dim_value, size_t n) {
        if (x >= X) return {nullptr, 0};

        size_t can_return = std::min(n, Z - z);

        void *ptr = reinterpret_cast<uint8_t*>(base_ptr_) + x * Y * Z + pin_dim_value * Z + z;

        return {ptr, can_return};
    }

    uint8_t *get_ptr(size_t pin_dim_value) {
        return reinterpret_cast<uint8_t*>(base_ptr_) + x * Y * Z + pin_dim_value * Z + z;
    }

    /**
     * Confirm that the fetched memory block has been consumed
     * @param n  number of bytes consumed
     * Advances the pointer.
     */
    void confirm_slice(size_t n) {
        z += n;
        if (z >= Z) {
            x += 1;
            z = 0;
        }
    }

    size_t get_total_bytes() const {
        return total_bytes;
    }
    /**
     * Reset the iterator to the start position
     */
    void reset() {
        x = 0;
        z = 0;
    }

    // Accessors for internal dimensions (needed for cudaMemcpy2D optimization)
    size_t get_X() const { return X; }
    size_t get_Y() const { return Y; }
    size_t get_Z() const { return Z; }
    uint8_t* get_base_ptr() const { return base_ptr_; }
    size_t get_current_x() const { return x; }
    size_t get_current_z() const { return z; }

private:
    /**
    * [X, p/Y, Z]
    * X is the large block, Z the small block; the small block of Z elements is contiguous.
    */
    uint8_t* base_ptr_;
    size_t X, Y, Z;
    size_t x, z;
    size_t total_bytes;
};

}