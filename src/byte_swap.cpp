#include <bit>
#include <cstdint>
#include <type_traits>
#include "new_subs.h"


template<typename T>
requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
constexpr T Subs::byte_swap(T value) noexcept {
    if constexpr (std::is_integral_v<T>) {
        // Directly use the library byte-swap for integers
        return byteswap(value);
    } else {
        // For floats/doubles: cast to unsigned integer of same size,
        // byteswap that, then cast back
        using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
        U as_int = std::bit_cast<U>(value);
        as_int = byteswap(as_int);
        return std::bit_cast<T>(as_int);
    }
}
