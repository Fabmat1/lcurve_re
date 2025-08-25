// Array1D.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace Subs {

template<typename T>
class Array1D {
public:
    using value_type      = T;
    using container_type  = std::vector<T>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using iterator        = typename container_type::iterator;
    using const_iterator  = typename container_type::const_iterator;

    /* construction ------------------------------------------------------------------ */

    Array1D() = default;
    explicit Array1D(size_type n)               : data_(n)                    {}
    explicit Array1D(container_type v)          : data_(std::move(v))         {}
    Array1D(std::initializer_list<T> il)        : data_(il)                   {}

    template<std::ranges::input_range R>
    explicit Array1D(R&& r)
        : data_(std::ranges::begin(r), std::ranges::end(r)) {}

    /* basic container interface ------------------------------------------------------ */

    [[nodiscard]] size_type  size()   const noexcept { return data_.size(); }
    [[nodiscard]] bool       empty()  const noexcept { return data_.empty(); }
    [[nodiscard]] reference       operator[](size_type i)       { return data_[i]; }
    [[nodiscard]] const_reference operator[](size_type i) const { return data_[i]; }
    [[nodiscard]] T*       data()       noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    iterator       begin()  noexcept { return data_.begin();  }
    const_iterator begin()  const noexcept { return data_.begin();  }
    iterator       end()    noexcept { return data_.end();    }
    const_iterator end()    const noexcept { return data_.end();    }

    // growth / size mgmt
    void      reserve(size_type n)           { data_.reserve(n); }
    void      resize (size_type n)           { data_.resize (n); }
    size_type capacity()               const { return data_.capacity(); }
    void      clear()                        { data_.clear(); }

    // element insertion
    void push_back(const T& v)               { data_.push_back(v); }
    void push_back(T&& v)                    { data_.push_back(std::move(v)); }

    template<typename... Args>
    T& emplace_back(Args&&... args) {
        return data_.emplace_back(std::forward<Args>(args)...);
    }

    /* element-wise arithmetic with a scalar ------------------------------------------ */

    Array1D& operator+=(const T& c) { for (auto& e : data_) e += c; return *this; }
    Array1D& operator-=(const T& c) { for (auto& e : data_) e -= c; return *this; }
    Array1D& operator*=(const T& c) { for (auto& e : data_) e *= c; return *this; }
    Array1D& operator/=(const T& c) { for (auto& e : data_) e /= c; return *this; }

    /* element-wise arithmetic with another array ------------------------------------- */

    template<typename U>
    Array1D& operator+=(const Array1D<U>& rhs) { apply(rhs, std::plus<>{});  return *this; }
    template<typename U>
    Array1D& operator-=(const Array1D<U>& rhs) { apply(rhs, std::minus<>{}); return *this; }
    template<typename U>
    Array1D& operator*=(const Array1D<U>& rhs) { apply(rhs, std::multiplies<>{}); return *this; }
    template<typename U>
    Array1D& operator/=(const Array1D<U>& rhs) { apply(rhs, std::divides<>{});    return *this; }

    /* math helpers ------------------------------------------------------------------- */

    [[nodiscard]] T max()   const { return *std::ranges::max_element(data_); }
    [[nodiscard]] T min()   const { return *std::ranges::min_element(data_); }
    [[nodiscard]] T sum()   const { return std::accumulate(begin(), end(), T{}); }
    [[nodiscard]] T mean()  const { return empty() ? T{} : sum() / static_cast<T>(size()); }

    [[nodiscard]] T length() const
    {
        return std::sqrt(std::transform_reduce(begin(), end(), T{}, std::plus<>{},
                                               [](T v) { return v * v; }));
    }

    bool monotonic() const
    {
        return std::ranges::is_sorted(data_) ||
               std::ranges::is_sorted(data_, std::greater<>{});
    }

    /* percentiles & selection -------------------------------------------------------- */

    // k-th smallest (0-based), scrambles the copy, NOT the original
    T select(size_type k) const
    {
        if (k >= size()) throw std::out_of_range("select(): k out of range");
        container_type tmp = data_;
        std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
        return tmp[k];
    }

    T centile(double p) const
    {
        if (empty()) throw std::runtime_error("centile() on empty array");
        p = std::clamp(p, 0.0, 100.0);
        size_type k = static_cast<size_type>(p / 100.0 * static_cast<double>(size() - 1));
        return select(k);
    }

    T median() const
    {
        if (empty()) throw std::runtime_error("median() on empty array");
        const size_type mid = size() / 2;
        if (size() % 2 == 1) return select(mid);
        return (select(mid - 1) + select(mid)) / T{2};
    }

    /* sort and permutation key ------------------------------------------------------- */

    // Sorts *in place* and returns the permutation used.
    std::vector<size_type> sort()
    {
        std::vector<size_type> idx(size());
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(),
                  [this](size_type i, size_type j) { return data_[i] < data_[j]; });

        container_type sorted(data_.size());
        for (size_type i = 0; i < size(); ++i) sorted[i] = data_[idx[i]];
        data_.swap(sorted);
        return idx;
    }

    /* locate / hunt ------------------------------------------------------------------ */

    // position of the first element that is >= x  (ascending)  / > x  (descending)
    size_type locate(const T& x) const
    {
        if (!monotonic())
            throw std::logic_error("locate(): array is not monotonic");

        if (data_.front() <= data_.back()) {          // ascending
            return std::ranges::lower_bound(data_, x) - begin();
        } else {                                      // descending
            return std::ranges::lower_bound(data_, x, std::greater<>{}) - begin();
        }
    }

    /* trigonometric transforms ------------------------------------------------------- */

    void to_cos() { transform_inplace([](T v){ return std::cos(v); }); }
    void to_sin() { transform_inplace([](T v){ return std::sin(v); }); }

private:
    container_type data_;

    /* helpers ------------------------------------------------------------------------ */

    template<typename U, typename Op>
    void apply(const Array1D<U>& rhs, Op op)
    {
        if (size() != rhs.size())
            throw std::length_error("Array size mismatch");
        for (size_type i = 0; i < size(); ++i)
            data_[i] = op(data_[i], rhs[i]);
    }

    template<typename F>
    void transform_inplace(F&& f)
    {
        std::ranges::for_each(data_, [&](T& v) { v = f(v); });
    }
};

/* ------------------------------------------------------------------------- */
/* free operators (defined in terms of the member operators)                 */
/* ------------------------------------------------------------------------- */

template<typename T, typename U>
Array1D<T> operator+(Array1D<T> lhs, const Array1D<U>& rhs) { lhs += rhs; return lhs; }
template<typename T, typename U>
Array1D<T> operator-(Array1D<T> lhs, const Array1D<U>& rhs) { lhs -= rhs; return lhs; }
template<typename T, typename U>
Array1D<T> operator*(Array1D<T> lhs, const Array1D<U>& rhs) { lhs *= rhs; return lhs; }
template<typename T, typename U>
Array1D<T> operator/(Array1D<T> lhs, const Array1D<U>& rhs) { lhs /= rhs; return lhs; }

template<typename T>
Array1D<T> operator+(Array1D<T> lhs, const T& c) { lhs += c; return lhs; }
template<typename T>
Array1D<T> operator-(Array1D<T> lhs, const T& c) { lhs -= c; return lhs; }
template<typename T>
Array1D<T> operator*(Array1D<T> lhs, const T& c) { lhs *= c; return lhs; }
template<typename T>
Array1D<T> operator/(Array1D<T> lhs, const T& c) { lhs /= c; return lhs; }

template<typename T>
Array1D<T> operator*(const T& c, Array1D<T> rhs) { rhs *= c; return rhs; }

/* convenience wrappers mirroring <cmath> ---------------------------------- */

template<typename T> Array1D<T> cos(Array1D<T> v) { v.to_cos(); return v; }
template<typename T> Array1D<T> sin(Array1D<T> v) { v.to_sin(); return v; }

} // namespace subs