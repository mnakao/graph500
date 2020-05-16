#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <stdint.h>
#include <algorithm>
#include <functional>
using std::ptrdiff_t;

#ifdef __cplusplus
#define restrict __restrict__
#endif

template <typename T> struct MpiTypeOf { };

#define NLEADING_ZERO_BITS __builtin_clz
#define NLEADING_ZERO_BITSL __builtin_clzl
#define NLEADING_ZERO_BITSLL __builtin_clzll

inline int get_msb_index(int64_t value) {
	return (sizeof(value)*8-1) - INT64_C(NLEADING_ZERO_BITS)(value);
}

#undef NLEADING_ZERO_BITS
#undef NLEADING_ZERO_BITSL
#undef NLEADING_ZERO_BITSLL

void* xMPI_Alloc_mem(size_t nbytes);
void* cache_aligned_xcalloc(const size_t size);
void* cache_aligned_xmalloc(const size_t size);
void* page_aligned_xcalloc(const size_t size);
void* page_aligned_xmalloc(const size_t size);

template <typename T1, typename T2>
class pointer_pair_value
{
public:
	T1 v1;
	T2 v2;
	pointer_pair_value(T1& v1_, T2& v2_)
		: v1(v1_)
		, v2(v2_)
	{ }
	operator T1 () const { return v1; }
	pointer_pair_value& operator=(const pointer_pair_value& o) { v1 = o.v1; v2 = o.v2; return *this; }
};

template <typename T1, typename T2>
class pointer_pair_reference
{
public:
	T1& v1;
	T2& v2;
	pointer_pair_reference(T1& v1_, T2& v2_)
		: v1(v1_)
		, v2(v2_)
	{ }
	operator T1 () const { return v1; }
	operator pointer_pair_value<T1, T2> () const { return pointer_pair_value<T1, T2>(v1, v2); }
	pointer_pair_reference& operator=(const pointer_pair_reference& o) { v1 = o.v1; v2 = o.v2; return *this; }
	pointer_pair_reference& operator=(const pointer_pair_value<T1, T2>& o) { v1 = o.v1; v2 = o.v2; return *this; }
};

template <typename T1, typename T2>
void swap(pointer_pair_reference<T1, T2> r1, pointer_pair_reference<T1, T2> r2)
{
	pointer_pair_value<T1, T2> tmp = r1;
	r1 = r2;
	r2 = tmp;
}

template <typename T1, typename T2>
class pointer_pair_iterator
	: public std::iterator<std::random_access_iterator_tag,
		pointer_pair_value<T1, T2>,
		ptrdiff_t,
		pointer_pair_value<T1, T2>*,
		pointer_pair_reference<T1, T2> >
{
public:
	T1* first;
	T2* second;

	typedef ptrdiff_t difference_type;
	typedef pointer_pair_reference<T1, T2> reference;

	pointer_pair_iterator(T1* v1, T2* v2) : first(v1) , second(v2) { }

	// Can be default-constructed
	pointer_pair_iterator() { }
	// Accepts equality/inequality comparisons
	bool operator==(const pointer_pair_iterator& ot) { return first == ot.first; }
	bool operator!=(const pointer_pair_iterator& ot) { return first != ot.first; }
	// Can be dereferenced
	reference operator*(){ return reference(*first, *second); }
	// Can be incremented and decremented
	pointer_pair_iterator operator++(int) { pointer_pair_iterator old(*this); ++first; ++second; return old; }
	pointer_pair_iterator operator--(int) { pointer_pair_iterator old(*this); --first; --second; return old; }
	pointer_pair_iterator& operator++() { ++first; ++second; return *this; }
	pointer_pair_iterator& operator--() { --first; --second; return *this; }
	// Supports arithmetic operators + and - between an iterator and an integer value, or subtracting an iterator from another
	pointer_pair_iterator operator+(const difference_type n) { pointer_pair_iterator t(first+n, second+n); return t; }
	pointer_pair_iterator operator-(const difference_type n) { pointer_pair_iterator t(first-n, second-n); return t; }
	size_t operator-(const pointer_pair_iterator& o) { return first - o.first; }
	// Supports inequality comparisons (<, >, <= and >=) between iterators
	bool operator<(const pointer_pair_iterator& o) { return first < o.first; }
	bool operator>(const pointer_pair_iterator& o) { return first > o.first; }
	bool operator<=(const pointer_pair_iterator& o) { return first <= o.first; }
	bool operator>=(const pointer_pair_iterator& o) { return first >= o.first; }
	// Supports compound assinment operations += and -=
	pointer_pair_iterator& operator+=(const difference_type n) { first += n; second += n; return *this; }
	pointer_pair_iterator& operator-=(const difference_type n) { first -= n; second -= n; return *this; }
	// Supports offset dereference operator ([])
	reference operator[](const difference_type n) { return reference(first[n], second[n]); }
};
template<typename IterKey, typename IterValue>
void sort2(IterKey* begin_key, IterValue* begin_value, size_t count)
{
	pointer_pair_iterator<IterKey, IterValue>
		begin(begin_key, begin_value), end(begin_key + count, begin_value + count);
	std::sort(begin, end);
}

template<typename IterKey, typename IterValue, typename Compare>
void sort2(IterKey* begin_key, IterValue* begin_value, size_t count, Compare comp)
{
	pointer_pair_iterator<IterKey, IterValue>
		begin(begin_key, begin_value), end(begin_key + count, begin_value + count);
	std::sort(begin, end, comp);
}

struct SeparatedId {
	uint64_t value;

	explicit SeparatedId(uint64_t v) : value(v) { }
	SeparatedId(int high, uint64_t low, int lgl)
		: value((uint64_t(high) << lgl) | low) { }
	uint64_t raw() const { return value; }
	uint64_t compact(int lgl, int64_t L) const { return high(lgl) * L + low(lgl); }
	int high(int lgl) const { return value >> lgl; }
	uint64_t low(int lgl) const { return value & ((uint64_t(1) << lgl) - 1); }
	int64_t swaplow(int mid, int lgl) {
		int64_t low_v = value >> (mid + lgl);
		int64_t mid_v = (value >> lgl) & ((1 << mid) - 1);
		return (mid_v << lgl) | low_v;
	}
};

#endif /* UTILS_HPP_ */
