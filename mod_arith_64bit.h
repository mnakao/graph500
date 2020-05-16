#ifndef MOD_ARITH_64BIT_H
#define MOD_ARITH_64BIT_H

#include <stdint.h>
#include <assert.h>

static inline uint_fast32_t mod_mul(uint_fast32_t a, uint_fast32_t b) {
  assert (a <= 0x7FFFFFFE);
  assert (b <= 0x7FFFFFFE);
  return (uint_fast32_t)((uint_fast64_t)a * b % 0x7FFFFFFF);
}

static inline uint_fast32_t mod_mac(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b) {
  assert (sum <= 0x7FFFFFFE);
  assert (a <= 0x7FFFFFFE);
  assert (b <= 0x7FFFFFFE);
  return (uint_fast32_t)(((uint_fast64_t)a * b + sum) % 0x7FFFFFFF);
}

static inline uint_fast32_t mod_mac2(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b, uint_fast32_t c, uint_fast32_t d) {
  assert (sum <= 0x7FFFFFFE);
  assert (a <= 0x7FFFFFFE);
  assert (b <= 0x7FFFFFFE);
  assert (c <= 0x7FFFFFFE);
  assert (d <= 0x7FFFFFFE);
  return (uint_fast32_t)(((uint_fast64_t)a * b + (uint_fast64_t)c * d + sum) % 0x7FFFFFFF);
}

static inline uint_fast32_t mod_mac3(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b, uint_fast32_t c, uint_fast32_t d, uint_fast32_t e, uint_fast32_t f) {
  assert (sum <= 0x7FFFFFFE);
  assert (a <= 0x7FFFFFFE);
  assert (b <= 0x7FFFFFFE);
  assert (c <= 0x7FFFFFFE);
  assert (d <= 0x7FFFFFFE);
  assert (e <= 0x7FFFFFFE);
  assert (f <= 0x7FFFFFFE);
  return (uint_fast32_t)(((uint_fast64_t)a * b + (uint_fast64_t)c * d + (uint_fast64_t)e * f + sum) % 0x7FFFFFFF);
}

static inline uint_fast32_t mod_mac4(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b, uint_fast32_t c, uint_fast32_t d, uint_fast32_t e, uint_fast32_t f, uint_fast32_t g, uint_fast32_t h) {
  assert (sum <= 0x7FFFFFFE);
  assert (a <= 0x7FFFFFFE);
  assert (b <= 0x7FFFFFFE);
  assert (c <= 0x7FFFFFFE);
  assert (d <= 0x7FFFFFFE);
  assert (e <= 0x7FFFFFFE);
  assert (f <= 0x7FFFFFFE);
  assert (g <= 0x7FFFFFFE);
  assert (h <= 0x7FFFFFFE);
  return (uint_fast32_t)(((uint_fast64_t)a * b + (uint_fast64_t)c * d + (uint_fast64_t)e * f + (uint_fast64_t)g * h + sum) % 0x7FFFFFFF);
}

static inline uint_fast32_t mod_mul_x(uint_fast32_t a) {
  return mod_mul(a, 107374182);
}

static inline uint_fast32_t mod_mac_y(uint_fast32_t sum, uint_fast32_t a) {
  return mod_mac(sum, a, 104480);
}

#endif /* MOD_ARITH_64BIT_H */
