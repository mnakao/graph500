/* Copyright (C) 2010 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include "splittable_mrg.h"
#include "mod_arith_64bit.h"
#define GENERATOR_USE_PACKED_EDGE_TYPE
#define FAST_64BIT_ARITHMETIC

/* Multiple recursive generator from L'Ecuyer, P., Blouin, F., and       */
/* Couture, R. 1993. A search for good multiple recursive random number  */
/* generators. ACM Trans. Model. Comput. Simul. 3, 2 (Apr. 1993), 87-98. */
/* DOI= http://doi.acm.org/10.1145/169702.169698 -- particular generator */
/* used is from table 3, entry for m = 2^31 - 1, k = 5 (same generator   */
/* is used in GNU Scientific Library).                                   */
/*                                                                       */
/* MRG state is 5 numbers mod 2^31 - 1, and there is a transition matrix */
/* A from one state to the next:                                         */
/*                                                                       */
/* A = [x 0 0 0 y]                                                       */
/*     [1 0 0 0 0]                                                       */
/*     [0 1 0 0 0]                                                       */
/*     [0 0 1 0 0]                                                       */
/*     [0 0 0 1 0]                                                       */
/* where x = 107374182 and y = 104480                                    */
/*                                                                       */
/* To do leapfrogging (applying multiple steps at once so that we can    */
/* create a tree of generators), we need powers of A.  These (from an    */
/* analysis with Maple) all look like:                                   */
/*                                                                       */
/* let a = x * s + t                                                     */
/*     b = x * a + u                                                     */
/*     c = x * b + v                                                     */
/*     d = x * c + w in                                                  */
/* A^n = [d   s*y a*y b*y c*y]                                           */
/*       [c   w   s*y a*y b*y]                                           */
/*       [b   v   w   s*y a*y]                                           */
/*       [a   u   v   w   s*y]                                           */
/*       [s   t   u   v   w  ]                                           */
/* for some values of s, t, u, v, and w                                  */
/* Note that A^n is determined by its bottom row (and x and y, which are */
/* fixed), and that it has a large part that is a Toeplitz matrix.  You  */
/* can multiply two A-like matrices by:                                  */
/* (defining a..d1 and a..d2 for the two matrices)                       */
/* s3 = s1 d2 + t1 c2 + u1 b2 + v1 a2 + w1 s2,                           */
/* t3 = s1 s2 y + t1 w2 + u1 v2 + v1 u2 + w1 t2,                         */
/* u3 = s1 a2 y + t1 s2 y + u1 w2 + v1 v2 + w1 u2,                       */
/* v3 = s1 b2 y + t1 a2 y + u1 s2 y + v1 w2 + w1 v2,                     */
/* w3 = s1 c2 y + t1 b2 y + u1 a2 y + v1 s2 y + w1 w2                    */

typedef struct mrg_transition_matrix {
  uint_fast32_t s, t, u, v, w;
  /* Cache for other parts of matrix (see mrg_update_cache function)     */
  uint_fast32_t a, b, c, d;
} mrg_transition_matrix;

/* r may alias st */
#ifdef __MTA__
#pragma mta inline
#endif
static void mrg_apply_transition(const mrg_transition_matrix* restrict mat, const mrg_state* restrict st, mrg_state* r) {
#ifdef __MTA__
  uint_fast64_t s = mat->s;
  uint_fast64_t t = mat->t;
  uint_fast64_t u = mat->u;
  uint_fast64_t v = mat->v;
  uint_fast64_t w = mat->w;
  uint_fast64_t z1 = st->z1;
  uint_fast64_t z2 = st->z2;
  uint_fast64_t z3 = st->z3;
  uint_fast64_t z4 = st->z4;
  uint_fast64_t z5 = st->z5;
  uint_fast64_t temp = s * z1 + t * z2 + u * z3 + v * z4;
  r->z5 = mod_down(mod_down_fast(temp) + w * z5);
  uint_fast64_t a = mod_down(107374182 * s + t);
  uint_fast64_t sy = mod_down(104480 * s);
  r->z4 = mod_down(mod_down_fast(a * z1 + u * z2 + v * z3) + w * z4 + sy * z5);
  uint_fast64_t b = mod_down(107374182 * a + u);
  uint_fast64_t ay = mod_down(104480 * a);
  r->z3 = mod_down(mod_down_fast(b * z1 + v * z2 + w * z3) + sy * z4 + ay * z5);
  uint_fast64_t c = mod_down(107374182 * b + v);
  uint_fast64_t by = mod_down(104480 * b);
  r->z2 = mod_down(mod_down_fast(c * z1 + w * z2 + sy * z3) + ay * z4 + by * z5);
  uint_fast64_t d = mod_down(107374182 * c + w);
  uint_fast64_t cy = mod_down(104480 * c);
  r->z1 = mod_down(mod_down_fast(d * z1 + sy * z2 + ay * z3) + by * z4 + cy * z5);
/* A^n = [d   s*y a*y b*y c*y]                                           */
/*       [c   w   s*y a*y b*y]                                           */
/*       [b   v   w   s*y a*y]                                           */
/*       [a   u   v   w   s*y]                                           */
/*       [s   t   u   v   w  ]                                           */
#else
  uint_fast32_t o1 = mod_mac_y(mod_mul(mat->d, st->z1), mod_mac4(0, mat->s, st->z2, mat->a, st->z3, mat->b, st->z4, mat->c, st->z5));
  uint_fast32_t o2 = mod_mac_y(mod_mac2(0, mat->c, st->z1, mat->w, st->z2), mod_mac3(0, mat->s, st->z3, mat->a, st->z4, mat->b, st->z5));
  uint_fast32_t o3 = mod_mac_y(mod_mac3(0, mat->b, st->z1, mat->v, st->z2, mat->w, st->z3), mod_mac2(0, mat->s, st->z4, mat->a, st->z5));
  uint_fast32_t o4 = mod_mac_y(mod_mac4(0, mat->a, st->z1, mat->u, st->z2, mat->v, st->z3, mat->w, st->z4), mod_mul(mat->s, st->z5));
  uint_fast32_t o5 = mod_mac2(mod_mac3(0, mat->s, st->z1, mat->t, st->z2, mat->u, st->z3), mat->v, st->z4, mat->w, st->z5);
  r->z1 = o1;
  r->z2 = o2;
  r->z3 = o3;
  r->z4 = o4;
  r->z5 = o5;
#endif
}

#ifdef __MTA__
#pragma mta inline
#endif
static void mrg_step(const mrg_transition_matrix* mat, mrg_state* state) {
  mrg_apply_transition(mat, state, state);
}

#ifdef __MTA__
#pragma mta inline
#endif
static void mrg_orig_step(mrg_state* state) { /* Use original A, not fully optimized yet */
  uint_fast32_t new_elt = mod_mac_y(mod_mul_x(state->z1), state->z5);
  state->z5 = state->z4;
  state->z4 = state->z3;
  state->z3 = state->z2;
  state->z2 = state->z1;
  state->z1 = new_elt;
}

#include "mrg_transitions.c"

void mrg_skip(mrg_state* state, uint_least64_t exponent_high, uint_least64_t exponent_middle, uint_least64_t exponent_low) {
  /* fprintf(stderr, "skip(%016" PRIXLEAST64 "%016" PRIXLEAST64 "%016" PRIXLEAST64 ")\n", exponent_high, exponent_middle, exponent_low); */
  int byte_index;
  for (byte_index = 0; exponent_low; ++byte_index, exponent_low >>= 8) {
    uint_least8_t val = (uint_least8_t)(exponent_low & 0xFF);
    if (val != 0) mrg_step(&mrg_skip_matrices[byte_index][val], state);
  }
  for (byte_index = 8; exponent_middle; ++byte_index, exponent_middle >>= 8) {
    uint_least8_t val = (uint_least8_t)(exponent_middle & 0xFF);
    if (val != 0) mrg_step(&mrg_skip_matrices[byte_index][val], state);
  }
  for (byte_index = 16; exponent_high; ++byte_index, exponent_high >>= 8) {
    uint_least8_t val = (uint_least8_t)(exponent_high & 0xFF);
    if (val != 0) mrg_step(&mrg_skip_matrices[byte_index][val], state);
  }
}

/* Returns integer value in [0, 2^31-1) using original transition matrix. */
uint_fast32_t mrg_get_uint_orig(mrg_state* state) {
  mrg_orig_step(state);
  return state->z1;
}

/* Returns real value in [0, 1) using original transition matrix. */
double mrg_get_double_orig(mrg_state* state) {
  return (double)mrg_get_uint_orig(state) * .000000000465661287524579692 /* (2^31 - 1)^(-1) */ +
         (double)mrg_get_uint_orig(state) * .0000000000000000002168404346990492787 /* (2^31 - 1)^(-2) */
    ;
}

void mrg_seed(mrg_state* st, const uint_fast32_t seed[5]) {
  st->z1 = seed[0];
  st->z2 = seed[1];
  st->z3 = seed[2];
  st->z4 = seed[3];
  st->z5 = seed[4];
}
