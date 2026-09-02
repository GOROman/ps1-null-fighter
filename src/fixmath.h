/* Small fixed point vector helpers (4096 == 1.0) shared by anim.c / ik.c */
#ifndef FIXMATH_H
#define FIXMATH_H

#include <stdint.h>
#include <psxgte.h>

#define FIX_ONE 4096

static inline uint32_t isqrt32(uint32_t n) {
	uint32_t res = 0, bit = 1u << 30;
	while (bit > n) bit >>= 2;
	while (bit) {
		if (n >= res + bit) { n -= res + bit; res = (res >> 1) + bit; }
		else res >>= 1;
		bit >>= 2;
	}
	return res;
}

static inline VECTOR vec(int32_t x, int32_t y, int32_t z) {
	VECTOR v;
	v.vx = x; v.vy = y; v.vz = z;
	return v;
}
static inline VECTOR vsub(VECTOR a, VECTOR b) { return vec(a.vx - b.vx, a.vy - b.vy, a.vz - b.vz); }
static inline VECTOR vadd(VECTOR a, VECTOR b) { return vec(a.vx + b.vx, a.vy + b.vy, a.vz + b.vz); }
static inline VECTOR vscale(VECTOR a, int32_t s) {   /* a * s / 4096 */
	return vec((a.vx * s) >> 12, (a.vy * s) >> 12, (a.vz * s) >> 12);
}
static inline int64_t vdot(VECTOR a, VECTOR b) {
	return (int64_t)a.vx * b.vx + (int64_t)a.vy * b.vy + (int64_t)a.vz * b.vz;
}
static inline int32_t vlen(VECTOR a) { return (int32_t)isqrt32((uint32_t)vdot(a, a)); }
/* unit vector, 4096 = 1.0 (zero vector stays zero) */
static inline VECTOR vnorm(VECTOR a) {
	int32_t l = vlen(a);
	if (l == 0) return vec(0, 0, 0);
	return vec((int32_t)(((int64_t)a.vx << 12) / l), (int32_t)(((int64_t)a.vy << 12) / l),
	           (int32_t)(((int64_t)a.vz << 12) / l));
}
/* cross product of two 4096-scaled vectors, result 4096-scaled */
static inline VECTOR vcross(VECTOR a, VECTOR b) {
	return vec((int32_t)(((int64_t)a.vy * b.vz - (int64_t)a.vz * b.vy) >> 12),
	           (int32_t)(((int64_t)a.vz * b.vx - (int64_t)a.vx * b.vz) >> 12),
	           (int32_t)(((int64_t)a.vx * b.vy - (int64_t)a.vy * b.vx) >> 12));
}

#endif
