/* Two-bone IK on world matrices (no assumptions about the bones' local
 * axes): the chain is re-posed by minimal "rotate vector a onto b"
 * rotations (Rodrigues), which works for both the auto rig and the FBX
 * skeleton with arbitrary bind orientations.  All math in 4096 fixed point,
 * positions in PS1 units. */
#include <string.h>
#include <psxgte.h>
#include "ik.h"
#include "fixmath.h"

/* head may turn at most ~75 degrees away from its animated direction */
#define LOOK_COS   1060           /* cos 75 */
#define LOOK_SIN   3956           /* sin 75 */

static int16_t clamp16(int32_t v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t)v; }

/* rotation taking unit vector a onto unit vector b (both 4096 scaled) */
static void rot_between(VECTOR a, VECTOR b, MATRIX *r) {
	int32_t c = (int32_t)(vdot(a, b) >> 12);
	VECTOR v = vcross(a, b);
	if (c <= -4090) {
		/* 180 degrees about any axis perpendicular to a */
		VECTOR ref = (a.vx > -2000 && a.vx < 2000) ? vec(4096, 0, 0) : vec(0, 4096, 0);
		VECTOR p = vnorm(vcross(a, ref));
		int32_t px = p.vx, py = p.vy, pz = p.vz;
		r->m[0][0] = clamp16(((2 * px * px) >> 12) - 4096);
		r->m[0][1] = clamp16((2 * px * py) >> 12);
		r->m[0][2] = clamp16((2 * px * pz) >> 12);
		r->m[1][0] = r->m[0][1];
		r->m[1][1] = clamp16(((2 * py * py) >> 12) - 4096);
		r->m[1][2] = clamp16((2 * py * pz) >> 12);
		r->m[2][0] = r->m[0][2];
		r->m[2][1] = r->m[1][2];
		r->m[2][2] = clamp16(((2 * pz * pz) >> 12) - 4096);
		return;
	}
	/* R = I + K + K^2 / (1 + c),  K = [v]x */
	int32_t k = (4096 * 4096) / (4096 + c);
	int32_t K[3][3] = { { 0, -v.vz, v.vy }, { v.vz, 0, -v.vx }, { -v.vy, v.vx, 0 } };
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			int32_t k2 = (K[i][0] * K[0][j] + K[i][1] * K[1][j] + K[i][2] * K[2][j]) >> 12;
			int32_t e = (i == j ? 4096 : 0) + K[i][j] + ((k2 * k) >> 12);
			r->m[i][j] = clamp16(e);
		}
	r->t[0] = r->t[1] = r->t[2] = 0;
}

static VECTOR mat_t(const MATRIX *m) { return vec(m->t[0], m->t[1], m->t[2]); }

/* rotate a world matrix by r about pivot */
static void rotate_world(MATRIX *w, const MATRIX *r, VECTOR pivot) {
	MATRIX out;
	VECTOR d = vsub(mat_t(w), pivot), rd;
	MulMatrix0((MATRIX *)r, w, &out);
	ApplyMatrixLV((MATRIX *)r, &d, &rd);
	VECTOR t = vadd(pivot, rd);
	out.t[0] = t.vx; out.t[1] = t.vy; out.t[2] = t.vz;
	*w = out;
}

static VECTOR effector_pos(const Pose *pose, const ModelIKChain *c) {
	const MATRIX *w = &pose->world[c->lower];
	VECTOR e = vec(c->end_local[0], c->end_local[1], c->end_local[2]), r;
	ApplyMatrixLV((MATRIX *)w, &e, &r);
	return vadd(r, mat_t(w));
}

void ik_enter(IKState *s, const Model *m, Pose *pose) {
	memset(s, 0, sizeof(*s));
	if (!m->ik) return;
	s->active = 1;
	pose->hip_bone = m->ik->hip;
	pose->hip_offset = vec(0, 0, 0);
	pose->playing = 0;
	pose_eval(pose);
	for (int i = 0; i < IK_CHAINS; i++) {
		const ModelIKChain *c = &m->ik->chains[i];
		if (c->upper < 0) continue;
		s->has[i] = 1;
		s->target[i] = effector_pos(pose, c);
	}
}

void ik_leave(IKState *s, Pose *pose) {
	s->active = 0;
	pose->hip_bone = -1;
	pose->hip_offset = vec(0, 0, 0);
	pose->playing = 1;
}

/* pole direction expressed in model space, then made orthogonal to u */
static VECTOR bend_dir(VECTOR u, VECTOR pole) {
	int32_t d = (int32_t)(vdot(pole, u) >> 12);
	VECTOR v = vsub(pole, vscale(u, d));
	if (vlen(v) < 200) {
		VECTOR ref = (u.vy > -2000 && u.vy < 2000) ? vec(0, 4096, 0) : vec(4096, 0, 0);
		v = vcross(u, ref);
	}
	return vnorm(v);
}

static void solve_chain(Pose *pose, const ModelIKChain *c, VECTOR target, uint8_t *modified) {
	MATRIX *wu = &pose->world[c->upper];
	MATRIX *wl = &pose->world[c->lower];
	VECTOR A = mat_t(wu), B = mat_t(wl), C = effector_pos(pose, c);
	int32_t L1 = vlen(vsub(B, A)), L2 = vlen(vsub(C, B));
	if (L1 == 0 || L2 == 0) return;

	VECTOR AT = vsub(target, A);
	int32_t d = vlen(AT);
	int32_t dmax = L1 + L2 - 4, dmin = (L1 > L2 ? L1 - L2 : L2 - L1) + 4;
	if (d < 1) { AT = vec(0, 4096, 0); d = 4096; }
	if (d > dmax) { AT = vec((int32_t)((int64_t)AT.vx * dmax / d), (int32_t)((int64_t)AT.vy * dmax / d), (int32_t)((int64_t)AT.vz * dmax / d)); d = dmax; }
	if (d < dmin) { AT = vec((int32_t)((int64_t)AT.vx * dmin / d), (int32_t)((int64_t)AT.vy * dmin / d), (int32_t)((int64_t)AT.vz * dmin / d)); d = dmin; }
	VECTOR T = vadd(A, AT);

	/* law of cosines: angle at A between (A->B') and (A->T) */
	int64_t num = (int64_t)L1 * L1 + (int64_t)d * d - (int64_t)L2 * L2;
	int64_t den = 2 * (int64_t)L1 * d;
	int32_t cosa = (int32_t)((num << 12) / den);
	if (cosa > 4096) cosa = 4096;
	if (cosa < -4096) cosa = -4096;
	int32_t sina = (int32_t)isqrt32((uint32_t)(4096 * 4096 - cosa * cosa));

	VECTOR u = vnorm(AT);
	VECTOR v = bend_dir(u, vec(c->pole[0], c->pole[1], c->pole[2]));
	VECTOR dir = vadd(vscale(u, cosa), vscale(v, sina));           /* 4096 scaled */
	VECTOR Bn = vadd(A, vscale(dir, L1));

	/* upper: rotate (A->B) onto (A->B') about A */
	MATRIX r1;
	rot_between(vnorm(vsub(B, A)), vnorm(vsub(Bn, A)), &r1);
	rotate_world(wu, &r1, A);
	/* lower: inherit r1 (moves it to B'), then align (B'->C) onto (B'->T) */
	rotate_world(wl, &r1, A);
	VECTOR C1 = effector_pos(pose, c);
	MATRIX r2;
	rot_between(vnorm(vsub(C1, Bn)), vnorm(vsub(T, Bn)), &r2);
	rotate_world(wl, &r2, Bn);
	modified[c->upper] = modified[c->lower] = 1;
	if (c->end >= 0) {
		MATRIX *we = &pose->world[c->end];
		rotate_world(we, &r1, A);
		rotate_world(we, &r2, Bn);
		modified[c->end] = 1;
	}
}

static void look_at(Pose *pose, const ModelIK *ik, const Camera *cam, uint8_t *modified) {
	if (ik->head < 0) return;
	MATRIX *wh = &pose->world[ik->head];
	/* camera position in model space: p = -R^T t */
	const MATRIX *vw = &cam->view;
	VECTOR cp = vec(-((vw->m[0][0] * vw->t[0] + vw->m[1][0] * vw->t[1] + vw->m[2][0] * vw->t[2]) >> 12),
	                -((vw->m[0][1] * vw->t[0] + vw->m[1][1] * vw->t[1] + vw->m[2][1] * vw->t[2]) >> 12),
	                -((vw->m[0][2] * vw->t[0] + vw->m[1][2] * vw->t[1] + vw->m[2][2] * vw->t[2]) >> 12));
	VECTOR fwd_local = vec(ik->head_fwd[0], ik->head_fwd[1], ik->head_fwd[2]), fwd;
	ApplyMatrixLV(wh, &fwd_local, &fwd);
	VECTOR fc = vnorm(fwd);
	VECTOR fd = vnorm(vsub(cp, mat_t(wh)));
	if (fc.vx == 0 && fc.vy == 0 && fc.vz == 0) return;
	int32_t c = (int32_t)(vdot(fc, fd) >> 12);
	if (c < LOOK_COS) {
		/* clamp: rotate only up to the limit angle towards the camera */
		VECTOR w = vnorm(vsub(fd, vscale(fc, c)));
		fd = vadd(vscale(fc, LOOK_COS), vscale(w, LOOK_SIN));
	}
	MATRIX r;
	rot_between(fc, fd, &r);
	rotate_world(wh, &r, mat_t(wh));
	modified[ik->head] = 1;
}

void ik_apply(IKState *s, const Model *m, Pose *pose, const Camera *cam) {
	if (!s->active || !m->ik) return;
	uint8_t modified[MAX_BONES];
	memset(modified, 0, sizeof(modified));
	for (int i = 0; i < IK_CHAINS; i++)
		if (s->has[i])
			solve_chain(pose, &m->ik->chains[i], s->target[i], modified);
	look_at(pose, m->ik, cam, modified);
	pose_refresh(pose, modified);
}
