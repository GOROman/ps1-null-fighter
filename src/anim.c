/* Skeletal pose evaluation: quaternion nlerp between sampled frames, then
 * fixed point quaternion -> matrix and parent chaining with the GTE
 * accelerated MulMatrix0 / ApplyMatrixLV helpers. */
#include <string.h>
#include <psxgte.h>
#include "anim.h"
#include "fixmath.h"

void pose_init(Pose *p, const Model *m, int anim) {
	memset(p, 0, sizeof(*p));
	p->model = m;
	p->playing = 1;
	p->hip_bone = -1;
	pose_set_anim(p, anim);
}

void pose_set_anim(Pose *p, int anim) {
	int n = p->model->hdr->nanims;
	if (n == 0) return;
	while (anim < 0) anim += n;
	p->anim = anim % n;
	p->frame = 0;
	p->subframe = 0;
	p->loops = 0;
}

void pose_step(Pose *p, int display_hz) {
	const ModelAnim *a = &p->model->anims[p->anim];
	if (!p->playing || a->nframes <= 1) return;
	/* subframe advances fps/display_hz per display frame (in 1/256 units) */
	p->subframe += (a->fps * 256) / display_hz;
	while (p->subframe >= 256) {
		p->subframe -= 256;
		p->frame++;
		if (p->frame >= a->nframes) {
			p->frame = 0;
			p->loops++;
		}
	}
}

/* q (any scale) -> 3x3 rotation, 4096 = 1.0 */
static void quat_to_matrix(int32_t qx, int32_t qy, int32_t qz, int32_t qw, MATRIX *m) {
	/* normalize to 4096 */
	uint32_t n = isqrt32((uint32_t)(qx * qx + qy * qy + qz * qz + qw * qw));
	if (n == 0) n = 1;
	int32_t x = (qx << 12) / (int32_t)n;
	int32_t y = (qy << 12) / (int32_t)n;
	int32_t z = (qz << 12) / (int32_t)n;
	int32_t w = (qw << 12) / (int32_t)n;
	int32_t xx = (x * x) >> 12, yy = (y * y) >> 12, zz = (z * z) >> 12;
	int32_t xy = (x * y) >> 12, xz = (x * z) >> 12, yz = (y * z) >> 12;
	int32_t wx = (w * x) >> 12, wy = (w * y) >> 12, wz = (w * z) >> 12;
	m->m[0][0] = 4096 - 2 * (yy + zz);
	m->m[0][1] = 2 * (xy - wz);
	m->m[0][2] = 2 * (xz + wy);
	m->m[1][0] = 2 * (xy + wz);
	m->m[1][1] = 4096 - 2 * (xx + zz);
	m->m[1][2] = 2 * (yz - wx);
	m->m[2][0] = 2 * (xz - wy);
	m->m[2][1] = 2 * (yz + wx);
	m->m[2][2] = 4096 - 2 * (xx + yy);
}

void pose_compose(const MATRIX *pw, const MATRIX *local, MATRIX *out) {
	VECTOR lt = vec(local->t[0], local->t[1], local->t[2]);
	VECTOR wt;
	MulMatrix0((MATRIX *)pw, (MATRIX *)local, out);
	ApplyMatrixLV((MATRIX *)pw, &lt, &wt);
	out->t[0] = wt.vx + pw->t[0];
	out->t[1] = wt.vy + pw->t[1];
	out->t[2] = wt.vz + pw->t[2];
}

void pose_eval(Pose *p) {
	const Model *m = p->model;
	const ModelAnim *a = &m->anims[p->anim];
	int f0 = p->frame;
	int f1 = (f0 + 1 < a->nframes) ? f0 + 1 : 0;
	int t = p->subframe;   /* 0..255 */
	const ModelQuat  *q0 = model_frame_quats(m, p->anim, f0);
	const ModelQuat  *q1 = model_frame_quats(m, p->anim, f1);
	const ModelTrans *t0 = model_frame_trans(m, p->anim, f0);
	const ModelTrans *t1 = model_frame_trans(m, p->anim, f1);
	int nb = m->hdr->nbones;

	for (int b = 0; b < nb; b++) {
		const ModelBone *bone = &m->bones[b];
		MATRIX *local = &p->local[b];
		int32_t ax = q0[b].x, ay = q0[b].y, az = q0[b].z, aw = q0[b].w;
		int32_t bx = q1[b].x, by = q1[b].y, bz = q1[b].z, bw = q1[b].w;
		/* shortest path */
		if (ax * bx + ay * by + az * bz + aw * bw < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; }
		int32_t qx = ax + (((bx - ax) * t) >> 8);
		int32_t qy = ay + (((by - ay) * t) >> 8);
		int32_t qz = az + (((bz - az) * t) >> 8);
		int32_t qw = aw + (((bw - aw) * t) >> 8);
		quat_to_matrix(qx, qy, qz, qw, local);

		if (bone->flags & BONE_FLAG_TRANS) {
			const ModelTrans *s0 = &t0[bone->trans_slot];
			const ModelTrans *s1 = &t1[bone->trans_slot];
			local->t[0] = s0->x + (((s1->x - s0->x) * t) >> 8);
			local->t[1] = s0->y + (((s1->y - s0->y) * t) >> 8);
			local->t[2] = s0->z + (((s1->z - s0->z) * t) >> 8);
		} else {
			local->t[0] = bone->tx; local->t[1] = bone->ty; local->t[2] = bone->tz;
		}
		if (b == p->hip_bone) {
			local->t[0] += p->hip_offset.vx;
			local->t[1] += p->hip_offset.vy;
			local->t[2] += p->hip_offset.vz;
		}

		MATRIX *out = &p->world[b];
		if (bone->parent < 0)
			*out = *local;
		else
			pose_compose(&p->world[bone->parent], local, out);
	}
}

void pose_refresh(Pose *p, const uint8_t *modified) {
	const Model *m = p->model;
	int nb = m->hdr->nbones;
	uint8_t dirty[MAX_BONES];
	for (int b = 0; b < nb; b++) {
		int parent = m->bones[b].parent;
		dirty[b] = modified[b];
		if (!modified[b] && parent >= 0 && dirty[parent]) {
			pose_compose(&p->world[parent], &p->local[b], &p->world[b]);
			dirty[b] = 1;
		}
	}
}
