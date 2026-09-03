/* Rigid skinned mesh renderer.
 *
 * Every vertex lives in the local space of exactly one bone, so for each
 * bone we load (view * bone_world) into the GTE once and push its vertices
 * through RTPT in batches of three, caching screen XY, Z and the lit colour.
 * Triangles are then assembled from the cache (NCLIP for backface culling,
 * average Z for ordering table depth).  Flat shading lights the stored face
 * normal instead of the vertex normals.
 *
 * Pass 2 is the hot loop (PS1 has no data cache, every load hits RAM), so
 * it avoids the SDK's store-to-memory GTE macros with "memory" clobbers
 * (which force the compiler to reload everything) and writes the primitive
 * in whole 32-bit words. */
#include <psxgpu.h>
#include <psxgte.h>
#include <inline_c.h>
#include "render.h"
#include "prof.h"

static VCache vcache[MAX_VERTS];

const VCache *render_get_cache(void) { return vcache; }

/* read NCLIP's result (MAC0) into a register without a memory clobber */
#define gte_stopz_r(p) __asm__ volatile("mfc2 %0, $24" : "=r"(p))
/* GTE result stores without the SDK's "memory" clobber (nothing reads the
 * cache inside pass 1, so the compiler may keep its pointers in registers) */
#define STSXY3(a, b, c) __asm__ volatile("swc2 $12, 0(%0); swc2 $13, 0(%1); swc2 $14, 0(%2)" :: "r"(a), "r"(b), "r"(c))
#define STSZ3(a, b, c)  __asm__ volatile("swc2 $17, 0(%0); swc2 $18, 0(%1); swc2 $19, 0(%2)" :: "r"(a), "r"(b), "r"(c))
#define STRGB3(a, b, c) __asm__ volatile("swc2 $20, 0(%0); swc2 $21, 0(%1); swc2 $22, 0(%2)" :: "r"(a), "r"(b), "r"(c))
#define STSXY(a)        __asm__ volatile("swc2 $14, 0(%0)" :: "r"(a))
#define STSZ(a)         __asm__ volatile("swc2 $19, 0(%0)" :: "r"(a))
#define STRGB(a)        __asm__ volatile("swc2 $22, 0(%0)" :: "r"(a))

/* light colour matrix: one directional light, ~0.42 intensity so that the
 * result stays near 128 (= neutral texture modulation) with some headroom */
static MATRIX color_mtx = {
	{ { 1700, 0, 0 },
	  { 1700, 0, 0 },
	  { 1700, 0, 0 } },
	{ 0, 0, 0 }
};

/* light direction in camera space (from upper-left-front of the viewer) */
static MATRIX light_mtx = {
	{ { -1800, -2600, -2400 },
	  { 0, 0, 0 },
	  { 0, 0, 0 } },
	{ 0, 0, 0 }
};

void renderer_init(Renderer *r, const uint32_t *const tims[MAX_TEX]) {
	r->ntex = 0;
	for (int i = 0; i < MAX_TEX; i++) {
		TIM_IMAGE tim;
		if (!tims[i]) break;
		GetTimInfo(tims[i], &tim);
		if (tim.mode & 0x8)
			LoadImage(tim.crect, tim.caddr);
		LoadImage(tim.prect, tim.paddr);
		r->tpage[i] = getTPage(tim.mode & 3, 0, tim.prect->x, tim.prect->y);
		r->clut[i]  = getClut(tim.crect->x, tim.crect->y);
		r->ntex++;
	}
	DrawSync(0);
	/* a model referencing a missing slot falls back to slot 0 */
	for (int i = r->ntex; i < MAX_TEX; i++) {
		r->tpage[i] = r->tpage[0];
		r->clut[i]  = r->clut[0];
	}
	r->shading = SHADE_GOURAUD;
	r->tris_drawn = 0;
	r->cut = 0;
	r->bone_mask = 0;
	r->unlit_rgb = 0x00808080;
	r->otz_shift = 2 + OTZ_SHIFT;
	r->otz_base = 0;
	r->otz_limit = OT_LEN;

	gte_SetBackColor(52, 52, 56);
	gte_SetColorMatrix(&color_mtx);
}

/* view * bone: rotation via GTE, translation = view.m * bone.t + view.t */
static void concat_view(const MATRIX *view, const MATRIX *bone, MATRIX *out) {
	VECTOR bt = { bone->t[0], bone->t[1], bone->t[2] };
	VECTOR wt;
	MulMatrix0((MATRIX *)view, (MATRIX *)bone, out);
	ApplyMatrixLV((MATRIX *)view, &bt, &wt);
	out->t[0] = wt.vx + view->t[0];
	out->t[1] = wt.vy + view->t[1];
	out->t[2] = wt.vz + view->t[2];
}

char *render_model(Renderer *r, const Model *m, const Pose *pose, const Camera *cam,
                   uint32_t *ot, char *nextpri) {
	const int nb = m->hdr->nbones;
	const ModelVert *verts = m->verts;
	MATRIX mv, lm;
	int drawn = 0;

	/* ---- pass 1: transform + light vertices, bone by bone ---------------- */
	for (int b = 0; b < nb; b++) {
		const ModelBone *bone = &m->bones[b];
		if (bone->vert_count == 0 || (r->bone_mask && !r->bone_mask[b]))
			continue;
		concat_view(&cam->view, &pose->world[b], &mv);
		/* MulMatrix0 loads its first operand into the GTE rotation matrix,
		 * so build the light matrix BEFORE setting the transform for RTPT. */
		MulMatrix0(&light_mtx, &mv, &lm);
		gte_SetLightMatrix(&lm);
		gte_SetRotMatrix(&mv);
		gte_SetTransMatrix(&mv);

		int i = bone->vert_start;
		int end = i + bone->vert_count;
		const ModelVert *v = &verts[i];
		VCache *c = &vcache[i];
		if (r->shading == SHADE_GOURAUD) {
			for (; i + 3 <= end; i += 3, v += 3, c += 3) {
				gte_ldv3(&v[0].x, &v[1].x, &v[2].x);
				gte_rtpt();
				STSXY3(&c[0].sxy, &c[1].sxy, &c[2].sxy);
				STSZ3(&c[0].sz, &c[1].sz, &c[2].sz);
				gte_ldv3(&v[0].nx, &v[1].nx, &v[2].nx);
				gte_nct();
				STRGB3(&c[0].rgb, &c[1].rgb, &c[2].rgb);
			}
			for (; i < end; i++, v++, c++) {
				gte_ldv0(&v->x);
				gte_rtps();
				STSXY(&c->sxy);
				STSZ(&c->sz);
				gte_ldv0(&v->nx);
				gte_ncs();
				STRGB(&c->rgb);
			}
		} else {
			for (; i + 3 <= end; i += 3, v += 3, c += 3) {
				gte_ldv3(&v[0].x, &v[1].x, &v[2].x);
				gte_rtpt();
				STSXY3(&c[0].sxy, &c[1].sxy, &c[2].sxy);
				STSZ3(&c[0].sz, &c[1].sz, &c[2].sz);
			}
			for (; i < end; i++, v++, c++) {
				gte_ldv0(&v->x);
				gte_rtps();
				STSXY(&c->sxy);
				STSZ(&c->sz);
			}
		}
	}
	__asm__ volatile("" ::: "memory");          /* pass 2 reads what the GTE stored */
	prof_mark(PROF_VERTS);

	/* ---- pass 2: build primitives --------------------------------------- */
	const ModelTri *tri = m->tris;
	const ModelTri *tri_end = tri + m->hdr->ntris;
	static const uint8_t no_cut[MAX_VERTS];
	const uint8_t *cut = r->cut ? r->cut : no_cut;
	const uint8_t *mask = r->bone_mask;
	const int otz_shift = r->otz_shift, otz_base = r->otz_base, otz_limit = r->otz_limit;
	uint32_t clut_hi[MAX_TEX], tpage_hi[MAX_TEX];
	for (int i = 0; i < MAX_TEX; i++) {
		clut_hi[i]  = (uint32_t)r->clut[i] << 16;
		tpage_hi[i] = (uint32_t)r->tpage[i] << 16;
	}

	if (r->shading == SHADE_GOURAUD) {
		POLY_GT3 *pri = (POLY_GT3 *)nextpri;
		for (; tri < tri_end; tri++) {
			if (cut[tri - m->tris] || (mask && !mask[tri->bone]))
				continue;
			const int i0 = tri->i0;
			int i1 = tri->i1, i2 = tri->i2;
			/* with a bone mask every vertex must have been transformed too */
			if (mask && (!mask[verts[i0].bone] || !mask[verts[i1].bone] || !mask[verts[i2].bone]))
				continue;
			const VCache *c0 = &vcache[i0], *c1 = &vcache[i1], *c2 = &vcache[i2];
			int p;
			gte_ldsxy3(c0->sxy, c1->sxy, c2->sxy);
			gte_nclip();
			gte_stopz_r(p);
			if (p <= 0) {
				if (!(tri->flags & TRI_FLAG_DOUBLE_SIDED))
					continue;
				const VCache *t = c1; c1 = c2; c2 = t;      /* back face: reverse the winding */
				const int ti = i1; i1 = i2; i2 = ti;
			}
			const int sz0 = c0->sz, sz1 = c1->sz, sz2 = c2->sz;
			if (!sz0 || !sz1 || !sz2)               /* vertex at/behind the camera */
				continue;
			int otz = ((sz0 + sz1 + sz2) >> otz_shift) + otz_base;   /* ~avg/3, cheap */
			if (otz >= otz_limit)
				continue;
			*(uint32_t *)&pri->r0 = c0->rgb;
			*(uint32_t *)&pri->r1 = c1->rgb;
			*(uint32_t *)&pri->r2 = c2->rgb;
			setPolyGT3(pri);                        /* len + code (r0's 4th byte) */
			*(uint32_t *)&pri->x0 = c0->sxy;
			*(uint32_t *)&pri->x1 = c1->sxy;
			*(uint32_t *)&pri->x2 = c2->sxy;
			*(uint32_t *)&pri->u0 = verts[i0].uv | clut_hi[tri->tex];
			*(uint32_t *)&pri->u1 = verts[i1].uv | tpage_hi[tri->tex];
			*(uint32_t *)&pri->u2 = verts[i2].uv;
			addPrim(ot + otz, pri);
			pri++;
			drawn++;
		}
		nextpri = (char *)pri;
	} else {
		POLY_FT3 *pri = (POLY_FT3 *)nextpri;
		int cur_bone = -1;
		const int unlit = r->shading == SHADE_NONE;
		const uint32_t unlit_rgb = r->unlit_rgb & 0x00ffffff;
		for (; tri < tri_end; tri++) {
			if (cut[tri - m->tris] || (mask && !mask[tri->bone]))
				continue;
			const int i0 = tri->i0;
			int i1 = tri->i1, i2 = tri->i2;
			/* with a bone mask every vertex must have been transformed too */
			if (mask && (!mask[verts[i0].bone] || !mask[verts[i1].bone] || !mask[verts[i2].bone]))
				continue;
			const VCache *c0 = &vcache[i0], *c1 = &vcache[i1], *c2 = &vcache[i2];
			int p;
			gte_ldsxy3(c0->sxy, c1->sxy, c2->sxy);
			gte_nclip();
			gte_stopz_r(p);
			if (p <= 0) {
				if (!(tri->flags & TRI_FLAG_DOUBLE_SIDED))
					continue;
				const VCache *t = c1; c1 = c2; c2 = t;
				const int ti = i1; i1 = i2; i2 = ti;
			}
			const int sz0 = c0->sz, sz1 = c1->sz, sz2 = c2->sz;
			if (!sz0 || !sz1 || !sz2)
				continue;
			int otz = ((sz0 + sz1 + sz2) >> otz_shift) + otz_base;
			if (otz >= otz_limit)
				continue;
			if (unlit) {
				*(uint32_t *)&pri->r0 = unlit_rgb;     /* texture modulated by a constant */
			} else {
				/* triangles are sorted by bone: reload the light matrix only
				 * when the bone changes */
				if (tri->bone != cur_bone) {
					cur_bone = tri->bone;
					concat_view(&cam->view, &pose->world[cur_bone], &mv);
					MulMatrix0(&light_mtx, &mv, &lm);
					gte_SetLightMatrix(&lm);
				}
				gte_ldv0(&tri->nx);
				gte_ncs();
				gte_strgb(&pri->r0);
			}
			setPolyFT3(pri);
			*(uint32_t *)&pri->x0 = c0->sxy;
			*(uint32_t *)&pri->x1 = c1->sxy;
			*(uint32_t *)&pri->x2 = c2->sxy;
			*(uint32_t *)&pri->u0 = verts[i0].uv | clut_hi[tri->tex];
			*(uint32_t *)&pri->u1 = verts[i1].uv | tpage_hi[tri->tex];
			*(uint32_t *)&pri->u2 = verts[i2].uv;
			addPrim(ot + otz, pri);
			pri++;
			drawn++;
		}
		nextpri = (char *)pri;
	}
	r->tris_drawn = drawn;
	prof_mark(PROF_PRIMS);
	return nextpri;
}
