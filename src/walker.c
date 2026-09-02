#include <psxgpu.h>
#include "walker.h"
#include "render.h"

#define WALK_SPEED_PX   (4096 / 3)       /* 1/3 pixel per frame, in 1/4096 */
#define SPRITE_W        16
#define SPRITE_H        16

static uint16_t sprite_tpage, sprite_clut;

void walker_load_sprite(const uint32_t *tim_data) {
	TIM_IMAGE tim;
	GetTimInfo(tim_data, &tim);
	if (tim.mode & 0x8)
		LoadImage(tim.crect, tim.caddr);
	LoadImage(tim.prect, tim.paddr);
	DrawSync(0);
	sprite_tpage = getTPage(tim.mode & 3, 0, tim.prect->x, tim.prect->y);
	sprite_clut  = getClut(tim.crect->x, tim.crect->y);
}

static uint32_t rnd(Walker *w) {
	w->rng = w->rng * 1664525u + 1013904223u;
	return w->rng >> 8;
}

void walker_reset(Walker *w, const Model *m) {
	for (int i = 0; i < MAX_EDGES; i++) w->visited[i] = 0;
	w->rng = 0x12345678u ^ (uint32_t)m->hdr->nedges;
	w->edge = 0;
	w->t = 0;
	w->frame = 0;
	if (m->hdr->nedges == 0) return;
	w->from = m->edges[0];
	w->to   = m->edges[1];
	w->visited[0] = 1;
}

/* arrived at vertex `to`: pick a random edge leaving it (not the one we came
 * from unless it is the only one) */
static void next_edge(Walker *w, const Model *m) {
	int n = m->hdr->nedges;
	int at = w->to;
	int count = 0;
	for (int e = 0; e < n; e++) {
		if (e == w->edge) continue;
		if (m->edges[2 * e] == at || m->edges[2 * e + 1] == at) count++;
	}
	if (count == 0) {              /* dead end: turn around */
		int tmp = w->from; w->from = w->to; w->to = tmp;
		w->t = 0;
		return;
	}
	int pick = rnd(w) % count;
	for (int e = 0; e < n; e++) {
		if (e == w->edge) continue;
		int a = m->edges[2 * e], b = m->edges[2 * e + 1];
		if (a == at || b == at) {
			if (pick-- == 0) {
				if (e < MAX_EDGES) w->visited[e] = 1;
				w->edge = e;
				w->from = at;
				w->to = (a == at) ? b : a;
				w->t = 0;
				return;
			}
		}
	}
}

static int iabs(int v) { return v < 0 ? -v : v; }

#define SXY(i) (vc[i].sxy)
#define SZ(i)  (vc[i].sz)

char *walker_draw(Walker *w, const Model *m, const void *vcache, uint32_t *ot, char *nextpri) {
	const VCache *vc = (const VCache *)vcache;
	int n = m->hdr->nedges;
	if (n == 0) return nextpri;

	/* ---- white wire on the quad edges ----------------------------------- */
	LINE_F2 *l = (LINE_F2 *)nextpri;
	for (int e = 0; e < n; e++) {
		int a = m->edges[2 * e], b = m->edges[2 * e + 1];
		if (SZ(a) <= 0 || SZ(b) <= 0) continue;
		int otz = ((SZ(a) + SZ(b)) * 3) >> (3 + OTZ_SHIFT);
		otz -= 6;                                   /* a little in front of the skirt surface */
		if (otz <= 0 || otz >= OT_LEN) continue;
		setLineF2(l);
		if (e < MAX_EDGES && w->visited[e])
			setRGB0(l, 255, 230, 40);               /* walked: yellow */
		else
			setRGB0(l, 255, 255, 255);
		*(uint32_t *)&l->x0 = SXY(a);
		*(uint32_t *)&l->x1 = SXY(b);
		addPrim(ot + otz, l);
		l++;
	}
	nextpri = (char *)l;

	/* ---- monkey ---------------------------------------------------------- */
	int ax = (int16_t)(SXY(w->from) & 0xffff), ay = (int16_t)(SXY(w->from) >> 16);
	int bx = (int16_t)(SXY(w->to) & 0xffff),   by = (int16_t)(SXY(w->to) >> 16);
	int len = iabs(bx - ax) + iabs(by - ay);
	if (len < 1) len = 1;
	w->t += WALK_SPEED_PX / len;
	if (w->t >= 4096) {
		next_edge(w, m);
		ax = (int16_t)(SXY(w->from) & 0xffff); ay = (int16_t)(SXY(w->from) >> 16);
		bx = (int16_t)(SXY(w->to) & 0xffff);   by = (int16_t)(SXY(w->to) >> 16);
	}
	w->frame++;
	int x = ax + (((bx - ax) * w->t) >> 12);
	int y = ay + (((by - ay) * w->t) >> 12);
	int depth = SZ(w->from) + (((SZ(w->to) - SZ(w->from)) * w->t) >> 12);
	if (depth > 0) {
		int otz = (depth * 3) >> (2 + OTZ_SHIFT);
		otz -= 8;
		if (otz <= 0) otz = 1;
		if (otz >= OT_LEN) otz = OT_LEN - 1;
		int facing_left = bx < ax;
		int anim = (w->frame >> 3) & 1;
		/* the sprite uses the GPU's current texture page, and every textured
		 * polygon changes it, so the DR_TPAGE must come immediately before
		 * the sprite: same OT bucket, added after it (addPrim links at the
		 * head, so the last added is drawn first) */
		SPRT_16 *s = (SPRT_16 *)nextpri;
		setSprt16(s);
		setXY0(s, x - SPRITE_W / 2, y - SPRITE_H + 2);
		setUV0(s, (facing_left ? 32 : 0) + anim * 16, 0);
		setRGB0(s, 128, 128, 128);
		s->clut = sprite_clut;
		addPrim(ot + otz, s);
		nextpri = (char *)(s + 1);
		DR_TPAGE *tp = (DR_TPAGE *)nextpri;
		setDrawTPage(tp, 0, 0, sprite_tpage);
		addPrim(ot + otz, tp);
		nextpri = (char *)(tp + 1);
	}
	return nextpri;
}
