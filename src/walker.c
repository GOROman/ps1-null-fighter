#include <string.h>
#include <psxgpu.h>
#include "walker.h"
#include "fixmath.h"

#define WALK_SPEED_PX   (4096 / 3)       /* 1/3 pixel per frame, in 1/4096 */
#define SPRITE_W        16
#define SPRITE_H        16
#define PIECE_FRAMES    70

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

static int find_edge(const Model *m, int a, int b) {
	int n = m->hdr->nedges;
	for (int e = 0; e < n; e++) {
		int ea = m->edges[2 * e], eb = m->edges[2 * e + 1];
		if ((ea == a && eb == b) || (ea == b && eb == a)) return e;
	}
	return -1;
}

void walker_reset(Walker *w, const Model *m) {
	memset(w, 0, sizeof(*w));
	w->rng = 0x12345678u ^ (uint32_t)m->hdr->nedges;
	if (m->hdr->nedges == 0) return;
	w->from = m->edges[0];
	w->to   = m->edges[1];
	w->visited[0] = 1;
	/* cell <-> edge tables */
	for (int e = 0; e < MAX_EDGES; e++) w->edge_quad[e][0] = w->edge_quad[e][1] = -1;
	w->nquads = m->quads ? m->hdr->nquads : 0;
	if (w->nquads > MAX_QUADS) w->nquads = MAX_QUADS;
	for (int q = 0; q < w->nquads; q++) {
		const ModelQuad *mq = &m->quads[q];
		for (int k = 0; k < 4; k++) {
			int e = find_edge(m, mq->v[k], mq->v[(k + 1) & 3]);
			w->quad_edge[q][k] = e;
			if (e >= 0 && e < MAX_EDGES) {
				if (w->edge_quad[e][0] < 0) w->edge_quad[e][0] = q;
				else w->edge_quad[e][1] = q;
			}
		}
	}
	/* pieces: connected cell grids (jacket parts, skirt) - a cut only ever
	 * compares regions inside the same piece */
	for (int q = 0; q < w->nquads; q++) w->quad_piece[q] = -1;
	w->npieces = 0;
	for (int q = 0; q < w->nquads; q++) {
		if (w->quad_piece[q] >= 0) continue;
		static int16_t stack[MAX_QUADS];
		int sp = 0;
		stack[sp++] = q;
		w->quad_piece[q] = w->npieces;
		while (sp) {
			int c = stack[--sp];
			for (int k = 0; k < 4; k++) {
				int e = w->quad_edge[c][k];
				if (e < 0) continue;
				int o = w->edge_quad[e][0] == c ? w->edge_quad[e][1] : w->edge_quad[e][0];
				if (o >= 0 && w->quad_piece[o] < 0) { w->quad_piece[o] = w->npieces; stack[sp++] = o; }
			}
		}
		w->npieces++;
	}
}

/* after an edge got walked: cells no longer connected to the largest
 * region through unwalked edges are cut out */
static void cut_enclosed(Walker *w, const Model *m, const VCache *vc) {
	static int16_t comp[MAX_QUADS], stack[MAX_QUADS];
	static int sizes[MAX_QUADS], comp_piece[MAX_QUADS];
	static int best_of_piece[32];
	int ncomp = 0;
	for (int p = 0; p < 32; p++) best_of_piece[p] = -1;
	for (int q = 0; q < w->nquads; q++) comp[q] = w->quad_cut[q] ? -2 : -1;
	for (int q = 0; q < w->nquads; q++) {
		if (comp[q] != -1) continue;
		int sp = 0, size = 0;
		stack[sp++] = q;
		comp[q] = ncomp;
		while (sp) {
			int c = stack[--sp];
			size++;
			for (int k = 0; k < 4; k++) {
				int e = w->quad_edge[c][k];
				if (e < 0 || w->visited[e]) continue;      /* walked edges are cuts */
				int o = w->edge_quad[e][0] == c ? w->edge_quad[e][1] : w->edge_quad[e][0];
				if (o >= 0 && comp[o] == -1) { comp[o] = ncomp; stack[sp++] = o; }
			}
		}
		sizes[ncomp] = size;
		comp_piece[ncomp] = w->quad_piece[q];
		int p = w->quad_piece[q];
		if (p >= 0 && p < 32 && (best_of_piece[p] < 0 || size > sizes[best_of_piece[p]]))
			best_of_piece[p] = ncomp;
		ncomp++;
	}
	int any = 0;
	for (int c = 0; c < ncomp; c++)
		if (comp_piece[c] >= 0 && comp_piece[c] < 32 && best_of_piece[comp_piece[c]] != c) any = 1;
	if (!any) return;
	w->cut_event++;
	for (int q = 0; q < w->nquads; q++) {
		if (comp[q] < 0) continue;
		int p = w->quad_piece[q];
		if (p < 0 || p >= 32 || best_of_piece[p] == comp[q]) continue;
		const ModelQuad *mq = &m->quads[q];
		w->quad_cut[q] = 1;
		w->ncut++;
		for (int k = 0; k < 2; k++)
			if (mq->tri[k] < MAX_CUT) w->tri_cut[mq->tri[k]] = 1;
		/* spawn a falling piece */
		for (int p = 0; p < MAX_PIECES; p++) {
			Piece *pc = &w->pieces[p];
			if (pc->active) continue;
			pc->active = 1;
			pc->t = 0;
			int zsum = 0;
			for (int k = 0; k < 4; k++) {
				uint32_t s = vc[mq->v[k]].sxy;
				pc->x[k] = (int16_t)(s & 0xffff);
				pc->y[k] = (int16_t)(s >> 16);
				zsum += vc[mq->v[k]].sz;
				pc->uv[k] = m->verts[mq->v[k]].uv;
			}
			pc->otz = (zsum * 3) >> (4 + OTZ_SHIFT);   /* avg of 4 -> same scale as triangles */
			if (pc->otz <= 8) pc->otz = 8;
			if (pc->otz >= OT_LEN) pc->otz = OT_LEN - 1;
			break;
		}
	}
}

/* arrived at vertex `to`: choose the next edge - the one matching the
 * d-pad direction on screen, else the straightest continuation */
static void next_edge(Walker *w, const Model *m, const VCache *vc) {
	int n = m->hdr->nedges;
	int at = w->to;
	int32_t cur_dx = (int16_t)(vc[w->to].sxy & 0xffff) - (int16_t)(vc[w->from].sxy & 0xffff);
	int32_t cur_dy = (int16_t)(vc[w->to].sxy >> 16) - (int16_t)(vc[w->from].sxy >> 16);
	int steering = w->dir_x || w->dir_y;
	int32_t want_x = w->dir_x * 64, want_y = w->dir_y * 64;
	(void)cur_dx; (void)cur_dy;
	int best = -1, best_to = -1;
	int64_t best_score = -(1 << 30);
	for (int e = 0; e < n; e++) {
		if (e == w->edge) continue;
		int a = m->edges[2 * e], b = m->edges[2 * e + 1];
		if (a != at && b != at) continue;
		int other = (a == at) ? b : a;
		int64_t score;
		if (steering) {
			/* the branch pointing closest to the d-pad direction on screen */
			int32_t dx = (int16_t)(vc[other].sxy & 0xffff) - (int16_t)(vc[at].sxy & 0xffff);
			int32_t dy = (int16_t)(vc[other].sxy >> 16) - (int16_t)(vc[at].sxy >> 16);
			int32_t len = (int32_t)isqrt32((uint32_t)(dx * dx + dy * dy)) + 1;
			score = ((int64_t)(dx * want_x + dy * want_y) << 8) / len + (rnd(w) & 31);
		} else {
			score = rnd(w) & 0xffff;                    /* no input: random branch */
		}
		if (score > best_score) { best_score = score; best = e; best_to = other; }
	}
	if (best < 0) {               /* dead end: turn around */
		int tmp = w->from; w->from = w->to; w->to = tmp;
		w->t = 0;
		return;
	}
	w->edge = best;
	w->from = at;
	w->to = best_to;
	w->t = 0;
	if (best < MAX_EDGES && !w->visited[best]) {
		w->visited[best] = 1;
		cut_enclosed(w, m, vc);
	}
}

static int iabs(int v) { return v < 0 ? -v : v; }

char *walker_draw(Walker *w, const Model *m, const Renderer *r, const VCache *vc, uint32_t *ot, char *nextpri) {
	int n = m->hdr->nedges;
	if (n == 0) return nextpri;

	/* ---- wire on the quad edges: white, walked = yellow ------------------ */
	LINE_F2 *l = (LINE_F2 *)nextpri;
	for (int e = 0; e < n; e++) {
		int a = m->edges[2 * e], b = m->edges[2 * e + 1];
		if (vc[a].sz <= 0 || vc[b].sz <= 0) continue;
		/* edges of cut cells only remain where a live cell still uses them */
		if (e < MAX_EDGES) {
			int q0 = w->edge_quad[e][0], q1 = w->edge_quad[e][1];
			int live = (q0 >= 0 && !w->quad_cut[q0]) || (q1 >= 0 && !w->quad_cut[q1]);
			if (!live) continue;
		}
		int otz = ((vc[a].sz + vc[b].sz) * 3) >> (3 + OTZ_SHIFT);
		otz -= 6;
		if (otz <= 0 || otz >= OT_LEN) continue;
		setLineF2(l);
		if (e < MAX_EDGES && w->visited[e])
			setRGB0(l, 255, 230, 40);
		else
			setRGB0(l, 255, 255, 255);
		*(uint32_t *)&l->x0 = vc[a].sxy;
		*(uint32_t *)&l->x1 = vc[b].sxy;
		addPrim(ot + otz, l);
		l++;
	}
	nextpri = (char *)l;

	/* ---- falling pieces --------------------------------------------------- */
	{
		POLY_FT4 *q = (POLY_FT4 *)nextpri;
		for (int p = 0; p < MAX_PIECES; p++) {
			Piece *pc = &w->pieces[p];
			if (!pc->active) continue;
			int t = pc->t++;
			if (t >= PIECE_FRAMES) { pc->active = 0; continue; }
			int dy = (t * t) / 40;
			int dx = (t & 16) ? (t & 15) - 8 : 8 - (t & 15);
			setPolyFT4(q);
			setRGB0(q, 128, 128, 128);
			setXY4(q, pc->x[0] + dx, pc->y[0] + dy, pc->x[1] + dx, pc->y[1] + dy,
			          pc->x[3] + dx, pc->y[3] + dy, pc->x[2] + dx, pc->y[2] + dy);
			*(uint32_t *)&q->u0 = pc->uv[0] | ((uint32_t)r->clut[2] << 16);
			*(uint32_t *)&q->u1 = pc->uv[1] | ((uint32_t)r->tpage[2] << 16);
			*(uint32_t *)&q->u2 = pc->uv[3];
			*(uint32_t *)&q->u3 = pc->uv[2];
			addPrim(ot + pc->otz, q);
			q++;
		}
		nextpri = (char *)q;
	}

	/* ---- monkey ---------------------------------------------------------- */
	int ax = (int16_t)(vc[w->from].sxy & 0xffff), ay = (int16_t)(vc[w->from].sxy >> 16);
	int bx = (int16_t)(vc[w->to].sxy & 0xffff),   by = (int16_t)(vc[w->to].sxy >> 16);
	int len = iabs(bx - ax) + iabs(by - ay);
	if (len < 1) len = 1;
	w->t += WALK_SPEED_PX / len;
	if (w->t >= 4096) {
		next_edge(w, m, vc);
		ax = (int16_t)(vc[w->from].sxy & 0xffff); ay = (int16_t)(vc[w->from].sxy >> 16);
		bx = (int16_t)(vc[w->to].sxy & 0xffff);   by = (int16_t)(vc[w->to].sxy >> 16);
	}
	w->frame++;
	int x = ax + (((bx - ax) * w->t) >> 12);
	int y = ay + (((by - ay) * w->t) >> 12);
	int depth = vc[w->from].sz + (((vc[w->to].sz - vc[w->from].sz) * w->t) >> 12);
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
