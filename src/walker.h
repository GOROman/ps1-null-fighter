/* Dancing-Eyes style cloth game: a monkey walks the wire grid of the
 * skirt (steered with the d-pad, straight on by default), and every cell
 * enclosed by the walked path is cut out and drops off. */
#ifndef WALKER_H
#define WALKER_H

#include <stdint.h>
#include <psxgpu.h>
#include "model.h"
#include "render.h"

#define MAX_EDGES   512
#define MAX_QUADS   256
#define MAX_CUT     4096          /* per-triangle cut flags (index = model triangle) */
#define MAX_PIECES  12

typedef struct {
	int active, t;
	int32_t x[4], y[4];       /* screen corners at cut time */
	int otz;
	uint16_t uv[4];
} Piece;

typedef struct {
	uint8_t visited[MAX_EDGES];   /* edges the monkey has walked (drawn yellow) */
	uint8_t quad_cut[MAX_QUADS];
	uint8_t tri_cut[MAX_CUT];     /* read by the renderer */
	int16_t quad_edge[MAX_QUADS][4];
	int8_t  quad_piece[MAX_QUADS];    /* cloth piece (connected grid) a cell belongs to */
	int npieces;
	int16_t edge_quad[MAX_EDGES][2];
	int nquads, ncut;
	int edge;            /* current edge index */
	int from, to;        /* vertex indices (walking from -> to) */
	int t;               /* 0..4096 along the edge */
	int frame;           /* animation counter */
	int dir_x, dir_y;    /* d-pad steering in screen space (-1/0/1) */
	int cut_event;       /* incremented whenever cloth drops (main reacts) */
	uint32_t rng;
	Piece pieces[MAX_PIECES];
} Walker;

void walker_load_sprite(const uint32_t *tim);
void walker_reset(Walker *w, const Model *m);
char *walker_draw(Walker *w, const Model *m, const Renderer *r, const VCache *vc, uint32_t *ot, char *nextpri);

#endif
