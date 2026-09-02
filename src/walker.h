/* Skirt wire overlay + a monkey sprite that walks along the edges, picking
 * a random branch at every vertex. */
#ifndef WALKER_H
#define WALKER_H

#include <stdint.h>
#include <psxgpu.h>
#include "model.h"

#define MAX_EDGES 512

typedef struct {
	uint8_t visited[MAX_EDGES];   /* edges the monkey has walked (drawn yellow) */
	int edge;            /* current edge index */
	int from, to;        /* vertex indices (walking from -> to) */
	int t;               /* 0..4096 along the edge */
	int frame;           /* animation counter */
	uint32_t rng;
} Walker;

void walker_load_sprite(const uint32_t *tim);
void walker_reset(Walker *w, const Model *m);
/* screen positions come from the renderer's vertex cache: sxy packed
 * (x | y << 16) and sz per vertex */
struct VCacheTag;
char *walker_draw(Walker *w, const Model *m, const void *vcache, uint32_t *ot, char *nextpri);

#endif
