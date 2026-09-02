#ifndef RENDER_H
#define RENDER_H

#include <stdint.h>
#include <psxgpu.h>
#include "model.h"
#include "anim.h"
#include "camera.h"

#define OT_LEN     4096
#define MAX_VERTS  4096
#define OTZ_SHIFT  3        /* camera z -> ordering table index */

enum { SHADE_GOURAUD = 0, SHADE_FLAT = 1 };

/* projected vertex, filled by render_model pass 1 (index = model vertex) */
typedef struct {
	uint32_t sxy;                   /* packed screen x,y (as stored by GTE) */
	int32_t  sz;
	uint32_t rgb;                   /* packed r,g,b,code */
} VCache;

#define MAX_TEX    3

typedef struct {
	uint16_t tpage[MAX_TEX], clut[MAX_TEX];
	int ntex;
	int shading;
	int tris_drawn;
	const uint8_t *cut;          /* per model triangle: 1 = do not draw (cloth cut out); NULL = none */
} Renderer;

/* tims[i] may be NULL (unused slots fall back to slot 0) */
void renderer_init(Renderer *r, const uint32_t *const tims[MAX_TEX]);
/* transform + light the posed model and sort it into ot; returns the new
 * packet pointer. */
char *render_model(Renderer *r, const Model *m, const Pose *pose, const Camera *cam,
                   uint32_t *ot, char *nextpri);
/* projected vertices of the last render_model call */
const VCache *render_get_cache(void);

#endif
