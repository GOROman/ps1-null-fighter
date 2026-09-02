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

#define MAX_TEX    2

typedef struct {
	uint16_t tpage[MAX_TEX], clut[MAX_TEX];
	int ntex;
	int shading;
	int tris_drawn;
} Renderer;

/* tim1 may be NULL (single texture) */
void renderer_init(Renderer *r, const uint32_t *tim0, const uint32_t *tim1);
/* transform + light the posed model and sort it into ot; returns the new
 * packet pointer. */
char *render_model(Renderer *r, const Model *m, const Pose *pose, const Camera *cam,
                   uint32_t *ot, char *nextpri);

#endif
