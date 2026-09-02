/* Binary layout of model.bin produced by tools/fbx2ps1.py (little endian).
 * All fixed point values use 4096 == 1.0.  Positions are PS1 units
 * (4096 == 1.0 FBX unit, i.e. a 1 m tall character spans 4096 units) in the
 * local space of the vertex's bone.
 * Structures are not packed: every field sits at its natural alignment and
 * the converter pads each section to 4 bytes, so the MIPS compiler can use
 * whole-word loads instead of byte-wise access. */
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#define MODEL_MAGIC        "PS1M"
#define MODEL_FIX_ONE      4096
#define BONE_FLAG_TRANS    1        /* translation stored per frame */

typedef struct {
	char     magic[4];
	uint16_t nverts, ntris, nbones, nanims, ntrans, nedges;
	uint32_t off_verts, off_tris, off_bones, off_anims;
	uint32_t off_ik, off_edges;     /* 0: table absent; edges: nedges x {uint16 a, b} */
	uint32_t off_quads;             /* cloth quads (ModelQuad), 0: none */
	uint16_t nquads, pad3;
} ModelHeader;                      /* 48 bytes */

/* One cloth cell of the wire grid: its 4 corner vertices (ring order) and
 * the two triangles that render it. */
typedef struct {
	uint16_t v[4];
	uint16_t tri[2];
} ModelQuad;                        /* 12 bytes */

typedef struct {
	int16_t  x, y, z;
	uint8_t  bone, pad;
	int16_t  nx, ny, nz;            /* unit normal, bone-local */
	uint16_t uv;                    /* u | v << 8, texel coordinates */
} ModelVert;                        /* 16 bytes */

typedef struct {
	uint16_t i0, i1, i2;            /* front face: GTE nclip > 0 */
	uint8_t  bone;                  /* bone whose space nx/ny/nz live in */
	uint8_t  tex;                   /* texture slot (0 = body, 1 = face) */
	int16_t  nx, ny, nz;            /* face normal for flat shading */
	uint8_t  flags;                 /* TRI_FLAG_* */
	uint8_t  pad2;
} ModelTri;                         /* 16 bytes */

#define TRI_FLAG_DOUBLE_SIDED 1     /* draw the back face too (winding reversed) */

typedef struct {
	int8_t   parent;                /* -1 = root */
	uint8_t  flags;
	uint8_t  trans_slot;            /* index into the per-frame translation block */
	uint8_t  pad;
	uint16_t vert_start, vert_count;
	uint16_t tri_start, tri_count;
	int16_t  tx, ty, tz;            /* bind-pose translation from parent */
} ModelBone;                        /* 18 bytes */

typedef struct {
	char     name[16];
	uint16_t nframes;
	uint16_t fps;
	uint32_t offset;                /* byte offset of frame data from file start */
} ModelAnim;                        /* 24 bytes */

/* Frame data: nbones x { int16 qx, qy, qz, qw } followed by
 *             ntrans x { int16 tx, ty, tz, pad }                       */
typedef struct {
	int16_t x, y, z, w;
} ModelQuat;

typedef struct {
	int16_t x, y, z, pad;
} ModelTrans;

/* Two-bone IK chain: upper -> lower -> effector.  The effector sits at
 * end_local (lower bone space, PS1 units); end is the bone placed there
 * (-1 if the mesh has none).  pole: model-space direction the middle joint
 * bends towards (knee forward, elbow backward), 4096 = 1.0. */
typedef struct {
	int8_t   upper, lower, end, pad;
	int16_t  end_local[3];
	int16_t  pole[3];
} ModelIKChain;                     /* 16 bytes */

enum { IK_ARM_L = 0, IK_ARM_R, IK_LEG_L, IK_LEG_R, IK_CHAINS };

typedef struct {
	int8_t   hip;                   /* bone moved by the d-pad (ancestor of all chains) */
	int8_t   head;                  /* bone turned towards the camera (-1: none) */
	int8_t   pad0, pad1;
	int16_t  head_fwd[3];           /* head "forward" in head-local space, 4096 = 1.0 */
	int16_t  pad2;
	ModelIKChain chains[IK_CHAINS]; /* upper < 0: chain missing */
} ModelIK;                          /* 76 bytes */

typedef struct {
	const ModelHeader *hdr;
	const ModelVert   *verts;
	const ModelTri    *tris;
	const ModelBone   *bones;
	const ModelAnim   *anims;
	const ModelIK     *ik;          /* NULL if absent */
	const uint16_t    *edges;       /* wire overlay edges (vertex index pairs), NULL if absent */
	const ModelQuad   *quads;       /* cloth cells, NULL if absent */
	const uint8_t     *base;
	int frame_size;                 /* bytes per animation frame */
} Model;

static inline int model_open(Model *m, const void *data) {
	const ModelHeader *h = (const ModelHeader *)data;
	if (h->magic[0] != 'P' || h->magic[1] != 'S' || h->magic[2] != '1' || h->magic[3] != 'M')
		return -1;
	m->hdr   = h;
	m->base  = (const uint8_t *)data;
	m->verts = (const ModelVert *)(m->base + h->off_verts);
	m->tris  = (const ModelTri *)(m->base + h->off_tris);
	m->bones = (const ModelBone *)(m->base + h->off_bones);
	m->anims = (const ModelAnim *)(m->base + h->off_anims);
	m->ik    = h->off_ik ? (const ModelIK *)(m->base + h->off_ik) : (const ModelIK *)0;
	m->edges = h->off_edges ? (const uint16_t *)(m->base + h->off_edges) : (const uint16_t *)0;
	m->quads = h->off_quads ? (const ModelQuad *)(m->base + h->off_quads) : (const ModelQuad *)0;
	m->frame_size = h->nbones * sizeof(ModelQuat) + h->ntrans * sizeof(ModelTrans);
	return 0;
}

static inline const ModelQuat *model_frame_quats(const Model *m, int anim, int frame) {
	return (const ModelQuat *)(m->base + m->anims[anim].offset + frame * m->frame_size);
}

static inline const ModelTrans *model_frame_trans(const Model *m, int anim, int frame) {
	return (const ModelTrans *)((const uint8_t *)model_frame_quats(m, anim, frame)
	                            + m->hdr->nbones * sizeof(ModelQuat));
}

#endif
