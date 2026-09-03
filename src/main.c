/* NULL FIGHTER - skinned FBX character viewer for PlayStation 1
 *
 * Controls
 *   D-pad                steer the monkey on the cloth grid (Dancing Eyes style)
 *   Left stick           orbit camera        Right stick up/down: camera up / down
 *   SELECT + D-pad       orbit camera (for digital pads)
 *   SELECT + Triangle/Cross   camera up / down
 *   SELECT + Square      reset the cloth
 *   SELECT + Circle      face picture-in-picture on / off
 *   Triangle / Cross     dolly in / out (distance)
 *   Square / Circle      zoom in / out (field of view)
 *   L1                   rotate model (yaw)
 *   R1                   IK mode on / off (the dance restarts on entry)
 *   START + SELECT       fight mode (CPU vs CPU) <-> viewer
 *   L2 / R2              previous / next animation (outside IK mode)
 *   START (tap)          switch character
 *   L3 (stick click)     pause / resume animation
 *
 * The bar on the left is a frame profiler in h-blank units (a full frame
 * is the white line): yellow input, orange pose, green vertex transform,
 * cyan primitive setup, magenta text/overlay, red waiting for the GPU,
 * black waiting for v-blank.  A frame longer than 1/60 s spills into a
 * second column.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <psxgpu.h>
#include <psxgte.h>
#include <psxpad.h>
#include <psxapi.h>
#include <psxetc.h>
#include <inline_c.h>
#include "model.h"
#include "anim.h"
#include "camera.h"
#include "render.h"
#include "prof.h"
#include "ik.h"
#include "fixmath.h"
#include "walker.h"
#include "fight.h"

#define SCREEN_XRES   320
#define SCREEN_YRES   240
#define CENTERX       (SCREEN_XRES >> 1)
#define CENTERY       (SCREEN_YRES >> 1)
#define PACKET_LEN    (192 * 1024)

typedef struct {
	DISPENV  disp;
	DRAWENV  draw;
	uint32_t ot[OT_LEN];
	char     p[PACKET_LEN];
} DB;

static DB   db[2];
static int  db_active = 0;
static char *db_nextpri;

extern const uint32_t schoolgirl_bin[], schoolgirl_tim[], schoolgirl_face_tim[], schoolgirl_skirt_tim[];
extern const uint32_t character_bin[],  character_tim[];
extern const uint32_t schoolgirl2_bin[], schoolgirl2_tim[];
extern const uint32_t monkey_tim[];

typedef struct {
	const uint32_t *bin;
	const uint32_t *tims[MAX_TEX];          /* body, face, skirt (NULL = unused) */
	int yaw;                                /* model yaw that faces the camera */
	int look_at;                            /* head always turned towards the camera */
	int pip;                                /* face picture-in-picture on by default */
} ModelAsset;

static const ModelAsset assets[] = {
	{ character_bin,  { character_tim, 0, 0 }, 0, 1, 1 },      /* the animated character */
	{ schoolgirl_bin, { schoolgirl_tim, schoolgirl_face_tim, schoolgirl_skirt_tim }, 0, 1, 0 },  /* PiP costs too much here */
	{ schoolgirl2_bin, { schoolgirl2_tim, 0, 0 }, 0, 1, 0 },   /* Tripo rig + the blonde's animations */
};
#define NUM_ASSETS ((int)(sizeof(assets) / sizeof(assets[0])))

static uint8_t pad_buff[2][34];
static int fnt_hud = -1, fnt_dbg = -1, fnt_fps = -1;

/* Face camera picture-in-picture: the head bone rendered again from a
 * camera in front of the face into a small box at the top right, clipped
 * with DR_AREA.  Its primitives use OT indices 1..PIP_OT-1 so they are drawn
 * after the main scene. */
#define PIP_W    80
#define PIP_H    80
#define PIP_X    (SCREEN_XRES - 8 - PIP_W)
#define PIP_Y    22
#define PIP_OT   48
// #define PIP_NOCLIP 1
static char *draw_face_pip(Renderer *r, const Model *m, const Pose *pose, int model_yaw,
                           const DRAWENV *env, uint32_t *ot, char *nextpri) {
	if (!m->ik || m->ik->head < 0) return nextpri;
	static uint8_t mask[MAX_BONES];
	memset(mask, 0, sizeof(mask));
	mask[m->ik->head] = 1;
	/* camera in front of the face: along the head's forward vector (model
	 * space), looking back at a point a little above the head joint, so it
	 * follows the face whatever the model / head rotation is */
	(void)model_yaw;
	Camera pc;
	camera_init(&pc);
	pc.fov = 120;
	const MATRIX *wh = &pose->world[m->ik->head];
	VECTOR fwd_local = vec(m->ik->head_fwd[0], m->ik->head_fwd[1], m->ik->head_fwd[2]), fwd;
	ApplyMatrixLV((MATRIX *)wh, &fwd_local, &fwd);
	/* the camera position follows the point in front of the face with a
	 * low-pass lag (1/10 of the gap per frame); the look-at target is the
	 * face itself, every frame */
	static VECTOR lp_pos;
	static int lp_init;
	VECTOR f_now = vnorm(fwd);                              /* head forward, 4096 */
	VECTOR target = vec(wh->t[0], wh->t[1] - 300, wh->t[2]); /* face centre, a bit above the joint */
	const int dist = 1300;                                  /* close-up */
	VECTOR want = vadd(target, vscale(f_now, dist));
	if (!lp_init) { lp_pos = want; lp_init = 1; }
	lp_pos.vx += (want.vx - lp_pos.vx) / 10;
	lp_pos.vy += (want.vy - lp_pos.vy) / 10;
	lp_pos.vz += (want.vz - lp_pos.vz) / 10;
	VECTOR campos = lp_pos;
	VECTOR d = vnorm(vsub(target, campos));                 /* view direction: at the face */
	if (d.vx == 0 && d.vy == 0 && d.vz == 0) d = vscale(f_now, -4096);
	VECTOR up0 = vec(0, 4096, 0);                           /* y is down: "up" row = +y */
	VECTOR rt = vnorm(vcross(up0, d));
	if (rt.vx == 0 && rt.vy == 0 && rt.vz == 0) rt = vec(4096, 0, 0);
	VECTOR u = vcross(d, rt);
	MATRIX *v = &pc.view;
	v->m[0][0] = rt.vx; v->m[0][1] = rt.vy; v->m[0][2] = rt.vz;
	v->m[1][0] = u.vx; v->m[1][1] = u.vy; v->m[1][2] = u.vz;
	v->m[2][0] = d.vx; v->m[2][1] = d.vy; v->m[2][2] = d.vz;
	VECTOR cr;
	ApplyMatrixLV(v, &campos, &cr);                          /* t = -R * campos */
	v->t[0] = -cr.vx; v->t[1] = -cr.vy; v->t[2] = -cr.vz;
	gte_SetGeomOffset(PIP_X + PIP_W / 2, PIP_Y + PIP_H / 2);
	gte_SetGeomScreen(pc.fov);
	Renderer pr = *r;
	pr.bone_mask = mask;
	pr.otz_shift = 2 + OTZ_SHIFT + 3;      /* compress the depth range into the PiP's OT slice */
	pr.otz_base = 1;
	pr.otz_limit = PIP_OT;
	/* clip to the box while the PiP is drawn: set at PIP_OT (drawn before
	 * its primitives), restore at 0 (drawn after) */
	TILE *bg = (TILE *)nextpri;
	setTile(bg);
	setRGB0(bg, 20, 24, 40);
	setXY0(bg, PIP_X, PIP_Y);
	setWH(bg, PIP_W, PIP_H);
	addPrim(ot + PIP_OT, bg);
	nextpri = (char *)(bg + 1);
	/* DR_AREA takes VRAM coordinates: offset by this buffer's clip origin */
	DR_AREA *da = (DR_AREA *)nextpri;
	RECT rc = { env->clip.x + PIP_X, env->clip.y + PIP_Y, PIP_W, PIP_H };
	setDrawArea(da, &rc);
#ifndef PIP_NOCLIP
	addPrim(ot + PIP_OT, da);
#endif
	nextpri = (char *)(da + 1);
	nextpri = render_model(&pr, m, pose, &pc, ot, nextpri);
	/* frame: added before the restore so it is drawn after it (bucket 0 is
	 * drawn newest first) */
	TILE *fr = (TILE *)nextpri;
	setTile(fr); setRGB0(fr, 200, 200, 210); setXY0(fr, PIP_X - 1, PIP_Y - 1); setWH(fr, PIP_W + 2, 1); addPrim(ot, fr); fr++;
	setTile(fr); setRGB0(fr, 200, 200, 210); setXY0(fr, PIP_X - 1, PIP_Y + PIP_H); setWH(fr, PIP_W + 2, 1); addPrim(ot, fr); fr++;
	setTile(fr); setRGB0(fr, 200, 200, 210); setXY0(fr, PIP_X - 1, PIP_Y); setWH(fr, 1, PIP_H); addPrim(ot, fr); fr++;
	setTile(fr); setRGB0(fr, 200, 200, 210); setXY0(fr, PIP_X + PIP_W, PIP_Y); setWH(fr, 1, PIP_H); addPrim(ot, fr); fr++;
	nextpri = (char *)fr;
	DR_AREA *dr = (DR_AREA *)nextpri;
	RECT full = env->clip;
	setDrawArea(dr, &full);
#ifndef PIP_NOCLIP
	addPrim(ot, dr);
#endif
	nextpri = (char *)(dr + 1);
	return nextpri;
}

uint16_t prof_stage[PROF_STAGES];
uint16_t prof_shown[PROF_STAGES];
uint16_t prof_last;

static void init_graphics(void) {
	ResetGraph(0);

	SetDefDispEnv(&db[0].disp, 0, 0, SCREEN_XRES, SCREEN_YRES);
	SetDefDrawEnv(&db[0].draw, SCREEN_XRES, 0, SCREEN_XRES, SCREEN_YRES);
	SetDefDispEnv(&db[1].disp, SCREEN_XRES, 0, SCREEN_XRES, SCREEN_YRES);
	SetDefDrawEnv(&db[1].draw, 0, 0, SCREEN_XRES, SCREEN_YRES);

	for (int i = 0; i < 2; i++) {
		setRGB0(&db[i].draw, 28, 32, 48);
		db[i].draw.isbg = 1;
		db[i].draw.dtd  = 1;
	}
	PutDrawEnv(&db[0].draw);

	ClearOTagR(db[0].ot, OT_LEN);
	ClearOTagR(db[1].ot, OT_LEN);
	db_nextpri = db[0].p;

	InitGeom();
	gte_SetGeomOffset(CENTERX, CENTERY);
	gte_SetGeomScreen(CENTERX);

	FntLoad(960, 0);
	fnt_hud = FntOpen(12, 8, SCREEN_XRES - 20, 16, 0, 64);                 /* game HUD, top */
	fnt_dbg = FntOpen(12, SCREEN_YRES - 40, SCREEN_XRES - 20, 32, 0, 256);  /* debug, bottom */
	fnt_fps = FntOpen(SCREEN_XRES - 8 - 64, 8, 64, 8, 0, 16);                /* FPS, top right */
}

/* index of the animation whose name starts with `name` (0 if none) */
static int find_anim(const Model *m, const char *name) {
	for (int i = 0; i < m->hdr->nanims; i++) {
		const char *a = m->anims[i].name;
		int k = 0;
		while (name[k] && (a[k] | 0x20) == (name[k] | 0x20)) k++;   /* case insensitive */
		if (!name[k]) return i;
	}
	return 0;
}

/* sin(2 pi (t / period + phase / 4096)) in 4096 units, via the GTE rotation matrix */
static int dance_sin(int t, int period, int phase) {
	SVECTOR ang = { 0, ((t * 4096) / period + phase) & 4095, 0, 0 };
	MATRIX rm;
	RotMatrix(&ang, &rm);                              /* m[0][2] = sin(y angle) */
	return rm.m[0][2];
}

/* Floor: Space Harrier style checkerboard on the Y = 0 plane (model
 * space, feet level).  The grid vertices are projected once through the
 * GTE; a cell is drawn as a flat quad when its four corners are in front
 * of the near plane and it fits the GPU's primitive size limit. */
#define GRID_STEP  1100
#define GRID_N     4                      /* cells from -N..N-1 in each direction */
#define GRID_V     (2 * GRID_N + 1)
#define GRID_NEAR  96
static char *draw_floor(const Camera *cam, uint32_t *ot, char *nextpri) {
	static uint32_t gsxy[GRID_V][GRID_V];
	static int32_t  gsz[GRID_V][GRID_V];
	gte_SetRotMatrix(&cam->view);
	gte_SetTransMatrix(&cam->view);
	for (int j = 0; j < GRID_V; j++) {
		for (int i = 0; i < GRID_V; i++) {
			SVECTOR v = { (i - GRID_N) * GRID_STEP, 0, (j - GRID_N) * GRID_STEP, 0 };
			gte_ldv0(&v);
			gte_rtps();
			__asm__ volatile("swc2 $14, 0(%0); swc2 $19, 0(%1)" :: "r"(&gsxy[j][i]), "r"(&gsz[j][i]) : "memory");
		}
	}
	POLY_F4 *q = (POLY_F4 *)nextpri;
	for (int j = 0; j < GRID_V - 1; j++) {
		for (int i = 0; i < GRID_V - 1; i++) {
			int32_t z0 = gsz[j][i], z1 = gsz[j][i + 1], z2 = gsz[j + 1][i], z3 = gsz[j + 1][i + 1];
			if (z0 <= GRID_NEAR || z1 <= GRID_NEAR || z2 <= GRID_NEAR || z3 <= GRID_NEAR)
				continue;
			uint32_t s0 = gsxy[j][i], s1 = gsxy[j][i + 1], s2 = gsxy[j + 1][i], s3 = gsxy[j + 1][i + 1];
			int x0 = (int16_t)(s0 & 0xffff), x1 = (int16_t)(s1 & 0xffff), x2 = (int16_t)(s2 & 0xffff), x3 = (int16_t)(s3 & 0xffff);
			int y0 = (int16_t)(s0 >> 16), y1 = (int16_t)(s1 >> 16), y2 = (int16_t)(s2 >> 16), y3 = (int16_t)(s3 >> 16);
			int xmin = x0, xmax = x0, ymin = y0, ymax = y0;
			if (x1 < xmin) xmin = x1; if (x1 > xmax) xmax = x1;
			if (x2 < xmin) xmin = x2; if (x2 > xmax) xmax = x2;
			if (x3 < xmin) xmin = x3; if (x3 > xmax) xmax = x3;
			if (y1 < ymin) ymin = y1; if (y1 > ymax) ymax = y1;
			if (y2 < ymin) ymin = y2; if (y2 > ymax) ymax = y2;
			if (y3 < ymin) ymin = y3; if (y3 > ymax) ymax = y3;
			if (xmax < 0 || xmin >= SCREEN_XRES || ymax < 0 || ymin >= SCREEN_YRES)
				continue;                                   /* off screen */
			if (xmax - xmin > 1000 || ymax - ymin > 500)
				continue;                                   /* too big for the GPU */
			/* the floor is a single plane: draw it first (far end of the
			 * reversed OT) so nothing standing on it is ever hidden by a
			 * cell whose average depth happens to be nearer */
			int otz = OT_LEN - 1;
			setPolyF4(q);
			if ((i + j) & 1)
				setRGB0(q, 44, 76, 140);
			else
				setRGB0(q, 26, 46, 96);
			*(uint32_t *)&q->x0 = s0;
			*(uint32_t *)&q->x1 = s1;
			*(uint32_t *)&q->x2 = s2;
			*(uint32_t *)&q->x3 = s3;
			addPrim(ot + otz, q);
			q++;
		}
	}
	return (char *)q;
}

/* Debug: the model's textures at the bottom right (above the debug text) */
#define TEXPREV_SIZE 48
static char *draw_texture_preview(const Renderer *r, uint32_t *ot, char *nextpri) {
	POLY_FT4 *q = (POLY_FT4 *)nextpri;
	for (int i = 0; i < r->ntex; i++) {
		int x0 = SCREEN_XRES - 8 - TEXPREV_SIZE * (r->ntex - i) - 4 * (r->ntex - 1 - i);
		int y0 = SCREEN_YRES - 44 - TEXPREV_SIZE;
		setPolyFT4(q);
		setRGB0(q, 128, 128, 128);
		setXY4(q, x0, y0, x0 + TEXPREV_SIZE, y0, x0, y0 + TEXPREV_SIZE, x0 + TEXPREV_SIZE, y0 + TEXPREV_SIZE);
		setUV4(q, 0, 0, 255, 0, 0, 255, 255, 255);
		q->tpage = r->tpage[i];
		q->clut  = r->clut[i];
		addPrim(ot, q);
		q++;
	}
	return (char *)q;
}

/* Profiler bar: stages stacked top to bottom at the left edge, PROF_SCALE
 * pixels per frame (263 h-blanks).  Time beyond one frame (a 30 fps frame)
 * continues in a second column to the right, so a dropped frame shows up
 * as two bars. */
#define PROF_X      2
#define PROF_W      4
#define PROF_Y      8
#define PROF_SCALE  200
#define PROF_FRAME  263
static const uint8_t prof_color[PROF_STAGES][3] = {
	{ 255, 220,  40 },   /* input   yellow */
	{ 255, 160,  40 },   /* pose    orange */
	{  60, 220,  80 },   /* verts   green */
	{  60, 200, 240 },   /* prims   cyan */
	{ 230,  80, 230 },   /* misc    magenta */
	{ 240,  50,  50 },   /* gpu     red */
	{   0,   0,   0 },   /* vsync   black */
};
static int prof_peak = 0, prof_peak_acc = 0, prof_peak_n = 0;
static char *prof_draw(uint32_t *ot, char *nextpri) {
	TILE *t = (TILE *)nextpri;
	int pos = 0;                           /* h-blanks from the frame start */
	/* peak of the busy time (everything but the v-sync wait) over the last
	 * half second, updated every 0.5 s */
	int busy = 0;
	for (int i = 0; i < PROF_STAGES; i++)
		if (i != PROF_VSYNC) busy += prof_shown[i];
	if (busy > prof_peak_acc) prof_peak_acc = busy;
	if (++prof_peak_n >= 30) { prof_peak = prof_peak_acc; prof_peak_acc = 0; prof_peak_n = 0; }

	/* marks are added first: primitives added later to the same OT bucket
	 * are drawn earlier, so these end up on top of the stage tiles */
	{
		/* peak line (yellow, solid, across both columns) */
		int col = prof_peak / PROF_FRAME, in_col = prof_peak % PROF_FRAME;
		if (col <= 3) {
			setTile(t);
			setRGB0(t, 255, 255, 0);
			setXY0(t, PROF_X - 1, PROF_Y + (in_col * PROF_SCALE) / PROF_FRAME);
			setWH(t, (PROF_W + 2) * (col + 1), 1);
			addPrim(ot, t);
			t++;
		}
		/* half-frame and full-frame marks */
		for (int k = 1; k <= 2; k++) {
			int c = k == 2 ? 255 : 160;
			setTile(t);
			setRGB0(t, c, c, c);
			setXY0(t, PROF_X - 1, PROF_Y + (PROF_SCALE * k) / 2);
			setWH(t, PROF_W + 2, 1);
			addPrim(ot, t);
			t++;
		}
	}
	for (int i = 0; i < PROF_STAGES; i++) {
		int left = prof_shown[i];
		while (left > 0) {
			int col = pos / PROF_FRAME;
			int in_col = pos % PROF_FRAME;
			int n = left;
			if (in_col + n > PROF_FRAME) n = PROF_FRAME - in_col;   /* split at the frame boundary */
			if (col > 3) { pos += left; break; }
			int y0 = PROF_Y + (in_col * PROF_SCALE) / PROF_FRAME;
			int y1 = PROF_Y + ((in_col + n) * PROF_SCALE) / PROF_FRAME;
			if (y1 > y0) {
				setTile(t);
				setRGB0(t, prof_color[i][0], prof_color[i][1], prof_color[i][2]);
				setXY0(t, PROF_X + col * (PROF_W + 2), y0);
				setWH(t, PROF_W, y1 - y0);
				addPrim(ot, t);
				t++;
			}
			pos += n;
			left -= n;
		}
	}
	return (char *)t;
}

static void display(void) {
	DrawSync(0);
	prof_mark(PROF_GPU);
	VSync(0);
	prof_mark(PROF_VSYNC);
	prof_frame();
	db_active ^= 1;
	db_nextpri = db[db_active].p;
	ClearOTagR(db[db_active].ot, OT_LEN);
	PutDrawEnv(&db[db_active].draw);
	PutDispEnv(&db[db_active].disp);
	SetDispMask(1);
	DrawOTag(db[1 - db_active].ot + (OT_LEN - 1));
}

int main(void) {
	Model    model;
	Pose     pose;
	Camera   cam;
	Renderer renderer;
	IKState  ik;
	Walker   walker;
	static Fight fight;               /* two poses: keep it off the stack */
	static Model  fmodel[2];
	static Renderer frender[2];
	int mode_fight = 1;               /* boot straight into the fight */

	int cur_asset = 0;
	int model_yaw;
	int combo_used = 0;        /* START+SELECT fired: swallow the single-button releases */
	int dance = 0;             /* IK mode: hip driven by sines instead of the d-pad */
	int dance_t = 0;
	int shy = 0;               /* frames left of the "cloth fell" reaction */
	int pip = assets[0].pip;   /* face picture-in-picture on / off (SELECT + circle) */
	int clear_t = 0;           /* frames since the whole skirt was cut */
	int seen_cuts = 0;
	memset(&ik, 0, sizeof(ik));

	init_graphics();

	for (int i = 0; i < NUM_ASSETS; i++) {
		if (model_open(&model, assets[i].bin) < 0) {
			while (1) {
				FntPrint(-1, "BAD MODEL DATA %d\n", i);
				FntFlush(-1);
				display();
			}
		}
	}
	model_open(&model, assets[cur_asset].bin);
	renderer_init(&renderer, assets[cur_asset].tims);
	walker_load_sprite(monkey_tim);
	walker_reset(&walker, &model);
	renderer.cut = walker.tri_cut;
	model_yaw = assets[cur_asset].yaw;
	camera_init(&cam);
	/* fight mode: the two rigged characters (same animation set), textures
	 * on different VRAM pages, no lighting for speed */
	/* fight: the blonde against herself (one model, one texture, no lighting
	 * = the cheapest path); player 2 is tinted so they can be told apart */
	static const uint32_t *const fight_tims[MAX_TEX] = { character_tim, 0, 0 };
	model_open(&fmodel[0], character_bin);
	model_open(&fmodel[1], character_bin);
	renderer_init(&frender[0], fight_tims);
	frender[1] = frender[0];
	frender[0].shading = frender[1].shading = SHADE_NONE;
	frender[1].unlit_rgb = 0x00c07060;                          /* bluish */
	fight_init(&fight, &fmodel[0], &frender[0], &fmodel[1], &frender[1]);
	prof_init();
	pose_init(&pose, &model, find_anim(&model, "RUN"));

	InitPAD(pad_buff[0], 34, pad_buff[1], 34);
	StartPAD();
	ChangeClearPAD(0);

	uint16_t prev_btn = 0xffff;
	int frames = 0, fps = 0, fps_frames = 0, last_vsync = VSync(-1);

	while (1) {
		/* ---- input ------------------------------------------------------- */
		const PADTYPE *pad = (const PADTYPE *)pad_buff[0];
		uint16_t btn = 0xffff;
		int dyaw = 0, dpitch = 0, ddist = 0, dpan = 0, dfov = 0;
		if (pad->stat == 0 &&
		    (pad->type == PAD_ID_DIGITAL || pad->type == PAD_ID_ANALOG ||
		     pad->type == PAD_ID_ANALOG_STICK)) {
			btn = pad->btn;
			if (pad->type == PAD_ID_ANALOG || pad->type == PAD_ID_ANALOG_STICK) {
				int lx = (int)pad->ls_x - 128, ly = (int)pad->ls_y - 128;
				int ry = (int)pad->rs_y - 128;
				if (lx > 24 || lx < -24) dyaw   += lx / 4;
				if (ly > 24 || ly < -24) dpitch += ly / 4;
				if (ry > 24 || ry < -24) dpan   += ry / 2;
			}
		}
		uint16_t pressed  = (~btn) & prev_btn;  /* newly pressed this frame */
		uint16_t released = btn & (~prev_btn);  /* newly released this frame */
		uint16_t held     = ~btn;
		prev_btn = btn;

		if (held & PAD_SELECT) {
			/* SELECT + d-pad: orbit; SELECT + triangle/cross: camera up/down;
			 * SELECT + square: reset the cloth (digital pads have no sticks) */
			if (held & PAD_LEFT)     dyaw   -= 32;
			if (held & PAD_RIGHT)    dyaw   += 32;
			if (held & PAD_UP)       dpitch += 24;
			if (held & PAD_DOWN)     dpitch -= 24;
			if (held & PAD_TRIANGLE) dpan   -= 24;
			if (held & PAD_CROSS)    dpan   += 24;
			if ((pressed & PAD_SQUARE) && walker.nquads) {
				walker_reset(&walker, &model);
				renderer.cut = walker.tri_cut;
				clear_t = 0;
			}
			if (pressed & PAD_CIRCLE)
				pip ^= 1;
			if (held & (PAD_UP | PAD_DOWN | PAD_LEFT | PAD_RIGHT | PAD_TRIANGLE | PAD_CROSS | PAD_SQUARE | PAD_CIRCLE))
				combo_used = 1;               /* not a plain SELECT tap */
		} else {
			/* d-pad steers the monkey on the cloth grid (screen directions);
			 * without a grid it orbits the camera */
			if (model.hdr->nedges) {
				walker.dir_x = (held & PAD_LEFT) ? -1 : (held & PAD_RIGHT) ? 1 : 0;
				walker.dir_y = (held & PAD_UP) ? -1 : (held & PAD_DOWN) ? 1 : 0;
			} else {
				if (held & PAD_LEFT)  dyaw   -= 32;
				if (held & PAD_RIGHT) dyaw   += 32;
				if (held & PAD_UP)    dpitch += 24;
				if (held & PAD_DOWN)  dpitch -= 24;
			}
		}
		if (!(held & PAD_SELECT)) {
			if (held & PAD_TRIANGLE) ddist -= 96;
			if (held & PAD_CROSS)    ddist += 96;
			if (held & PAD_SQUARE)   dfov  += 8;
			if (held & PAD_CIRCLE)   dfov  -= 8;
		}
		if (held & PAD_L1) model_yaw += 32;
		model_yaw &= 4095;

		/* START + SELECT (both held, one just pressed) toggles IK mode; the
		 * single-button actions fire on release so the combo does not
		 * trigger them. */
		if ((pressed & (PAD_START | PAD_SELECT)) && (held & PAD_START) && (held & PAD_SELECT)) {
			combo_used = 1;
			mode_fight ^= 1;
			if (mode_fight) {
				renderer_init(&frender[0], fight_tims);          /* VRAM back to the fighters */
				frender[1] = frender[0];
				frender[0].shading = frender[1].shading = SHADE_NONE;
				frender[1].unlit_rgb = 0x00c07060;
				fight_init(&fight, &fmodel[0], &frender[0], &fmodel[1], &frender[1]);
			} else {
				renderer_init(&renderer, assets[cur_asset].tims);
				walker_load_sprite(monkey_tim);
				renderer.cut = walker.tri_cut;
			}
		}
		if (!mode_fight && (pressed & PAD_R1)) {
			if (ik.active) { ik_leave(&ik, &pose); dance = 0; }
			else           { ik_enter(&ik, &model, &pose); dance = 1; dance_t = 0; }
		}
		if (mode_fight) {
			/* ---- fight mode frame ------------------------------------ */
			prof_mark(PROF_INPUT);
			fight_update(&fight, &cam, fps >= 45 ? 60 : 30);
			gte_SetGeomOffset(CENTERX, CENTERY);
			gte_SetGeomScreen(cam.fov);
			prof_mark(PROF_POSE);
			db_nextpri = draw_floor(&cam, db[db_active].ot, db_nextpri);
			db_nextpri = fight_draw(&fight, &cam, db[db_active].ot, db_nextpri);
			db_nextpri = prof_draw(db[db_active].ot, db_nextpri);
			FntPrint(fnt_fps, "FPS %2d\n", fps);
			FntFlush(fnt_fps);
			prof_mark(PROF_MISC);
			display();
			frames++;
			fps_frames++;
			{
				int now = VSync(-1);
				if (now - last_vsync >= 60) {
					fps = fps_frames * 60 / (now - last_vsync);
					fps_frames = 0;
					last_vsync = now;
				}
			}
			continue;
		}
		if (ik.active && dance) {
			/* hip on a slow horizontal circle (5 s per turn) with a
			 * double-time bounce; the hands sway on their own sines */
			#define DANCE_SIN(period, phase) dance_sin(dance_t, period, phase)
			pose.hip_offset.vx = (220 * DANCE_SIN(300, 0)) >> 12;
			pose.hip_offset.vz = (220 * DANCE_SIN(300, 1024)) >> 12;
			pose.hip_offset.vy = 120 + ((110 * DANCE_SIN(150, 0)) >> 12);   /* down = positive */
			for (int i = 0; i < 2; i++) {               /* IK_ARM_L, IK_ARM_R */
				if (!ik.has[i]) continue;
				int ph = i ? 2048 : 0;                  /* arms in counter-phase */
				ik.target[i].vx = ik.base[i].vx + ((90  * DANCE_SIN(300, ph + 512)) >> 12);
				ik.target[i].vy = ik.base[i].vy + ((140 * DANCE_SIN(150, ph + 1024)) >> 12) - 60;
				ik.target[i].vz = ik.base[i].vz + ((160 * DANCE_SIN(300, ph)) >> 12);
				if (shy) {
					/* cloth fell: hands rush to the front of the skirt */
					int k = shy > 40 ? (60 - shy) * 4096 / 20 : shy * 4096 / 40;   /* ease in / out */
					if (k > 4096) k = 4096;
					VECTOR hip = vec(pose.world[model.ik->hip].t[0], pose.world[model.ik->hip].t[1],
					                 pose.world[model.ik->hip].t[2]);
					int side = i ? -1 : 1;
					VECTOR goal = vec(hip.vx + side * 140, hip.vy + 250, hip.vz - 420);
					ik.target[i].vx += ((goal.vx - ik.target[i].vx) * k) >> 12;
					ik.target[i].vy += ((goal.vy - ik.target[i].vy) * k) >> 12;
					ik.target[i].vz += ((goal.vz - ik.target[i].vz) * k) >> 12;
				}
			}
			dance_t++;
		}
		if (walker.cut_event != seen_cuts) { seen_cuts = walker.cut_event; shy = 60; }
		if (shy) shy--;
		if (walker.nquads && walker.ncut >= walker.nquads) {
			if (++clear_t > 240) {                      /* 4 s of CLEAR, then a new skirt */
				walker_reset(&walker, &model);
				renderer.cut = walker.tri_cut;
				clear_t = 0;
			}
		}
		if (!(held & (PAD_START | PAD_SELECT)))
			combo_used = 0;
		if (!ik.active) {
			if (pressed & PAD_L2)
				pose_set_anim(&pose, pose.anim - 1);
			if (pressed & PAD_R2)
				pose_set_anim(&pose, pose.anim + 1);
		}
		if (pressed & PAD_L3)
			pose.playing ^= 1;
		if ((released & PAD_START) && !combo_used) {
			int shading = renderer.shading;
			ik_leave(&ik, &pose);
			dance = 0;
			cur_asset = (cur_asset + 1) % NUM_ASSETS;
			model_open(&model, assets[cur_asset].bin);
			renderer_init(&renderer, assets[cur_asset].tims);
			renderer.shading = shading;
			walker_load_sprite(monkey_tim);      /* renderer_init reloads VRAM */
			walker_reset(&walker, &model);
			renderer.cut = walker.tri_cut;
			model_yaw = assets[cur_asset].yaw;
			camera_init(&cam);                         /* back in front of the character */
			pip = assets[cur_asset].pip;
			pose_init(&pose, &model, find_anim(&model, "RUN"));
			if (assets[cur_asset].look_at) {          /* the schoolgirl starts dancing */
				ik_enter(&ik, &model, &pose);
				dance = 1;
				dance_t = 0;
			}
		}

		camera_orbit(&cam, dyaw, dpitch, ddist);
		camera_pan(&cam, dpan);
		camera_zoom(&cam, dfov);
		camera_update(&cam, model_yaw);
		gte_SetGeomScreen(cam.fov);
		prof_mark(PROF_INPUT);

		/* ---- animation --------------------------------------------------- */
		pose_step(&pose, 60);
		pose_eval(&pose);
		if (ik.active)
			ik_apply(&ik, &model, &pose, &cam);          /* includes the look-at */
		else if (assets[cur_asset].look_at)
			ik_look_at(&model, &pose, &cam);
		prof_mark(PROF_POSE);

		/* ---- render ------------------------------------------------------ */
		db_nextpri = render_model(&renderer, &model, &pose, &cam, db[db_active].ot, db_nextpri);
		db_nextpri = draw_floor(&cam, db[db_active].ot, db_nextpri);
		db_nextpri = walker_draw(&walker, &model, &renderer, render_get_cache(), db[db_active].ot, db_nextpri);
		db_nextpri = prof_draw(db[db_active].ot, db_nextpri);
		db_nextpri = draw_texture_preview(&renderer, db[db_active].ot, db_nextpri);
		if (pip) {
			/* the OT built now is drawn after the buffer swap, i.e. with the
			 * other draw environment: its clip origin is the one DR_AREA needs */
			db_nextpri = draw_face_pip(&renderer, &model, &pose, model_yaw, &db[1 - db_active].draw,
			                           db[db_active].ot, db_nextpri);
			gte_SetGeomOffset(CENTERX, CENTERY);
			gte_SetGeomScreen(cam.fov);
		}

		const ModelAnim *a = &model.anims[pose.anim];
		/* game HUD (top) */
		if (walker.nquads)
			FntPrint(fnt_hud, "CLOTH %d/%d %s\n", walker.ncut, walker.nquads,
			         clear_t ? "  * CLEAR *" : "");
		FntFlush(fnt_hud);
		/* debug (bottom) */
		if (ik.active && !dance)
			FntPrint(fnt_dbg, "IK MODE HIP %d,%d,%d  (%s %d)\n", pose.hip_offset.vx,
			         pose.hip_offset.vy, pose.hip_offset.vz, a->name, pose.frame);
		else if (!ik.active)
			FntPrint(fnt_dbg, "ANIM %d/%d %-15s %3d/%3d %s\n", pose.anim + 1, model.hdr->nanims, a->name,
			         pose.frame, a->nframes, pose.playing ? "" : "PAUSE");
		else
			FntPrint(fnt_dbg, "\n");
		FntPrint(fnt_dbg, "%s TRI%d CPU%d FOV%d Y%d/%d\n",
		         renderer.shading == SHADE_FLAT ? "FLAT" : "GOUR", renderer.tris_drawn,
		         prof_shown[PROF_INPUT] + prof_shown[PROF_POSE] + prof_shown[PROF_VERTS] +
		         prof_shown[PROF_PRIMS] + prof_shown[PROF_MISC], cam.fov, model_yaw, cam.yaw);
		FntFlush(fnt_dbg);
		FntPrint(fnt_fps, "FPS %2d\n", fps);
		FntFlush(fnt_fps);
		prof_mark(PROF_MISC);

		display();

		/* ---- fps counter ------------------------------------------------- */
		frames++;
		fps_frames++;
		int now = VSync(-1);
		if (now - last_vsync >= 60) {
			fps = fps_frames * 60 / (now - last_vsync);
			fps_frames = 0;
			last_vsync = now;
		}
	}
	return 0;
}
