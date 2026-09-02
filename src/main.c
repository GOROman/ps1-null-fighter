/* NULL FIGHTER - skinned FBX character viewer for PlayStation 1
 *
 * Controls
 *   D-pad / left stick   orbit camera
 *   Right stick up/down  move camera up / down
 *   SELECT + D-pad       up/down: move camera up / down, left/right: dolly
 *   Triangle / Cross     dolly in / out (distance)
 *   Square / Circle      zoom in / out (field of view)
 *   L1                   rotate model (yaw)
 *   R1                   IK mode on / off
 *   L2 / R2              previous / next animation
 *   SELECT               toggle flat / Gouraud shading (on release)
 *   START                switch character (on release)
 *   START + SELECT       IK mode on / off (same as R1)
 *   L3 (stick click)     pause / resume animation
 *
 * IK mode: the pose is frozen, hands and feet stay where they were (two
 * bone IK), the d-pad moves the hip (left/right = X, up/down = height,
 * L2/R2 = forward/back) and the head looks at the camera.
 *
 * The bar on the left is a frame profiler in h-blank units (a full frame
 * is the white line): yellow input, orange pose, green vertex transform,
 * cyan primitive setup, magenta text/overlay, red waiting for the GPU,
 * grey waiting for v-blank.
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

#define SCREEN_XRES   320
#define SCREEN_YRES   240
#define CENTERX       (SCREEN_XRES >> 1)
#define CENTERY       (SCREEN_YRES >> 1)
#define PACKET_LEN    (128 * 1024)

typedef struct {
	DISPENV  disp;
	DRAWENV  draw;
	uint32_t ot[OT_LEN];
	char     p[PACKET_LEN];
} DB;

static DB   db[2];
static int  db_active = 0;
static char *db_nextpri;

extern const uint32_t schoolgirl_bin[], schoolgirl_tim[], schoolgirl_face_tim[];
extern const uint32_t character_bin[],  character_tim[];

typedef struct {
	const uint32_t *bin, *tim, *face_tim;   /* face_tim: NULL if single texture */
	int yaw;                                /* model yaw that faces the camera */
} ModelAsset;

static const ModelAsset assets[] = {
	{ character_bin,  character_tim,  0,                   2048 },   /* boots on the animated character */
	{ schoolgirl_bin, schoolgirl_tim, schoolgirl_face_tim, 0 },
};
#define NUM_ASSETS ((int)(sizeof(assets) / sizeof(assets[0])))

static uint8_t pad_buff[2][34];

uint16_t prof_stage[PROF_STAGES];
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
	FntOpen(12, 8, SCREEN_XRES - 20, 64, 0, 256);
}

/* Draw the 256x256 model texture scaled down in the bottom-right corner so
 * the TIM conversion / UV mapping can be checked visually.  ot[0] is drawn
 * last (reversed table), so the quad lands on top of the model. */
#define TEXPREV_SIZE 72
static char *draw_texture_preview(const Renderer *r, uint32_t *ot, char *nextpri) {
	POLY_FT4 *q = (POLY_FT4 *)nextpri;
	for (int i = 0; i < r->ntex; i++) {
		int x0 = SCREEN_XRES - 8 - TEXPREV_SIZE * (r->ntex - i) - 4 * (r->ntex - 1 - i);
		int y0 = SCREEN_YRES - 8 - TEXPREV_SIZE;
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

/* Floor: wire grid on the Y = 0 plane (model space, feet level), X axis red,
 * Z axis blue.  Lines with an end point behind the camera are dropped. */
#define GRID_STEP  512
#define GRID_N     8                      /* lines from -N..N in each direction */
#define GRID_NEAR  128                    /* near plane, camera space */

/* Liang-Barsky clip of a 2D segment to [-32, SCREEN_XRES+32] x [-32, SCREEN_YRES+32];
 * returns 0 if nothing remains */
static int clip_line_2d(int *x0, int *y0, int *x1, int *y1) {
	const int xmin = -32, xmax = SCREEN_XRES + 32, ymin = -32, ymax = SCREEN_YRES + 32;
	int64_t dx = *x1 - *x0, dy = *y1 - *y0;
	int64_t t0 = 0, t1 = 4096;            /* parameter range, 4096 = 1.0 */
	int64_t p[4] = { -dx, dx, -dy, dy };
	int64_t q[4] = { *x0 - xmin, xmax - *x0, *y0 - ymin, ymax - *y0 };
	for (int i = 0; i < 4; i++) {
		if (p[i] == 0) {
			if (q[i] < 0) return 0;
			continue;
		}
		int64_t t = (q[i] << 12) / p[i];
		if (p[i] < 0) { if (t > t1) return 0; if (t > t0) t0 = t; }
		else          { if (t < t0) return 0; if (t < t1) t1 = t; }
	}
	int nx0 = *x0 + (int)((dx * t0) >> 12), ny0 = *y0 + (int)((dy * t0) >> 12);
	int nx1 = *x0 + (int)((dx * t1) >> 12), ny1 = *y0 + (int)((dy * t1) >> 12);
	*x0 = nx0; *y0 = ny0; *x1 = nx1; *y1 = ny1;
	return 1;
}
static char *draw_floor(const Camera *cam, uint32_t *ot, char *nextpri) {
	LINE_F2 *l = (LINE_F2 *)nextpri;
	const MATRIX *v = &cam->view;
	for (int axis = 0; axis < 2; axis++) {
		for (int i = -GRID_N; i <= GRID_N; i++) {
			VECTOR p[2], c[2];
			if (axis == 0) {   /* lines along X at z = i*step */
				p[0] = vec(-GRID_N * GRID_STEP, 0, i * GRID_STEP);
				p[1] = vec( GRID_N * GRID_STEP, 0, i * GRID_STEP);
			} else {           /* lines along Z at x = i*step */
				p[0] = vec(i * GRID_STEP, 0, -GRID_N * GRID_STEP);
				p[1] = vec(i * GRID_STEP, 0,  GRID_N * GRID_STEP);
			}
			/* to camera space (GTE), then clip against the near plane */
			for (int k = 0; k < 2; k++) {
				ApplyMatrixLV((MATRIX *)v, &p[k], &c[k]);
				c[k].vx += v->t[0]; c[k].vy += v->t[1]; c[k].vz += v->t[2];
			}
			if (c[0].vz < GRID_NEAR && c[1].vz < GRID_NEAR)
				continue;
			for (int k = 0; k < 2; k++) {
				if (c[k].vz < GRID_NEAR) {
					int o = 1 - k;
					int32_t t = ((GRID_NEAR - c[k].vz) << 12) / (c[o].vz - c[k].vz);   /* 0..4096 */
					c[k].vx += (int32_t)(((int64_t)(c[o].vx - c[k].vx) * t) >> 12);
					c[k].vy += (int32_t)(((int64_t)(c[o].vy - c[k].vy) * t) >> 12);
					c[k].vz = GRID_NEAR;
				}
			}
			int sx[2], sy[2];
			for (int k = 0; k < 2; k++) {
				sx[k] = CENTERX + (int)(((int64_t)c[k].vx * cam->fov) / c[k].vz);
				sy[k] = CENTERY + (int)(((int64_t)c[k].vy * cam->fov) / c[k].vz);
			}
			/* the GPU drops primitives wider than 1023 / taller than 511, so
			 * clip the 2D segment to (a little more than) the screen */
			if (!clip_line_2d(&sx[0], &sy[0], &sx[1], &sy[1]))
				continue;
			int otz = ((c[0].vz + c[1].vz) * 3) >> (3 + OTZ_SHIFT);   /* same scale as the model */
			if (otz >= OT_LEN) otz = OT_LEN - 1;
			if (otz <= 0) otz = 1;
			setLineF2(l);
			if (i == 0)
				setRGB0(l, axis == 0 ? 200 : 60, 60, axis == 0 ? 60 : 220);
			else
				setRGB0(l, 70, 74, 90);
			setXY2(l, sx[0], sy[0], sx[1], sy[1]);
			addPrim(ot + otz, l);
			l++;
		}
	}
	return (char *)l;
}

/* Profiler bar: stages stacked top to bottom at the left edge, PROF_SCALE
 * pixels per frame (263 h-blanks).  A white line marks one full frame. */
#define PROF_X      2
#define PROF_W      4
#define PROF_Y      8
#define PROF_SCALE  200
static const uint8_t prof_color[PROF_STAGES][3] = {
	{ 255, 220,  40 },   /* input   yellow */
	{ 255, 160,  40 },   /* pose    orange */
	{  60, 220,  80 },   /* verts   green */
	{  60, 200, 240 },   /* prims   cyan */
	{ 230,  80, 230 },   /* misc    magenta */
	{ 240,  50,  50 },   /* gpu     red */
	{  90,  90, 100 },   /* vsync   grey */
};
static char *prof_draw(uint32_t *ot, char *nextpri) {
	TILE *t = (TILE *)nextpri;
	int y = PROF_Y;
	for (int i = 0; i < PROF_STAGES; i++) {
		int h = (prof_stage[i] * PROF_SCALE) / 263;
		if (y + h > SCREEN_YRES - 2) h = SCREEN_YRES - 2 - y;
		if (h <= 0) continue;
		setTile(t);
		setRGB0(t, prof_color[i][0], prof_color[i][1], prof_color[i][2]);
		setXY0(t, PROF_X, y);
		setWH(t, PROF_W, h);
		addPrim(ot, t);
		t++;
		y += h;
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
	return (char *)t;
}

static void display(void) {
	DrawSync(0);
	prof_mark(PROF_GPU);
	VSync(0);
	prof_mark(PROF_VSYNC);
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

	int cur_asset = 0;
	int model_yaw;
	int combo_used = 0;        /* START+SELECT fired: swallow the single-button releases */
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
	renderer_init(&renderer, assets[cur_asset].tim, assets[cur_asset].face_tim);
	model_yaw = assets[cur_asset].yaw;
	camera_init(&cam);
	prof_init();
	pose_init(&pose, &model, 0);

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
			/* SELECT + d-pad: camera height / distance (digital pads have no right stick) */
			if (held & PAD_UP)    dpan  -= 24;
			if (held & PAD_DOWN)  dpan  += 24;
			if (held & PAD_LEFT)  ddist += 48;
			if (held & PAD_RIGHT) ddist -= 48;
			if (held & (PAD_UP | PAD_DOWN | PAD_LEFT | PAD_RIGHT))
				combo_used = 1;               /* do not toggle shading on release */
		} else if (ik.active) {
			/* d-pad drives the hip bone */
			if (held & PAD_LEFT)  pose.hip_offset.vx -= 24;
			if (held & PAD_RIGHT) pose.hip_offset.vx += 24;
			if (held & PAD_UP)    pose.hip_offset.vy -= 24;   /* y is down */
			if (held & PAD_DOWN)  pose.hip_offset.vy += 24;
			if (held & PAD_L2)    pose.hip_offset.vz -= 24;
			if (held & PAD_R2)    pose.hip_offset.vz += 24;
		} else {
			if (held & PAD_LEFT)  dyaw   -= 32;
			if (held & PAD_RIGHT) dyaw   += 32;
			if (held & PAD_UP)    dpitch += 24;
			if (held & PAD_DOWN)  dpitch -= 24;
		}
		if (held & PAD_TRIANGLE) ddist -= 96;
		if (held & PAD_CROSS)    ddist += 96;
		if (held & PAD_SQUARE) dfov += 8;
		if (held & PAD_CIRCLE) dfov -= 8;
		if (held & PAD_L1) model_yaw += 32;
		model_yaw &= 4095;

		/* START + SELECT (both held, one just pressed) toggles IK mode; the
		 * single-button actions fire on release so the combo does not
		 * trigger them. */
		if (((pressed & (PAD_START | PAD_SELECT)) && (held & PAD_START) && (held & PAD_SELECT)) ||
		    (pressed & PAD_R1)) {
			if (held & PAD_START) combo_used = 1;
			if (ik.active) ik_leave(&ik, &pose);
			else           ik_enter(&ik, &model, &pose);
		}
		if (!(held & (PAD_START | PAD_SELECT)))
			combo_used = 0;
		if ((released & PAD_SELECT) && !combo_used)
			renderer.shading ^= 1;
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
			cur_asset = (cur_asset + 1) % NUM_ASSETS;
			model_open(&model, assets[cur_asset].bin);
			renderer_init(&renderer, assets[cur_asset].tim, assets[cur_asset].face_tim);
			renderer.shading = shading;
			model_yaw = assets[cur_asset].yaw;
			pose_init(&pose, &model, 0);
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
		ik_apply(&ik, &model, &pose, &cam);
		prof_mark(PROF_POSE);

		/* ---- render ------------------------------------------------------ */
		db_nextpri = render_model(&renderer, &model, &pose, &cam, db[db_active].ot, db_nextpri);
		db_nextpri = draw_floor(&cam, db[db_active].ot, db_nextpri);
		db_nextpri = draw_texture_preview(&renderer, db[db_active].ot, db_nextpri);
		db_nextpri = prof_draw(db[db_active].ot, db_nextpri);

		const ModelAnim *a = &model.anims[pose.anim];
		FntPrint(-1, "NULL FIGHTER  %s\n", renderer.shading == SHADE_FLAT ? "FLAT" : "GOURAUD");
		if (ik.active)
			FntPrint(-1, "IK MODE  HIP %d,%d,%d  (%s %d)\n", pose.hip_offset.vx, pose.hip_offset.vy,
			         pose.hip_offset.vz, a->name, pose.frame);
		else
			FntPrint(-1, "ANIM %d/%d %-15s %3d/%3d %s\n", pose.anim + 1, model.hdr->nanims, a->name,
			         pose.frame, a->nframes, pose.playing ? "" : "PAUSE");
		FntPrint(-1, "TRIS %d  FPS %d  CPU %d HB  FOV %d\n", renderer.tris_drawn, fps,
		         prof_stage[PROF_INPUT] + prof_stage[PROF_POSE] + prof_stage[PROF_VERTS] +
		         prof_stage[PROF_PRIMS] + prof_stage[PROF_MISC], cam.fov);
		FntFlush(-1);
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
