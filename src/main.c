/* NULL FIGHTER - skinned FBX character viewer for PlayStation 1
 *
 * Controls
 *   D-pad                steer the monkey on the cloth grid (Dancing Eyes style)
 *   Left stick           orbit camera
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
 * L2/R2 = forward/back) and the head looks at the camera.  The viewer
 * boots in IK mode with the hip "dancing" on a sine circle; touching the
 * d-pad takes over, R1 (re-enter) restarts the dance.
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
#include "walker.h"

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

extern const uint32_t schoolgirl_bin[], schoolgirl_tim[], schoolgirl_face_tim[], schoolgirl_skirt_tim[];
extern const uint32_t character_bin[],  character_tim[];
extern const uint32_t monkey_tim[];

typedef struct {
	const uint32_t *bin;
	const uint32_t *tims[MAX_TEX];          /* body, face, skirt (NULL = unused) */
	int yaw;                                /* model yaw that faces the camera */
	int look_at;                            /* head always turned towards the camera */
} ModelAsset;

static const ModelAsset assets[] = {
	{ schoolgirl_bin, { schoolgirl_tim, schoolgirl_face_tim, schoolgirl_skirt_tim }, 0, 1 },
	{ character_bin,  { character_tim, 0, 0 }, 2048, 0 },   /* boots on the animated character */
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

/* sin(2 pi (t / period + phase / 4096)) in 4096 units, via the GTE rotation matrix */
static int dance_sin(int t, int period, int phase) {
	SVECTOR ang = { 0, ((t * 4096) / period + phase) & 4095, 0, 0 };
	MATRIX rm;
	RotMatrix(&ang, &rm);                              /* m[0][2] = sin(y angle) */
	return rm.m[0][2];
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
	int32_t dx = *x1 - *x0, dy = *y1 - *y0;
	int32_t t0 = 0, t1 = 4096;            /* parameter range, 4096 = 1.0 */
	int32_t p[4] = { -dx, dx, -dy, dy };
	int32_t q[4] = { *x0 - xmin, xmax - *x0, *y0 - ymin, ymax - *y0 };
	for (int i = 0; i < 4; i++) {
		if (p[i] == 0) {
			if (q[i] < 0) return 0;
			continue;
		}
		int32_t t = (q[i] << 12) / p[i];
		if (p[i] < 0) { if (t > t1) return 0; if (t > t0) t0 = t; }
		else          { if (t < t0) return 0; if (t < t1) t1 = t; }
	}
	int nx0 = *x0 + ((dx * t0) >> 12), ny0 = *y0 + ((dy * t0) >> 12);
	int nx1 = *x0 + ((dx * t1) >> 12), ny1 = *y0 + ((dy * t1) >> 12);
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
					c[k].vx += (int32_t)(((int32_t)(c[o].vx - c[k].vx) * t) >> 12);
					c[k].vy += (int32_t)(((int32_t)(c[o].vy - c[k].vy) * t) >> 12);
					c[k].vz = GRID_NEAR;
				}
			}
			int sx[2], sy[2];
			for (int k = 0; k < 2; k++) {
				sx[k] = CENTERX + (int)(((int32_t)c[k].vx * cam->fov) / c[k].vz);
				sy[k] = CENTERY + (int)(((int32_t)c[k].vy * cam->fov) / c[k].vz);
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
	Walker   walker;

	int cur_asset = 0;
	int model_yaw;
	int combo_used = 0;        /* START+SELECT fired: swallow the single-button releases */
	int dance = 0;             /* IK mode: hip driven by sines instead of the d-pad */
	int dance_t = 0;
	int shy = 0;               /* frames left of the "cloth fell" reaction */
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
	prof_init();
	pose_init(&pose, &model, 0);
	ik_enter(&ik, &model, &pose);
	dance = 1;

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
		if (held & PAD_TRIANGLE) ddist -= 96;
		if (held & PAD_CROSS)    ddist += 96;
		if ((pressed & PAD_CROSS) && walker.nquads) {   /* reset the cloth */
			walker_reset(&walker, &model);
			renderer.cut = walker.tri_cut;
			clear_t = 0;
		}
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
			if (ik.active) { ik_leave(&ik, &pose); dance = 0; }
			else           { ik_enter(&ik, &model, &pose); dance = 1; dance_t = 0; }
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
			dance = 0;
			cur_asset = (cur_asset + 1) % NUM_ASSETS;
			model_open(&model, assets[cur_asset].bin);
			renderer_init(&renderer, assets[cur_asset].tims);
			renderer.shading = shading;
			walker_load_sprite(monkey_tim);      /* renderer_init reloads VRAM */
			walker_reset(&walker, &model);
			renderer.cut = walker.tri_cut;
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

		const ModelAnim *a = &model.anims[pose.anim];
		FntPrint(-1, "NULL FIGHTER  %s\n", renderer.shading == SHADE_FLAT ? "FLAT" : "GOURAUD");
		if (ik.active)
			FntPrint(-1, "IK %s HIP %d,%d,%d  (%s %d)\n", dance ? "DANCE" : "MODE ", pose.hip_offset.vx,
			         pose.hip_offset.vy, pose.hip_offset.vz, a->name, pose.frame);
		else
			FntPrint(-1, "ANIM %d/%d %-15s %3d/%3d %s\n", pose.anim + 1, model.hdr->nanims, a->name,
			         pose.frame, a->nframes, pose.playing ? "" : "PAUSE");
		FntPrint(-1, "TRIS %d  FPS %d  CPU %d HB  FOV %d\n", renderer.tris_drawn, fps,
		         prof_stage[PROF_INPUT] + prof_stage[PROF_POSE] + prof_stage[PROF_VERTS] +
		         prof_stage[PROF_PRIMS] + prof_stage[PROF_MISC], cam.fov);
		if (walker.nquads)
			FntPrint(-1, "CLOTH %d/%d %s\n", walker.ncut, walker.nquads,
			         clear_t ? "  * CLEAR *" : "");
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
