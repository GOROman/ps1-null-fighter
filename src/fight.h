/* Fighting game mode: two rigged characters (same animation set) fight
 * each other CPU vs CPU on the x axis, with life bars, a round timer,
 * ROUND / FIGHT / K.O. announcements and an automatic camera. */
#ifndef FIGHT_H
#define FIGHT_H

#include <stdint.h>
#include <psxgpu.h>
#include "model.h"
#include "anim.h"
#include "render.h"
#include "camera.h"

enum { FS_IDLE, FS_APPROACH, FS_WALK, FS_RETREAT, FS_BACKSTEP, FS_ATTACK, FS_HIT, FS_DOWN, FS_KO, FS_WIN };
enum { FA_PUNCH, FA_KICK, FA_SPECIAL };

typedef struct {
	const Model *model;
	Renderer    *renderer;
	Pose         pose;
	int x, z, yaw;             /* world placement (PS1 units; yaw 4096 = 360 deg) */
	int hp;                    /* 0..100 */
	int state, attack, hit_done, cooldown;
	int combo[4], combo_len, combo_i;   /* queued attack chain (FA_*) */
	int kb;                    /* knock back velocity (units per frame, signed), decays */
	int last_backstep;         /* the previous idle decision was a backstep: do not chain them */
	int down_phase;            /* 0: falling, 1: getting up (fall clip played backwards) */
	int hp_disp;               /* displayed hp (drains slowly towards hp) */
	int anim_idle, anim_run, anim_punch, anim_kick, anim_special, anim_hit, anim_ko, anim_win, anim_jump, anim_fall;
} Fighter;

#define MAX_FX 6
typedef struct { int active, t, x, y, z; } HitFx;   /* world position, age */

enum { FP_ROUND, FP_FIGHT, FP_FIGHTING, FP_KO, FP_END };

typedef struct {
	Fighter f[2];
	int phase, phase_t;
	int round, wins[2];
	int timer;                 /* frames left in the round */
	int winner;                /* -1 while fighting */
	int t;                     /* global frame counter */
	uint32_t rng;
	/* camera smoothing */
	int cam_yaw, cam_dist, cam_pitch;
	VECTOR cam_target;
	HitFx fx[MAX_FX];
	int shake;                 /* camera shake frames after a hit */
	int hitstop;               /* frames both fighters freeze after a hit */
} Fight;

void fight_init(Fight *fg, const Model *m0, Renderer *r0, const Model *m1, Renderer *r1);
/* one frame of AI / animation / camera; writes the auto camera into cam.
 * hz: the actual frame rate (60 or 30) so motion and animation keep real-time speed */
void fight_update(Fight *fg, Camera *cam, int hz);
/* both fighters, their shadows and the HUD */
char *fight_draw(Fight *fg, const Camera *cam, uint32_t *ot, char *nextpri);

#endif
