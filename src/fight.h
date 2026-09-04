/* Fighting game mode: two rigged characters (same animation set) fight
 * each other on the x axis (CPU or pad controlled), with life bars, a round timer,
 * ROUND / FIGHT / K.O. announcements, an automatic camera and a table of
 * 32 moves (tools/moves.py -> moves_table.h). */
#ifndef FIGHT_H
#define FIGHT_H

#include <stdint.h>
#include <psxgpu.h>
#include "model.h"
#include "anim.h"
#include "render.h"
#include "camera.h"

enum { FS_IDLE, FS_APPROACH, FS_WALK, FS_RETREAT, FS_BACKSTEP, FS_ATTACK, FS_HIT, FS_DOWN, FS_KO, FS_WIN, FS_GUARD };
enum { CAT_PUNCH = 0, CAT_KICK, CAT_SPECIAL };
enum { LIMB_HAND = 0, LIMB_FOOT, LIMB_BODY };
enum { H_ANY = 0, H_HIGH, H_LOW };
#define MF_LAUNCH    1
#define MF_KNOCKDOWN 2
#define MF_NOLOOK    4

typedef struct {
	const char *name, *clip;
	int cat, speed, hit_from, hit_to, dmg, kb, stop, limb, height, travel, reach, rehit, flags, pad;
} MoveDef;

#include "moves_table.h"

typedef struct {
	const Model *model;
	Renderer    *renderer;
	Pose         pose;
	int x, z, yaw;             /* world placement (PS1 units; yaw 4096 = 360 deg) */
	int hp, hp_disp;
	int guard_t, hits_taken;   /* guarding frames left; consecutive hits received */
	int state, cooldown;
	int move;                  /* current attack (index into MOVES) */
	int hit_done, rehit_at, yaw_corr;
	int combo[4], combo_len, combo_i;
	int kb;                    /* knock back velocity, decays */
	int last_backstep, down_phase, plan_special, special_cd;
	int move_anim[NUM_MOVES];  /* animation index per move */
	int human;                 /* 1: driven by a pad (fight_input) instead of the AI */
	uint16_t in_held, in_pressed;   /* pad state for this frame (fight_input) */
	int anim_idle, anim_run, anim_hit, anim_ko, anim_win, anim_jump, anim_fall, anim_guard;
} Fighter;

#define MAX_FX 6
typedef struct { int active, t, x, y, z; } HitFx;

enum { FP_ROUND, FP_FIGHT, FP_FIGHTING, FP_KO, FP_END };

typedef struct {
	Fighter f[2];
	int phase, phase_t;
	int round, wins[2];
	int timer;
	int winner;
	int t;
	uint32_t rng;
	int cam_yaw, cam_dist, cam_pitch;
	VECTOR cam_target;
	HitFx fx[MAX_FX];
	int shake, hitstop;
	int counter_t, counter_side;        /* COUNTER! popup */
	int ringout;                        /* round ended by a ring out */
	int name_t, name_move, name_side;   /* move name popup */
	int demo;                           /* title screen backdrop: no HUD */
	int match_over;                     /* set when a best-of-3 match has ended */
} Fight;

void fight_init(Fight *fg, const Model *m0, Renderer *r0, const Model *m1, Renderer *r1);
/* which fighters are pad controlled (call after fight_init) */
void fight_set_players(Fight *fg, int p0_human, int p1_human);
/* pad state for fighter `side` this frame: `held` / `pressed` are ~btn masks (PAD_*).
 * Virtua Fighter style P / K / G:
 *   D-pad left/right   walk towards / away (screen direction)
 *   Triangle (P)       punch: jab / straight (fwd) / hook (back) / uppercut (down)
 *   Circle (K)         kick: high kick / roundhouse (fwd) / back kick (back) / sweep (down)
 *   P + K              special: SBK / dragon kick (fwd) / hurricane (back) / slide (down)
 *   Square (G, hold)   guard
 *   an attack button pressed during an attack chains the next move (up to 4) */
void fight_input(Fight *fg, int side, uint16_t held, uint16_t pressed);
void fight_update(Fight *fg, Camera *cam, int hz);
char *fight_draw(Fight *fg, const Camera *cam, uint32_t *ot, char *nextpri);
/* 5x7 bitmap text (0-9 A-Z ! .), `scale` px per font pixel, centred at cx, OT bucket 0 */
char *fight_text(const char *s, int cx, int y, int scale, int r, int g, int b, uint32_t *ot, char *nextpri);

#endif
