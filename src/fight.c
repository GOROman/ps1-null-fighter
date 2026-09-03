#include <string.h>
#include <psxgpu.h>
#include <psxgte.h>
#include <inline_c.h>
#include "fight.h"
#include "ik.h"
#include "fixmath.h"

#define SCREEN_XRES 320
#define SCREEN_YRES 240
#define CENTERX     (SCREEN_XRES >> 1)
#define CENTERY     (SCREEN_YRES >> 1)

#define ROUND_FRAMES   (60 * 60)      /* 60 s */
#define START_X        1500           /* fighters start at +-START_X */
#define REACH_PUNCH    1600
#define REACH_KICK     1450
#define REACH_SPECIAL  1650
#define APPROACH_STOP  1200
#define MIN_DIST       1000
#define RUN_SPEED      44
#define WALK_SPEED     20
#define BACKSTEP_SPEED 48
#define ATTACK_HIT_AT  40             /* % of the attack animation where it lands */
#define FLOOR_OTZ      (OT_LEN - 1)

/* ---------------------------------------------------------------------- */
/* 5x7 bitmap font for the big announcements (0-9 A-Z ! .)                  */
/* ---------------------------------------------------------------------- */
static const char *glyph(char c) {
	static const char *G[] = {
		/* 0 */ "01110" "10001" "10011" "10101" "11001" "10001" "01110",
		/* 1 */ "00100" "01100" "00100" "00100" "00100" "00100" "01110",
		/* 2 */ "01110" "10001" "00001" "00010" "00100" "01000" "11111",
		/* 3 */ "11111" "00010" "00100" "00010" "00001" "10001" "01110",
		/* 4 */ "00010" "00110" "01010" "10010" "11111" "00010" "00010",
		/* 5 */ "11111" "10000" "11110" "00001" "00001" "10001" "01110",
		/* 6 */ "00110" "01000" "10000" "11110" "10001" "10001" "01110",
		/* 7 */ "11111" "00001" "00010" "00100" "01000" "01000" "01000",
		/* 8 */ "01110" "10001" "10001" "01110" "10001" "10001" "01110",
		/* 9 */ "01110" "10001" "10001" "01111" "00001" "00010" "01100",
		/* A */ "01110" "10001" "10001" "11111" "10001" "10001" "10001",
		/* B */ "11110" "10001" "10001" "11110" "10001" "10001" "11110",
		/* C */ "01110" "10001" "10000" "10000" "10000" "10001" "01110",
		/* D */ "11100" "10010" "10001" "10001" "10001" "10010" "11100",
		/* E */ "11111" "10000" "10000" "11110" "10000" "10000" "11111",
		/* F */ "11111" "10000" "10000" "11110" "10000" "10000" "10000",
		/* G */ "01110" "10001" "10000" "10111" "10001" "10001" "01111",
		/* H */ "10001" "10001" "10001" "11111" "10001" "10001" "10001",
		/* I */ "01110" "00100" "00100" "00100" "00100" "00100" "01110",
		/* J */ "00111" "00010" "00010" "00010" "00010" "10010" "01100",
		/* K */ "10001" "10010" "10100" "11000" "10100" "10010" "10001",
		/* L */ "10000" "10000" "10000" "10000" "10000" "10000" "11111",
		/* M */ "10001" "11011" "10101" "10101" "10001" "10001" "10001",
		/* N */ "10001" "10001" "11001" "10101" "10011" "10001" "10001",
		/* O */ "01110" "10001" "10001" "10001" "10001" "10001" "01110",
		/* P */ "11110" "10001" "10001" "11110" "10000" "10000" "10000",
		/* Q */ "01110" "10001" "10001" "10001" "10101" "10010" "01101",
		/* R */ "11110" "10001" "10001" "11110" "10100" "10010" "10001",
		/* S */ "01111" "10000" "10000" "01110" "00001" "00001" "11110",
		/* T */ "11111" "00100" "00100" "00100" "00100" "00100" "00100",
		/* U */ "10001" "10001" "10001" "10001" "10001" "10001" "01110",
		/* V */ "10001" "10001" "10001" "10001" "10001" "01010" "00100",
		/* W */ "10001" "10001" "10001" "10101" "10101" "10101" "01010",
		/* X */ "10001" "10001" "01010" "00100" "01010" "10001" "10001",
		/* Y */ "10001" "10001" "01010" "00100" "00100" "00100" "00100",
		/* Z */ "11111" "00001" "00010" "00100" "01000" "10000" "11111",
		/* ! */ "00100" "00100" "00100" "00100" "00100" "00000" "00100",
		/* . */ "00000" "00000" "00000" "00000" "00000" "01100" "01100",
	};
	if (c >= '0' && c <= '9') return G[c - '0'];
	if (c >= 'A' && c <= 'Z') return G[10 + c - 'A'];
	if (c == '!') return G[36];
	if (c == '.') return G[37];
	return 0;
}

/* draw text with `scale` pixels per font pixel, centred at cx; colour with a
 * 1px dark shadow.  Goes to OT bucket 0 (on top of everything). */
static char *big_text(const char *s, int cx, int y, int scale, int r, int g, int b, uint32_t *ot, char *nextpri) {
	TILE *t = (TILE *)nextpri;
	int len = (int)strlen(s);
	int w = len * 6 * scale;
	int x0 = cx - w / 2;
	/* bucket 0 draws the newest primitive first: add the coloured pixels
	 * first so the shadow (added second) lands underneath */
	for (int pass = 0; pass < 2; pass++) {
		int ox = pass ? scale / 2 + 1 : 0, oy = pass ? scale / 2 + 1 : 0;
		for (int i = 0; i < len; i++) {
			const char *gl = glyph(s[i]);
			if (!gl) continue;
			for (int row = 0; row < 7; row++)
				for (int col = 0; col < 5; col++) {
					if (gl[row * 5 + col] != '1') continue;
					setTile(t);
					if (pass) setRGB0(t, 10, 10, 20); else setRGB0(t, r, g, b);
					setXY0(t, x0 + (i * 6 + col) * scale + ox, y + row * scale + oy);
					setWH(t, scale, scale);
					addPrim(ot, t);
					t++;
				}
		}
	}
	return (char *)t;
}

/* ---------------------------------------------------------------------- */
static uint32_t rnd(Fight *fg) {
	fg->rng = fg->rng * 1664525u + 1013904223u;
	return fg->rng >> 8;
}

static int find_anim(const Model *m, const char *name) {
	for (int i = 0; i < m->hdr->nanims; i++) {
		const char *a = m->anims[i].name;
		int k = 0;
		while (name[k] && (a[k] | 0x20) == (name[k] | 0x20)) k++;
		if (!name[k]) return i;
	}
	return 0;
}

static void set_anim(Fighter *f, int anim) {
	pose_set_anim(&f->pose, anim);
	f->pose.playing = 1;
	f->pose.speed = 256;
}

static void fighter_setup(Fighter *f, const Model *m, Renderer *r) {
	f->model = m;
	f->renderer = r;
	pose_init(&f->pose, m, 0);
	f->anim_idle    = find_anim(m, "idle");
	f->anim_run     = find_anim(m, "run");
	f->anim_punch   = find_anim(m, "box");
	f->anim_kick    = find_anim(m, "front_kick");
	f->anim_special = find_anim(m, "shoot");
	f->anim_hit     = find_anim(m, "hit_to_body");
	f->anim_ko      = find_anim(m, "defeat");
	f->anim_win     = find_anim(m, "laugh");
	f->anim_jump    = find_anim(m, "jump");
	f->anim_fall    = find_anim(m, "fall");
}

static void fighter_reset(Fighter *f, int side) {
	f->x = side ? START_X : -START_X;
	f->z = 0;
	f->yaw = side ? 1024 : 3072;      /* models face -Z at yaw 0; +-90 deg faces the opponent */
	f->hp = 100;
	f->hp_disp = 100;
	f->state = FS_IDLE;
	f->cooldown = 30;
	f->hit_done = 0;
	set_anim(f, f->anim_idle);
}

void fight_init(Fight *fg, const Model *m0, Renderer *r0, const Model *m1, Renderer *r1) {
	memset(fg, 0, sizeof(*fg));
	fg->rng = 0xC0FFEE;
	fighter_setup(&fg->f[0], m0, r0);
	fighter_setup(&fg->f[1], m1, r1);
	fighter_reset(&fg->f[0], 0);
	fighter_reset(&fg->f[1], 1);
	fg->round = 1;
	fg->phase = FP_ROUND;
	fg->phase_t = 0;
	fg->timer = ROUND_FRAMES;
	fg->winner = -1;
	fg->cam_yaw = 0;
	fg->cam_dist = 6000;
	fg->cam_pitch = 160;
	fg->cam_target = vec(0, -1900, 0);
}

static void start_round(Fight *fg) {
	fighter_reset(&fg->f[0], 0);
	fighter_reset(&fg->f[1], 1);
	fg->phase = FP_ROUND;
	fg->phase_t = 0;
	fg->timer = ROUND_FRAMES;
	fg->winner = -1;
}

/* ---- AI ------------------------------------------------------------------ */
/* attack chains: punch-punch-kick and friends */
static const int COMBOS[][4] = {
	{ FA_PUNCH, -1, -1, -1 },
	{ FA_KICK, -1, -1, -1 },
	{ FA_PUNCH, FA_PUNCH, FA_KICK, -1 },
	{ FA_PUNCH, FA_KICK, -1, -1 },
	{ FA_KICK, FA_PUNCH, -1, -1 },
	{ FA_PUNCH, FA_PUNCH, FA_PUNCH, FA_SPECIAL },
	{ FA_SPECIAL, -1, -1, -1 },
	{ FA_KICK, FA_KICK, -1, -1 },
};
#define NCOMBOS ((int)(sizeof(COMBOS) / sizeof(COMBOS[0])))

static void start_attack(Fighter *f, int attack) {
	f->attack = attack;
	f->state = FS_ATTACK;
	f->hit_done = 0;
	set_anim(f, attack == FA_PUNCH ? f->anim_punch : attack == FA_KICK ? f->anim_kick : f->anim_special);
	f->pose.speed = attack == FA_SPECIAL ? 320 : attack == FA_PUNCH ? 768 : 512;   /* punches fastest */
}

static int reach_of(int attack) {
	return attack == FA_PUNCH ? REACH_PUNCH : attack == FA_KICK ? REACH_KICK : REACH_SPECIAL;
}

static int g_step = 1;      /* movement multiplier: 1 at 60 Hz, 2 at 30 Hz */

/* world position of a bone joint (fighter placement applied) */
static VECTOR bone_world(const Fighter *f, int bone) {
	const MATRIX *w = &f->pose.world[bone];
	VECTOR h = vec(w->t[0], w->t[1], w->t[2]);
	SVECTOR r = { 0, f->yaw, 0, 0 };
	MATRIX m4;
	RotMatrix(&r, &m4);
	VECTOR out;
	ApplyMatrixLV(&m4, &h, &out);
	return vec(out.vx + f->x, out.vy, out.vz + f->z);
}

/* the hand (punch / special) or foot (kick) closest to the opponent */
static VECTOR strike_point(const Fighter *f, int dir) {
	const ModelIK *ik = f->model->ik;
	int ca = f->attack == FA_KICK ? IK_LEG_L : IK_ARM_L;
	int cb = f->attack == FA_KICK ? IK_LEG_R : IK_ARM_R;
	VECTOR best = vec(f->x + dir * 500, -2500, f->z);
	int best_d = -1 << 30;
	for (int c = ca; c <= cb; c++) {
		if (!ik || ik->chains[c].upper < 0) continue;
		int b = ik->chains[c].end >= 0 ? ik->chains[c].end : ik->chains[c].lower;
		VECTOR p = bone_world(f, b);
		int dd = (p.vx - f->x) * dir;
		if (dd > best_d) { best_d = dd; best = p; }
	}
	return best;
}

static void fighter_ai(Fight *fg, int i) {
	Fighter *f = &fg->f[i], *o = &fg->f[1 - i];
	int dir = (o->x > f->x) ? 1 : -1;
	int d = (o->x - f->x) * dir;                        /* distance, positive */
	f->yaw = dir > 0 ? 3072 : 1024;                     /* always face the opponent */
	const ModelAnim *a = &f->model->anims[f->pose.anim];

	switch (f->state) {
	case FS_IDLE:
		if (f->cooldown > 0) { f->cooldown--; break; }
		if (d > APPROACH_STOP + 1200) {
			f->state = FS_APPROACH;                        /* far: run */
			set_anim(f, f->anim_run);
		} else if (d > APPROACH_STOP + 200) {
			f->state = FS_WALK;                            /* close: walk in (run clip at half speed) */
			set_anim(f, f->anim_run);
			f->pose.speed = 128;
		} else if (o->state == FS_DOWN) {
			f->cooldown = 10;                              /* let them get up */
		} else if (o->state != FS_KO) {
			uint32_t r = rnd(fg) % 100;
			if (r < 45) {
				uint32_t cr = rnd(fg) % 100;
				const int *c = cr < 30 ? COMBOS[2]                    /* punch-punch-kick 30% */
				             : cr < 50 ? COMBOS[0]                    /* single punch */
				             : COMBOS[rnd(fg) % NCOMBOS];
				f->combo_len = 0;
				for (int k = 0; k < 4 && c[k] >= 0; k++) f->combo[f->combo_len++] = c[k];
				f->combo_i = 0;
				f->last_backstep = 0;
				start_attack(f, f->combo[0]);
			} else if (!f->last_backstep && r < 56) {
				f->state = FS_RETREAT;                     /* walk backwards: run clip reversed */
				f->last_backstep = 1;
				f->cooldown = 14 + (rnd(fg) % 16);        /* duration of the retreat */
				set_anim(f, f->anim_run);
				f->pose.speed = -128;
			} else if (!f->last_backstep && (r < 64 || (r < 75 && d < MIN_DIST + 250))) {
				f->state = FS_BACKSTEP;                    /* hop back to reset the spacing */
				f->last_backstep = 1;
				set_anim(f, f->anim_jump);
			} else {
				f->last_backstep = 0;
				f->cooldown = 4 + (rnd(fg) % 16);
			}
		}
		break;
	case FS_BACKSTEP: {
		/* move back during the first 60% of the hop, then land */
		/* the jump clip contains several hops: use only the first one */
		int pct = a->nframes > 1 ? (f->pose.frame * 100) / (a->nframes - 1) : 100;
		if (pct < 24 && d < APPROACH_STOP + 900)
			f->x -= dir * BACKSTEP_SPEED * g_step;      /* away from the opponent */
		if (f->pose.loops > 0 || pct >= 30) {
			f->state = FS_IDLE;
			f->cooldown = 6 + (rnd(fg) % 20);
			set_anim(f, f->anim_idle);
		}
		break;
	}
	case FS_WALK:
		if (d > APPROACH_STOP) {
			f->x += dir * WALK_SPEED * g_step;
		} else {
			f->state = FS_IDLE;
			f->cooldown = 2 + (rnd(fg) % 8);
			set_anim(f, f->anim_idle);
		}
		break;
	case FS_RETREAT:
		f->x -= dir * WALK_SPEED * g_step;
		if (--f->cooldown <= 0 || d > APPROACH_STOP + 1500) {
			f->state = FS_IDLE;
			f->cooldown = 4 + (rnd(fg) % 10);
			set_anim(f, f->anim_idle);
		}
		break;
	case FS_APPROACH:
		if (d > APPROACH_STOP + 600) {
			f->x += dir * RUN_SPEED * g_step;
		} else if (d > APPROACH_STOP) {
			f->state = FS_WALK;                            /* slow down for the last stretch */
			set_anim(f, f->anim_run);
			f->pose.speed = 128;
		} else {
			f->state = FS_IDLE;
			f->cooldown = 4 + (rnd(fg) % 12);
			set_anim(f, f->anim_idle);
		}
		break;
	case FS_ATTACK: {
		int pct = a->nframes > 1 ? (f->pose.frame * 100) / (a->nframes - 1) : 100;
		/* active window: keeps checking until it connects or the swing is over */
		int hit_from = f->attack == FA_PUNCH ? 25 : ATTACK_HIT_AT;
		if (!f->hit_done && pct >= hit_from && pct <= 65) {
			if (d <= reach_of(f->attack) && o->state != FS_KO && o->state != FS_DOWN) {
				f->hit_done = 1;
				int dmg = f->attack == FA_PUNCH ? 9 : f->attack == FA_KICK ? 14 : 20;
				o->hp -= dmg;
				/* hit effect where the strike lands: the opponent's body surface
				 * facing the attacker, at the height of the striking hand / foot */
				VECTOR sp = strike_point(f, dir);
				int hx = o->x - dir * 280;
				if ((sp.vx - hx) * dir > 0) hx = sp.vx;          /* limb already inside: use it */
				for (int k = 0; k < MAX_FX; k++) {
					if (fg->fx[k].active) continue;
					fg->fx[k].active = 1; fg->fx[k].t = 0;
					fg->fx[k].x = hx; fg->fx[k].y = sp.vy; fg->fx[k].z = (sp.vz + o->z) / 2;
					break;
				}
				fg->shake = 14;
				if (o->hp <= 0) {
					o->hp = 0;
					o->state = FS_KO;
					set_anim(o, o->anim_ko);
				} else {
					o->state = FS_HIT;
					set_anim(o, o->anim_hit);
				}
				/* pushed away: kicks send the opponent flying */
				o->kb = dir * (f->attack == FA_PUNCH ? 70 : f->attack == FA_KICK ? 260 : 150);
				fg->hitstop = f->attack == FA_PUNCH ? 5 : 8;
			} else if (pct > 65) {
				f->hit_done = 1;
			}
		}
		/* combos cut the recovery half of each clip: chain right after the hit */
		if (f->pose.loops > 0 || (pct >= 55 && f->combo_i + 1 < f->combo_len)) {
			if (f->combo_i + 1 < f->combo_len && o->state != FS_KO) {
				f->combo_i++;
				start_attack(f, f->combo[f->combo_i]);
			} else {
				f->state = FS_IDLE;
				f->cooldown = 10 + (rnd(fg) % 25);
				set_anim(f, f->anim_idle);
			}
		}
		break;
	}
	case FS_DOWN:
		if (f->pose.loops > 0) {
			if (f->down_phase == 0) {
				/* lying down: get up by playing the fall clip backwards */
				f->down_phase = 1;
				f->pose.frame = a->nframes - 1;
				f->pose.subframe = 0;
				f->pose.loops = 0;
				f->pose.speed = -320;
			} else {                                    /* back on their feet */
				f->state = FS_IDLE;
				f->cooldown = 12 + (rnd(fg) % 16);
				set_anim(f, f->anim_idle);
			}
		}
		break;
	case FS_HIT:
		if (f->pose.loops > 0) {
			f->state = FS_IDLE;
			f->cooldown = 6 + (rnd(fg) % 10);
			set_anim(f, f->anim_idle);
		}
		break;
	case FS_KO:
		/* hold the last frame of the defeat animation */
		if (f->pose.loops > 0) {
			f->pose.frame = a->nframes - 1;
			f->pose.subframe = 0;
			f->pose.playing = 0;
			f->pose.loops = 0;
		}
		break;
	case FS_WIN:
		break;
	}
	/* keep them apart and on the stage */
	if (f->x < -4200) f->x = -4200;
	if (f->x >  4200) f->x =  4200;
}

/* ---- camera ------------------------------------------------------------ */
static int fsin(int a) {   /* sin(a), a in 4096 units -> 4096 */
	SVECTOR r = { 0, a & 4095, 0, 0 };
	MATRIX m;
	RotMatrix(&r, &m);
	return m.m[0][2];
}

static void auto_camera(Fight *fg, Camera *cam) {
	Fighter *a = &fg->f[0], *b = &fg->f[1];
	int mid = (a->x + b->x) / 2;
	int sep = a->x > b->x ? a->x - b->x : b->x - a->x;
	int want_yaw, want_dist, want_pitch;
	VECTOR want_t;
	switch (fg->phase) {
	case FP_ROUND:
		/* one full lap around the stage, low and close */
		want_yaw = (fg->phase_t * 4096) / 150;
		want_dist = 4600;
		want_pitch = 70;
		want_t = vec(mid, -2200, 0);
		fg->cam_yaw = want_yaw & 4095;                    /* no lag: keep the lap smooth */
		break;
	case FP_KO:
	case FP_END: {
		Fighter *w = fg->winner >= 0 ? &fg->f[fg->winner] : a;
		want_yaw = (fg->phase_t * 6) & 4095;             /* slow orbit around the winner */
		want_dist = 3600;
		want_pitch = 120;
		want_t = vec(w->x, -2400, w->z);
		break;
	}
	default:
		want_yaw = (fsin(fg->t * 2) * 380) >> 12;        /* gentle sway, ~20 s period */
		want_dist = 3400 + sep;
		if (want_dist < 4600) want_dist = 4600;
		if (want_dist > 7500) want_dist = 7500;
		want_pitch = 140 + ((fsin(fg->t * 3 + 1024) * 40) >> 12);
		want_t = vec(mid, -2000, 0);
		break;
	}
	/* smooth */
	int dy = ((want_yaw - fg->cam_yaw + 2048) & 4095) - 2048;
	fg->cam_yaw = (fg->cam_yaw + dy / 12) & 4095;
	fg->cam_dist += (want_dist - fg->cam_dist) / 14;
	fg->cam_pitch += (want_pitch - fg->cam_pitch) / 14;
	fg->cam_target.vx += (want_t.vx - fg->cam_target.vx) / 10;
	fg->cam_target.vy += (want_t.vy - fg->cam_target.vy) / 10;
	fg->cam_target.vz += (want_t.vz - fg->cam_target.vz) / 10;
	cam->yaw = fg->cam_yaw;
	cam->pitch = fg->cam_pitch;
	cam->dist = fg->cam_dist;
	cam->target = fg->cam_target;
	if (fg->shake > 0) {
		int amp = 40 + fg->shake * 8;
		cam->target.vy += (fg->shake & 1) ? amp : -amp;
		cam->target.vx += (fg->shake & 2) ? amp / 2 : -amp / 2;
		fg->shake--;
	}
	camera_update(cam, 0);
}

/* ---- per frame --------------------------------------------------------- */
static VECTOR head_world(const Fighter *f) {
	const Model *m = f->model;
	VECTOR h = vec(0, -3400, 0);
	if (m->ik && m->ik->head >= 0) {
		const MATRIX *w = &f->pose.world[m->ik->head];
		h = vec(w->t[0], w->t[1], w->t[2]);
	}
	SVECTOR r = { 0, f->yaw, 0, 0 };
	MATRIX m4;
	RotMatrix(&r, &m4);
	VECTOR out;
	ApplyMatrixLV(&m4, &h, &out);
	return vec(out.vx + f->x, out.vy, out.vz + f->z);
}

void fight_update(Fight *fg, Camera *cam, int hz) {
	g_step = hz >= 45 ? 1 : 2;
	fg->t += g_step;
	fg->phase_t += g_step;
	switch (fg->phase) {
	case FP_ROUND:
		if (fg->phase_t > 150) { fg->phase = FP_FIGHT; fg->phase_t = 0; }
		break;
	case FP_FIGHT:
		if (fg->phase_t > 70) { fg->phase = FP_FIGHTING; fg->phase_t = 0; }
		break;
	case FP_FIGHTING:
		fighter_ai(fg, 0);
		fighter_ai(fg, 1);
		fg->timer -= g_step;
		if (fg->timer < 0) fg->timer = 0;
		for (int i = 0; i < 2; i++) {
			if (fg->f[i].state == FS_KO) {
				fg->winner = 1 - i;
				fg->phase = FP_KO;
				fg->phase_t = 0;
			}
		}
		if (fg->phase == FP_FIGHTING && fg->timer == 0) {
			fg->winner = fg->f[0].hp >= fg->f[1].hp ? 0 : 1;
			fg->phase = FP_KO;
			fg->phase_t = 0;
		}
		if (fg->phase == FP_KO) {
			Fighter *w = &fg->f[fg->winner];
			if (w->state != FS_KO) { w->state = FS_WIN; set_anim(w, w->anim_win); }
			fg->wins[fg->winner]++;
		}
		break;
	case FP_KO:
		fighter_ai(fg, 0);                                   /* lets the KO pose settle */
		fighter_ai(fg, 1);
		if (fg->phase_t > 150) { fg->phase = FP_END; fg->phase_t = 0; }
		break;
	case FP_END:
		fighter_ai(fg, 0);
		fighter_ai(fg, 1);
		if (fg->phase_t > 200) {
			if (fg->wins[0] >= 2 || fg->wins[1] >= 2 || fg->round >= 3) {
				fg->round = 1;
				fg->wins[0] = fg->wins[1] = 0;
			} else {
				fg->round++;
			}
			start_round(fg);
		}
		break;
	}
	/* never let them stand inside each other */
	{
		Fighter *a = &fg->f[0], *b = &fg->f[1];
		int d = b->x - a->x;
		int ad = d < 0 ? -d : d;
		if (ad < MIN_DIST) {
			int push = (MIN_DIST - ad) / 2 + 1;
			int dir = d >= 0 ? 1 : -1;
			a->x -= dir * push;
			b->x += dir * push;
		}
	}
	/* displayed life drains slowly */
	for (int i = 0; i < 2; i++) {
		Fighter *f = &fg->f[i];
		if (f->hp_disp > f->hp) f->hp_disp -= (f->hp_disp - f->hp > 20) ? 2 * g_step : g_step;
		if (f->hp_disp < f->hp) f->hp_disp = f->hp;
	}
	for (int k = 0; k < MAX_FX; k++)
		if (fg->fx[k].active && (fg->fx[k].t += g_step) > 16) fg->fx[k].active = 0;
	/* knock back: the hit fighter slides away over a few frames */
	for (int i = 0; i < 2; i++) {
		Fighter *f = &fg->f[i];
		if (f->kb) {
			f->x += f->kb * g_step;
			f->kb = (f->kb * 13) / 16;
			if (f->kb > -6 && f->kb < 6) f->kb = 0;
		}
	}
	/* animation (frozen during hit stop) + head look-at towards the opponent */
	for (int i = 0; i < 2; i++) {
		Fighter *f = &fg->f[i];
		if (fg->hitstop <= 0)
			pose_step(&f->pose, hz >= 45 ? 60 : 30);
		pose_eval(&f->pose);
	}
	if (fg->hitstop > 0) fg->hitstop -= g_step;
	for (int i = 0; i < 2; i++) {
		Fighter *f = &fg->f[i], *o = &fg->f[1 - i];
		VECTOR oh = head_world(o);
		/* into this fighter's model space: Ry(-yaw) * (p - pos) */
		SVECTOR r = { 0, (-f->yaw) & 4095, 0, 0 };
		MATRIX m4;
		RotMatrix(&r, &m4);
		VECTOR d = vec(oh.vx - f->x, oh.vy, oh.vz - f->z), local;
		ApplyMatrixLV(&m4, &d, &local);
		ik_look_at_point(f->model, &f->pose, local);
	}
	auto_camera(fg, cam);
}

/* ---- drawing ------------------------------------------------------------ */
/* view matrix for a fighter: view * (T(pos) * Ry(yaw)) */
static void fighter_view(const Camera *cam, const Fighter *f, MATRIX *out) {
	SVECTOR r = { 0, f->yaw, 0, 0 };
	MATRIX ry;
	RotMatrix(&r, &ry);
	MulMatrix0((MATRIX *)&cam->view, &ry, out);
	VECTOR p = vec(f->x, 0, f->z), cp;
	ApplyMatrixLV((MATRIX *)&cam->view, &p, &cp);
	out->t[0] = cp.vx + cam->view.t[0];
	out->t[1] = cp.vy + cam->view.t[1];
	out->t[2] = cp.vz + cam->view.t[2];
}

/* hit sparks: an expanding ring of 8 squares + a flash at the impact point */
static char *draw_fx(Fight *fg, const Camera *cam, uint32_t *ot, char *nextpri) {
	TILE *t = (TILE *)nextpri;
	gte_SetRotMatrix(&cam->view);
	gte_SetTransMatrix(&cam->view);
	static const int dx[8] = { 0, 7, 10, 7, 0, -7, -10, -7 };
	static const int dy[8] = { -10, -7, 0, 7, 10, 7, 0, -7 };
	for (int k = 0; k < MAX_FX; k++) {
		HitFx *fx = &fg->fx[k];
		if (!fx->active) continue;
		SVECTOR v = { fx->x, fx->y, fx->z, 0 };
		uint32_t sxy; int32_t sz;
		gte_ldv0(&v); gte_rtps();
		__asm__ volatile("swc2 $14, 0(%0); swc2 $19, 0(%1)" :: "r"(&sxy), "r"(&sz) : "memory");
		if (sz <= 0) continue;
		int cx = (int16_t)(sxy & 0xffff), cy = (int16_t)(sxy >> 16);
		int r = 2 + fx->t * 2, size = fx->t < 6 ? 4 : 2;
		for (int i = 0; i < 8; i++) {
			setTile(t);
			setRGB0(t, 255, fx->t < 6 ? 240 : 160, fx->t < 6 ? 120 : 40);
			setXY0(t, cx + (dx[i] * r) / 10 - size / 2, cy + (dy[i] * r) / 10 - size / 2);
			setWH(t, size, size);
			addPrim(ot, t);
			t++;
		}
		if (fx->t < 4) {                              /* impact flash */
			setTile(t); setRGB0(t, 255, 255, 255);
			setXY0(t, cx - 6, cy - 6); setWH(t, 12, 12); addPrim(ot, t); t++;
		}
	}
	return (char *)t;
}

static char *draw_hud(Fight *fg, uint32_t *ot, char *nextpri) {
	TILE *t = (TILE *)nextpri;
	/* timer: drawn first in bucket 0 (= on top), on a dark box */
	char buf[4];
	int secs = (fg->timer + 59) / 60;
	buf[0] = '0' + (secs / 10) % 10; buf[1] = '0' + secs % 10; buf[2] = 0;
	nextpri = big_text(buf, CENTERX, 8, 3, 255, 255, 255, ot, nextpri);
	t = (TILE *)nextpri;
	setTile(t); setRGB0(t, 20, 20, 30); setXY0(t, CENTERX - 20, 5); setWH(t, 40, 28); addPrim(ot, t); t++;
	nextpri = (char *)t;

	t = (TILE *)nextpri;
	/* life bars: 130 px wide, 12 px thick; yellow = remaining, red = lost.
	 * Bucket 0 draws the newest primitive first, so the fill goes in before
	 * the red background */
	for (int i = 0; i < 2; i++) {
		int x0 = i ? SCREEN_XRES - 16 - 120 : 16;
		int w = (fg->f[i].hp_disp * 120) / 100;
		if (w > 0) {
			setTile(t); setRGB0(t, 255, 220, 40);
			setXY0(t, i ? x0 + 120 - w : x0, 14); setWH(t, w, 12); addPrim(ot, t); t++;
		}
		setTile(t); setRGB0(t, 200, 30, 30); setXY0(t, x0, 14); setWH(t, 120, 12); addPrim(ot, t); t++;
		setTile(t); setRGB0(t, 20, 20, 30); setXY0(t, x0 - 2, 12); setWH(t, 124, 16); addPrim(ot, t); t++;
		/* round wins */
		for (int k = 0; k < fg->wins[i]; k++) {
			setTile(t); setRGB0(t, 255, 230, 60);
			setXY0(t, i ? SCREEN_XRES - 22 - k * 8 : 16 + k * 8, 30); setWH(t, 6, 4); addPrim(ot, t); t++;
		}
	}
	nextpri = (char *)t;
	/* announcements */
	switch (fg->phase) {
	case FP_ROUND: {
		char r[8] = "ROUND 1";
		r[6] = '0' + fg->round;
		if (fg->phase_t > 10) nextpri = big_text(r, CENTERX, 90, 4, 255, 220, 60, ot, nextpri);
		break;
	}
	case FP_FIGHT:
		nextpri = big_text("FIGHT!", CENTERX, 88, 5, 255, 80, 60, ot, nextpri);
		break;
	case FP_KO:
		if (fg->phase_t > 5)
			nextpri = big_text(fg->timer == 0 ? "TIME UP" : "K.O.", CENTERX, 88, 5, 255, 60, 60, ot, nextpri);
		break;
	case FP_END: {
		const char *s = fg->winner == 0 ? "P1 WIN" : "P2 WIN";
		nextpri = big_text(s, CENTERX, 90, 4, 255, 220, 60, ot, nextpri);
		break;
	}
	default:
		break;
	}
	return nextpri;
}

char *fight_draw(Fight *fg, const Camera *cam, uint32_t *ot, char *nextpri) {
	for (int i = 0; i < 2; i++) {
		Fighter *f = &fg->f[i];
		Camera fc = *cam;
		fighter_view(cam, f, &fc.view);
		nextpri = render_model(f->renderer, f->model, &f->pose, &fc, ot, nextpri);
	}
	nextpri = draw_fx(fg, cam, ot, nextpri);
	return draw_hud(fg, ot, nextpri);
}
