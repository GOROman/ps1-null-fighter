#include <string.h>
#include <psxgpu.h>
#include <psxgte.h>
#include <inline_c.h>
#include <psxpad.h>
#include "fight.h"
#include "ik.h"
#include "fixmath.h"

#define SCREEN_XRES 320
#define SCREEN_YRES 240
#define CENTERX     (SCREEN_XRES >> 1)
#define CENTERY     (SCREEN_YRES >> 1)

#define ROUND_FRAMES   (60 * 60)      /* 60 s */
#define START_X        3000           /* fighters start at +-START_X (2x wider stage) */
#define APPROACH_STOP  1200
#define MIN_DIST       1000
#define RUN_SPEED      44
#define WALK_SPEED     20
#define PLAYER_SPEED   30             /* pad controlled walk */
#define BACKSTEP_SPEED 48
#define FLOOR_OTZ      (OT_LEN - 1)
#define STAGE_EDGE     8400           /* ring out boundary (2x original 4200) */
#define STAGE_CLAMP    8800           /* hard clamp (2x original 4400) */

/* hadouken (projectile) balance - conservative to fit VF-style close combat */
#define HADOUKEN_SPEED    18          /* slow projectile (px/frame) */
#define HADOUKEN_RANGE    2400        /* short range: disappears after this distance */
#define HADOUKEN_CD       (4 * 60)    /* 4 second cooldown: no spam */
#define HADOUKEN_DMG      5           /* low damage */
#define HADOUKEN_KB       80          /* light knockback */
#define HADOUKEN_HITBOX   400         /* collision radius */
#define HADOUKEN_MOVE_IDX 32          /* index in MOVES table */

/* dogeza (prostration) - desperation move when HP <= 30% */
#define DOGEZA_HP_THRESHOLD 30        /* can only use when HP <= 30% */
#define DOGEZA_STARTUP     30         /* frames before speech bubble appears */
#define DOGEZA_ACTIVE      60         /* frames the speech bubble is active */
#define DOGEZA_RECOVERY    30         /* frames after bubble disappears */
#define DOGEZA_TOTAL       (DOGEZA_STARTUP + DOGEZA_ACTIVE + DOGEZA_RECOVERY)
#define DOGEZA_DMG         40         /* massive damage if bubble hits */
#define DOGEZA_KB          400        /* huge knockback */
#define DOGEZA_HITBOX_W    80         /* speech bubble hitbox width */
#define DOGEZA_HITBOX_H    50         /* speech bubble hitbox height */

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
char *fight_text(const char *s, int cx, int y, int scale, int r, int g, int b, uint32_t *ot, char *nextpri) {
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
	return -1;
}

static int find_anim_or(const Model *m, const char *name, int dflt) {
	int a = find_anim(m, name);
	return a < 0 ? dflt : a;
}

static int move_index(const char *name) {
	for (int i = 0; i < NUM_MOVES; i++) {
		const char *a = MOVES[i].name;
		int k = 0;
		while (name[k] && a[k] == name[k]) k++;
		if (!name[k] && !a[k]) return i;
	}
	return -1;
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
	f->anim_idle = find_anim_or(m, "idle", 0);
	f->anim_run  = find_anim_or(m, "run", f->anim_idle);
	f->anim_hit  = find_anim_or(m, "hit_to_body", f->anim_idle);
	f->anim_ko   = find_anim_or(m, "defeat", f->anim_idle);
	f->anim_win  = find_anim_or(m, "laugh", f->anim_idle);
	f->anim_jump = find_anim_or(m, "jump", f->anim_idle);
	f->anim_fall = find_anim_or(m, "fall", f->anim_ko);
	f->anim_guard = find_anim_or(m, "guard", f->anim_idle);
	f->anim_dogeza = find_anim_or(m, "defeat", f->anim_idle);  /* use defeat as dogeza pose */
	for (int i = 0; i < NUM_MOVES; i++)
		f->move_anim[i] = find_anim(m, MOVES[i].clip);       /* -1: this rig lacks the clip */
}

static void fighter_reset(Fighter *f, int side) {
	f->x = side ? START_X : -START_X;
	f->z = 0;
	f->yaw = side ? 1024 : 3072;      /* models face -Z at yaw 0; +-90 deg faces the opponent */
	f->hp = 100;
	f->hp_disp = 100;
	f->plan_special = 0;
	f->special_cd = 0;
	f->state = FS_IDLE;
	f->cooldown = 30;
	f->hit_done = 0;
	f->yaw_corr = 0;
	f->kb = 0;
	f->guard_t = 0;
	f->hits_taken = 0;
	f->in_pressed = 0;
	f->hadouken_cd = 0;
	f->cmd_idx = 0;
	for (int i = 0; i < 8; i++) f->cmd_hist[i] = 0;
	f->dogeza_t = 0;
	f->dogeza_hit = 0;
	set_anim(f, f->anim_idle);
}

/* ---- advice phrases ("GOIKEN") ----------------------------------------
 * Random advice/opinions that appear during fights, fitting the world of
 * "Goiken Yuuyou" (Your Opinion is Valuable).  Short, dignified lines. */
static const char *const ADVICE[] = {
	"FIST SPEAKS TRUTH",
	"HONOR YOUR FOES",
	"STYLE IS VICTORY",
	"BREATHE. STRIKE.",
	"FLOW LIKE WATER",
	"READ THE MOMENT",
	"NO WASTED MOTION",
	"PATIENCE WINS",
	"RESPECT THE RING",
	"COURAGE IS CALM",
	"FIGHT WITH GRACE",
	"ONE CHANCE. SEIZE.",
};
#define NUM_ADVICE ((int)(sizeof(ADVICE) / sizeof(ADVICE[0])))
#define ADVICE_SHOW_FRAMES  180    /* how long advice stays on screen (3 s) */
#define ADVICE_CD_MIN       300    /* minimum frames between advice (5 s) */
#define ADVICE_CD_MAX       600    /* maximum frames between advice (10 s) */

/* ---- combos ---------------------------------------------------------- */
/* attack chains by move name; resolved to move indices at init */
static const char *const COMBO_NAMES[][4] = {
	{ "jab", "jab", "knee", "high_kick" },      /* P-P-K: the kick lands twice (knee, then high) */
	{ "jab", "straight", 0, 0 },
	{ "jab", "jab", "roundhouse", 0 },
	{ "hook", "uppercut", 0, 0 },
	{ "elbow", "knee", "high_kick", 0 },
	{ "backfist", "back_kick", 0, 0 },
	{ "front_kick", "spin_kick", 0, 0 },
	{ "jab", "palm", 0, 0 },
	{ "knee", "high_kick", 0, 0 },
	{ "straight", "hammer", 0, 0 },
	{ "jab", "jab", "knee", "high_kick" },
	{ "hook", "roundhouse", 0, 0 },
};
#define NCOMBOS ((int)(sizeof(COMBO_NAMES) / sizeof(COMBO_NAMES[0])))
static int COMBOS[NCOMBOS][4];

static void resolve_combos(void) {
	for (int c = 0; c < NCOMBOS; c++)
		for (int k = 0; k < 4; k++)
			COMBOS[c][k] = COMBO_NAMES[c][k] ? move_index(COMBO_NAMES[c][k]) : -1;
}

void fight_init(Fight *fg, const Model *m0, Renderer *r0, const Model *m1, Renderer *r1) {
	memset(fg, 0, sizeof(*fg));
	fg->rng = 0xC0FFEE;
	resolve_combos();
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
	fg->advice_t = 0;
	fg->advice_idx = 0;
	fg->advice_cd = ADVICE_CD_MIN + (fg->rng % (ADVICE_CD_MAX - ADVICE_CD_MIN));
	fg->training = 0;
	fg->last_dmg = 0;
}

void fight_set_training(Fight *fg, int training) {
	fg->training = training;
	if (training) {
		fg->phase = FP_FIGHTING;
		fg->phase_t = 0;
		fg->timer = 0;
		fg->f[1].human = 0;
	}
}

void fight_set_players(Fight *fg, int p0_human, int p1_human) {
	fg->f[0].human = p0_human;
	fg->f[1].human = p1_human;
}

void fight_input(Fight *fg, int side, uint16_t held, uint16_t pressed) {
	Fighter *f = &fg->f[side & 1];
	f->in_held = held;
	f->in_pressed |= pressed;          /* consumed by fight_update */
}

static void start_round(Fight *fg) {
	fighter_reset(&fg->f[0], 0);
	fighter_reset(&fg->f[1], 1);
	fg->phase = FP_ROUND;
	fg->phase_t = 0;
	fg->timer = ROUND_FRAMES;
	fg->winner = -1;
	fg->ringout = 0;
	for (int i = 0; i < MAX_PROJECTILES; i++) fg->proj[i].active = 0;
}

/* ---- 236 command detection (quarter-circle forward + attack) ------------- */
/* Records d-pad state each frame and checks for down -> down-forward -> forward
 * sequence within the last ~12 frames, relative to the fighter's facing direction.
 * Returns 1 if the command was detected. */
static void record_dpad(Fighter *f, uint16_t held) {
	uint16_t dpad = held & (PAD_UP | PAD_DOWN | PAD_LEFT | PAD_RIGHT);
	f->cmd_hist[f->cmd_idx] = dpad;
	f->cmd_idx = (f->cmd_idx + 1) & 7;
}

static int check_236(const Fighter *f, int dir) {
	/* dir > 0: opponent is to the right, so forward = RIGHT
	 * dir < 0: opponent is to the left, so forward = LEFT */
	uint16_t down = PAD_DOWN;
	uint16_t fwd  = dir > 0 ? PAD_RIGHT : PAD_LEFT;
	uint16_t dfwd = down | fwd;  /* down-forward */

	/* scan the circular buffer backwards for the sequence: down -> down-fwd -> fwd
	 * allow some leniency (a few frames per step) */
	int idx = (f->cmd_idx - 1) & 7;
	int state = 0;  /* 0: looking for fwd, 1: looking for down-fwd, 2: looking for down */
	int frames[3] = {0, 0, 0};  /* frames spent in each state */
	const int MAX_GAP = 4;  /* max frames for each step */

	for (int i = 0; i < 8; i++) {
		uint16_t d = f->cmd_hist[idx];
		idx = (idx - 1) & 7;

		if (state == 0) {
			/* looking for forward (most recent input) */
			if ((d & (PAD_DOWN | fwd)) == fwd) {
				frames[0]++;
				if (frames[0] >= 1) state = 1;
			} else if ((d & (PAD_DOWN | fwd)) == dfwd) {
				/* can skip directly to down-forward */
				state = 1;
				frames[1] = 1;
			} else if (frames[0] > 0) {
				break;  /* gap too large */
			}
		} else if (state == 1) {
			/* looking for down-forward */
			if ((d & (PAD_DOWN | fwd)) == dfwd) {
				frames[1]++;
				if (frames[1] >= 1) state = 2;
			} else if ((d & (PAD_DOWN | fwd)) == down) {
				/* can skip directly to down */
				state = 2;
				frames[2] = 1;
			} else if (frames[1] > 0 && frames[1] < MAX_GAP) {
				/* allow small gap */
			} else if (frames[1] >= MAX_GAP) {
				break;
			}
		} else if (state == 2) {
			/* looking for down */
			if ((d & (PAD_DOWN | fwd)) == down) {
				frames[2]++;
				if (frames[2] >= 1) return 1;  /* success! */
			} else if (frames[2] > 0 && frames[2] < MAX_GAP) {
				/* allow small gap */
			} else if (frames[2] >= MAX_GAP) {
				break;
			}
		}
	}
	return 0;
}

/* spawn a hadouken projectile for fighter `side` */
static void spawn_hadouken(Fight *fg, int side) {
	Fighter *f = &fg->f[side];
	int dir = (fg->f[1 - side].x > f->x) ? 1 : -1;

	/* find a free projectile slot */
	for (int i = 0; i < MAX_PROJECTILES; i++) {
		Projectile *p = &fg->proj[i];
		if (p->active) continue;

		p->active = 1;
		p->owner = side;
		p->x = f->x + dir * 600;  /* spawn in front of the fighter */
		p->y = -1800;             /* chest height */
		p->z = f->z;
		p->vx = dir * HADOUKEN_SPEED;
		p->life = HADOUKEN_RANGE / HADOUKEN_SPEED;  /* frames until it disappears */
		p->dmg = HADOUKEN_DMG;
		break;
	}
}

/* ---- attacks --------------------------------------------------------- */
static int g_step = 1;      /* movement multiplier: 1 at 60 Hz, 2 at 30 Hz */

static void start_attack(Fight *fg, int i, int move) {
	Fighter *f = &fg->f[i];
	if (move < 0 || f->move_anim[move] < 0) move = 0;
	const MoveDef *mv = &MOVES[move];
	f->move = move;
	f->state = FS_ATTACK;
	f->hit_done = 0;
	f->rehit_at = 0;
	set_anim(f, f->move_anim[move]);
	f->pose.speed = mv->speed;
	if (mv->cat == CAT_KICK && f->combo_i > 0) {
		/* chained kick: skip the wind-up */
		const ModelAnim *a = &f->model->anims[f->pose.anim];
		f->pose.frame = (a->nframes * 20) / 100;
	}
	fg->name_t = 50;
	fg->name_move = move;
	fg->name_side = i;
}

/* start a combo (list of move indices, -1 terminated) */
static void start_combo(Fight *fg, int i, const int *c) {
	Fighter *f = &fg->f[i];
	f->combo_len = 0;
	for (int k = 0; k < 4 && c[k] >= 0; k++)
		if (f->move_anim[c[k]] >= 0) f->combo[f->combo_len++] = c[k];
	if (f->combo_len == 0) f->combo[f->combo_len++] = 0;
	f->combo_i = 0;
	f->last_backstep = 0;
	start_attack(fg, i, f->combo[0]);
}

static void start_single(Fight *fg, int i, int move) {
	int c[2] = { move, -1 };
	start_combo(fg, i, c);
}

/* a random move of a category that reaches distance d (or -1) */
static int pick_move(Fight *fg, const Fighter *f, int cat, int d) {
	int cand[NUM_MOVES], n = 0;
	for (int i = 0; i < NUM_MOVES; i++) {
		if (MOVES[i].cat != cat || f->move_anim[i] < 0) continue;
		if (MOVES[i].reach + MOVES[i].travel * 20 < d + 100) continue;
		cand[n++] = i;
	}
	return n ? cand[rnd(fg) % n] : -1;
}

/* world position of a bone joint (fighter placement applied) */
static VECTOR bone_world_yaw(const Fighter *f, int bone, int yaw) {
	const MATRIX *w = &f->pose.world[bone];
	VECTOR h = vec(w->t[0], w->t[1], w->t[2]);
	SVECTOR r = { 0, yaw & 4095, 0, 0 };
	MATRIX m4;
	RotMatrix(&r, &m4);
	VECTOR out;
	ApplyMatrixLV(&m4, &h, &out);
	return vec(out.vx + f->x, out.vy, out.vz + f->z);
}

/* the bone that strikes for the current move: the hand / foot / knee
 * (or head) that is furthest towards the opponent */
static int strike_bone(const Fighter *f, int dir) {
	const ModelIK *ik = f->model->ik;
	int limb = MOVES[f->move].limb;
	if (!ik) return 0;
	int cand[5], n = 0;
	if (limb == LIMB_HAND || limb == LIMB_FOOT) {
		int ca = limb == LIMB_FOOT ? IK_LEG_L : IK_ARM_L;
		for (int c = ca; c <= ca + 1; c++) {
			if (ik->chains[c].upper < 0) continue;
			cand[n++] = ik->chains[c].end >= 0 ? ik->chains[c].end : ik->chains[c].lower;
		}
	} else {
		for (int c = IK_LEG_L; c <= IK_LEG_R; c++)
			if (ik->chains[c].upper >= 0) cand[n++] = ik->chains[c].lower;   /* knees */
		if (ik->head >= 0) cand[n++] = ik->head;
		cand[n++] = ik->hip;
	}
	int best = cand[0], best_d = -1 << 30;
	for (int k = 0; k < n; k++) {
		VECTOR p = bone_world_yaw(f, cand[k], f->yaw);
		int dd = (p.vx - f->x) * dir;
		if (dd > best_d) { best_d = dd; best = cand[k]; }
	}
	return best;
}

/* ---- player control ------------------------------------------------------ */
/* Virtua Fighter layout: square = guard, triangle = punch, circle = kick,
 * punch + kick together = special */
#define BTN_G PAD_SQUARE
#define BTN_P PAD_TRIANGLE
#define BTN_K PAD_CIRCLE
enum { ATK_NONE = 0, ATK_P, ATK_K, ATK_S };
static int attack_button(uint16_t pressed, uint16_t held) {
	if (!(pressed & (BTN_P | BTN_K))) return ATK_NONE;
	if ((held & BTN_P) && (held & BTN_K)) return ATK_S;
	return (pressed & BTN_P) ? ATK_P : ATK_K;
}

/* the move for an attack button, given the d-pad direction (dir: towards
 * the opponent along x, which is the screen direction at camera yaw 0) */
static int player_move(const Fighter *f, uint16_t held, int dir, int btn) {
	int fwd  = held & (dir > 0 ? PAD_RIGHT : PAD_LEFT);
	int back = held & (dir > 0 ? PAD_LEFT : PAD_RIGHT);
	int down = held & PAD_DOWN;
	const char *n;
	if (btn == ATK_P)      n = down ? "uppercut" : fwd ? "straight"    : back ? "hook"      : "jab";
	else if (btn == ATK_K) n = down ? "sweep"    : fwd ? "roundhouse"  : back ? "back_kick" : "high_kick";
	else                   n = down ? "slide"    : fwd ? "dragon_kick" : back ? "hurricane" : "sbk";
	int m = move_index(n);
	if (m < 0 || f->move_anim[m] < 0) m = 0;
	return m;
}

/* start dogeza move */
static void start_dogeza(Fight *fg, int i) {
	Fighter *f = &fg->f[i];
	f->state = FS_DOGEZA;
	f->dogeza_t = 0;
	f->dogeza_hit = 0;
	set_anim(f, f->anim_dogeza);
	f->pose.speed = 384;
}

/* check for dogeza command: down + P + K + G simultaneously */
static int check_dogeza_cmd(uint16_t held, uint16_t pressed) {
	int down = held & PAD_DOWN;
	int all_buttons = (held & BTN_P) && (held & BTN_K) && (held & BTN_G);
	int any_pressed = (pressed & BTN_P) || (pressed & BTN_K) || (pressed & BTN_G);
	return down && all_buttons && any_pressed;
}

/* attack / guard input from a neutral state; 1 when an action started */
static int player_act(Fight *fg, int i, int dir) {
	Fighter *f = &fg->f[i];
	uint16_t held = f->in_held, pressed = f->in_pressed;
	f->in_pressed = 0;

	/* dogeza command: down + P + K + G when HP <= 30% */
	if (f->hp <= DOGEZA_HP_THRESHOLD && check_dogeza_cmd(held, pressed)) {
		start_dogeza(fg, i);
		return 1;
	}

	int btn = attack_button(pressed, held);
	if (btn) {
		/* check for 236+P hadouken command (quarter-circle forward + punch) */
		if (btn == ATK_P && f->hadouken_cd <= 0 && check_236(f, dir)) {
			int m = HADOUKEN_MOVE_IDX;
			if (f->move_anim[m] >= 0) {
				f->hadouken_cd = HADOUKEN_CD;
				start_single(fg, i, m);
				return 1;
			}
		}
		if (btn == ATK_S && f->special_cd > 0) btn = ATK_P;   /* special on cooldown: a punch */
		int m = player_move(f, held, dir, btn);
		if (MOVES[m].cat == CAT_SPECIAL) f->special_cd = 4 * 60;
		start_single(fg, i, m);
		return 1;
	}
	if (held & BTN_G) {
		f->state = FS_GUARD;
		f->guard_t = 0;
		set_anim(f, f->anim_guard);
		return 1;
	}
	return 0;
}

/* d-pad: idle / walk in / walk back */
static void player_walk(Fight *fg, int i, int dir, int d) {
	Fighter *f = &fg->f[i];
	int fwd  = f->in_held & (dir > 0 ? PAD_RIGHT : PAD_LEFT);
	int back = f->in_held & (dir > 0 ? PAD_LEFT : PAD_RIGHT);
	int want = fwd ? FS_WALK : back ? FS_RETREAT : FS_IDLE;
	if (want != f->state) {
		f->state = want;
		if (want == FS_IDLE) {
			set_anim(f, f->anim_idle);
		} else {
			set_anim(f, f->anim_run);
			f->pose.speed = want == FS_WALK ? 128 : -128;
		}
	}
	if (want == FS_WALK && d > MIN_DIST) f->x += dir * PLAYER_SPEED * g_step;
	if (want == FS_RETREAT)              f->x -= dir * PLAYER_SPEED * g_step;
}

/* ---- AI ----------------------------------------------------------------- */
static void fighter_ai(Fight *fg, int i) {
	Fighter *f = &fg->f[i], *o = &fg->f[1 - i];
	int dir = (o->x > f->x) ? 1 : -1;
	int d = (o->x - f->x) * dir;                        /* distance, positive */
	int base_yaw = dir > 0 ? 3072 : 1024;               /* always face the opponent */
	f->yaw = base_yaw;
	if (f->state == FS_ATTACK) f->yaw = (base_yaw + f->yaw_corr) & 4095;
	else f->yaw_corr = 0;
	const ModelAnim *a = &f->model->anims[f->pose.anim];
	/* training mode: P1 has no cooldowns */
	if (fg->training && i == 0) {
		f->special_cd = 0;
		f->hadouken_cd = 0;
	} else {
		if (f->special_cd > 0) f->special_cd -= g_step;
		if (f->hadouken_cd > 0) f->hadouken_cd -= g_step;
	}
	if (f->human) record_dpad(f, f->in_held);  /* track d-pad for command inputs */

	switch (f->state) {
	case FS_IDLE: {
		if (f->cooldown > 0) { f->cooldown--; break; }
		if (o->state == FS_KO) break;
		if (f->human) {
			if (!player_act(fg, i, dir)) player_walk(fg, i, dir, d);
			break;
		}
		/* ---- strategy ---------------------------------------------------
		 * read the opponent: attack startup / recovery, stun, distance zone,
		 * life lead and the clock; then pick an action.  rnd only breaks ties.
		 */
		int aggr = i == 0 ? 60 : 45;                       /* personality: P1 pushes, P2 counters */
		if (f->hp < 30) aggr -= 15;                        /* hurt: play safe */
		if (o->hp < 30) aggr += 15;                        /* smell blood */
		if (fg->timer < 15 * 60 && f->hp < o->hp) aggr += 30;   /* losing on time: go */
		const ModelAnim *oa = &o->model->anims[o->pose.anim];
		int opct = oa->nframes > 1 ? (o->pose.frame * 100) / (oa->nframes - 1) : 100;
		int o_startup  = o->state == FS_ATTACK && opct < 30;          /* about to swing */
		int o_recovery = o->state == FS_ATTACK && opct >= 55;         /* whiffed / recovering */
		int o_stunned  = o->state == FS_HIT;
		int o_backing  = o->state == FS_RETREAT || o->state == FS_BACKSTEP;
		int in_punch   = d <= 1450;                        /* commit only with margin */
		int in_kick    = d <= 1800;
		uint32_t r = rnd(fg) % 100;

		if (f->plan_special) {
			/* retreat done: unleash a special (the spinning bird kick half the time) */
			f->plan_special = 0;
			f->special_cd = 10 * 60;
			int m = (r < 50) ? move_index("sbk") : pick_move(fg, f, CAT_SPECIAL, d);
			if (m < 0 || f->move_anim[m] < 0) m = pick_move(fg, f, CAT_KICK, 0);
			start_single(fg, i, m);
		} else if (f->hp <= DOGEZA_HP_THRESHOLD && f->hp < o->hp && d <= 1600 && d >= 600 &&
		           o->state != FS_ATTACK && r < 8) {
			/* DOGEZA: desperation move when losing badly (rare: ~8% chance) */
			start_dogeza(fg, i);
		} else if (f->hp < 35 && f->hp < o->hp && f->special_cd <= 0 && d <= 2400 &&
		           o->state != FS_ATTACK) {
			/* losing: back off a few steps, then the special covers the distance */
			f->plan_special = 1;
			f->state = FS_RETREAT;
			f->cooldown = 16;
			set_anim(f, f->anim_run); f->pose.speed = -128;
		} else if (o->state == FS_GUARD && in_kick && r < 70) {
			/* they are blocking: go under the guard with a low attack */
			int low = -1, tries = 8;
			while (low < 0 && tries--) {
				int m = pick_move(fg, f, CAT_KICK, d);
				if (m >= 0 && (MOVES[m].height == H_LOW || (MOVES[m].flags & MF_KNOCKDOWN))) low = m;
			}
			if (low < 0) low = move_index("sweep");
			start_single(fg, i, low);
		} else if (o_recovery || o_stunned) {
			/* punish window: rush in with a combo (kick from farther out) */
			if (in_punch) {
				start_combo(fg, i, COMBOS[r < 50 ? 0 : rnd(fg) % NCOMBOS]);
			} else if (in_kick) {
				start_single(fg, i, pick_move(fg, f, CAT_KICK, d));
			} else {
				f->state = FS_APPROACH; set_anim(f, f->anim_run);
			}
		} else if (o_startup && in_kick) {
			/* the opponent is swinging: guard, beat it with a fast punch, or get out */
			if (r >= aggr && r < aggr + 35) {
				f->state = FS_GUARD;
				f->guard_t = 24 + (rnd(fg) % 16);
				set_anim(f, f->anim_guard);
			} else if (in_punch && r < aggr) {
				start_single(fg, i, r < aggr / 2 ? move_index("jab") : pick_move(fg, f, CAT_PUNCH, d));
			} else if (!f->last_backstep) {
				f->state = FS_BACKSTEP; f->last_backstep = 1; set_anim(f, f->anim_jump);
			} else {
				f->state = FS_RETREAT; f->cooldown = 10 + (rnd(fg) % 8);
				set_anim(f, f->anim_run); f->pose.speed = -128;
			}
		} else if (d > APPROACH_STOP + 1200) {
			if (r < 12 && f->special_cd <= 0) {
				/* surprise: a travelling special from far out */
				int m = pick_move(fg, f, CAT_SPECIAL, d);
				if (m >= 0) { f->special_cd = 6 * 60; start_single(fg, i, m); break; }
			}
			f->state = FS_APPROACH; set_anim(f, f->anim_run);
		} else if (d > APPROACH_STOP + 200) {
			/* mid range: walk in, or hang back and wait for a whiff when defensive */
			if (o_backing || r < aggr + 20) {
				f->state = FS_WALK; set_anim(f, f->anim_run); f->pose.speed = 128;
			} else {
				f->cooldown = 6 + (rnd(fg) % 10);         /* wait and watch */
			}
		} else if (o->state == FS_DOWN) {
			f->cooldown = 10;                              /* let them get up */
		} else if (in_punch && r < aggr) {
			/* close and the opponent is idle: open with a combo (mix-ups) */
			if (r < aggr / 3) start_combo(fg, i, COMBOS[0]);
			else if (r < (aggr * 2) / 3) start_combo(fg, i, COMBOS[rnd(fg) % NCOMBOS]);
			else start_single(fg, i, pick_move(fg, f, rnd(fg) % 3 ? CAT_PUNCH : CAT_KICK, d));
		} else if (in_kick && !in_punch && r < aggr) {
			start_single(fg, i, pick_move(fg, f, CAT_KICK, d));   /* kick range poke */
		} else if (!f->last_backstep && r < 85) {
			/* bait: step out so the opponent whiffs, then punish */
			f->state = FS_RETREAT; f->last_backstep = 1;
			f->cooldown = 12 + (rnd(fg) % 12);
			set_anim(f, f->anim_run); f->pose.speed = -128;
		} else {
			f->last_backstep = 0;
			f->cooldown = 4 + (rnd(fg) % 10);
		}
		break;
	}
	case FS_GUARD:
		if (f->human) {
			/* held as long as cross is down */
			f->in_pressed = 0;
			if (!(f->in_held & BTN_G)) {
				f->state = FS_IDLE;
				f->cooldown = 2;
				set_anim(f, f->anim_idle);
			}
			break;
		}
		/* hold the guard while the opponent is still swinging, then relax */
		if (--f->guard_t <= 0 && o->state != FS_ATTACK) {
			f->state = FS_IDLE;
			f->cooldown = 2 + (rnd(fg) % 6);
			set_anim(f, f->anim_idle);
		}
		if (f->guard_t < -60) { f->state = FS_IDLE; set_anim(f, f->anim_idle); }
		break;
	case FS_WALK:
		if (f->human) {
			if (!player_act(fg, i, dir)) player_walk(fg, i, dir, d);
			break;
		}
		if (d > APPROACH_STOP) {
			f->x += dir * WALK_SPEED * g_step;
		} else {
			f->state = FS_IDLE;
			f->cooldown = 2 + (rnd(fg) % 8);
			set_anim(f, f->anim_idle);
		}
		break;
	case FS_RETREAT:
		if (f->human) {
			if (!player_act(fg, i, dir)) player_walk(fg, i, dir, d);
			break;
		}
		f->x -= dir * WALK_SPEED * g_step;
		if (--f->cooldown <= 0 || (!f->plan_special && d > APPROACH_STOP + 1500)) {
			f->state = FS_IDLE;
			f->cooldown = f->plan_special ? 0 : 4 + (rnd(fg) % 10);
			set_anim(f, f->anim_idle);
		}
		break;
	case FS_BACKSTEP:
		/* one hop backwards on the jump clip */
		f->x -= dir * BACKSTEP_SPEED * g_step;
		if (f->pose.loops > 0) {
			f->state = FS_IDLE;
			f->cooldown = 6 + (rnd(fg) % 10);
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
		const MoveDef *mv = &MOVES[f->move];
		int pct = a->nframes > 1 ? (f->pose.frame * 100) / (a->nframes - 1) : 100;
		if (f->human) {
			/* an attack button during the swing buffers the next move of the chain */
			int btn = attack_button(f->in_pressed, f->in_held);
			f->in_pressed = 0;
			if (btn && f->combo_len < 4 && f->combo_len == f->combo_i + 1) {
				if (btn == ATK_S && f->special_cd > 0) btn = ATK_P;
				int m = player_move(f, f->in_held, dir, btn);
				if (MOVES[m].cat == CAT_SPECIAL) f->special_cd = 4 * 60;
				f->combo[f->combo_len++] = m;
			}
		}
		/* hadouken: spawn projectile at the "release" frame (50%) */
		if (f->move == HADOUKEN_MOVE_IDX && !f->hit_done && pct >= 50) {
			f->hit_done = 1;  /* only spawn once */
			spawn_hadouken(fg, i);
		}
		int active = pct >= mv->hit_from && pct <= mv->hit_to;
		/* travel: the move carries the body forward around its active frames */
		if (mv->travel && pct >= mv->hit_from - 15 && pct <= mv->hit_to && d > MIN_DIST)
			f->x += dir * mv->travel * g_step;
		else if (f->combo_i > 0 && !f->hit_done && d > MIN_DIST + 100)
			f->x += dir * 44 * g_step;                     /* chained attack: lunge to keep the range */
		/* multi-hit moves re-arm every `rehit` percent */
		if (mv->rehit && f->hit_done && pct >= f->rehit_at)
			f->hit_done = 0;
		/* aim: turn towards the opponent so the striking limb swings through
		 * their axis (look-at correction, measured on the actual pose) */
		int sb = strike_bone(f, dir);
		VECTOR spt = bone_world_yaw(f, sb, f->yaw);
		int fwd = (spt.vx - f->x) * dir;
		if (mv->limb != LIMB_BODY && fwd > 300 && pct <= mv->hit_to) {
			int err0 = spt.vz - o->z;
			VECTOR alt = bone_world_yaw(f, sb, f->yaw + 32);
			int err1 = alt.vz - o->z;
			int step = (err0 < 0 ? -err0 : err0) > 60 ? 32 : 8;
			if ((err1 < 0 ? -err1 : err1) < (err0 < 0 ? -err0 : err0)) f->yaw_corr += step;
			else f->yaw_corr -= step;
			if (f->yaw_corr >  800) f->yaw_corr =  800;
			if (f->yaw_corr < -800) f->yaw_corr = -800;
			f->yaw = (base_yaw + f->yaw_corr) & 4095;
			spt = bone_world_yaw(f, sb, f->yaw);
			fwd = (spt.vx - f->x) * dir;
		}
		if (!f->hit_done && active) {
			/* the limb reaches the opponent's body: from the near surface to
			 * well past the axis (a kick that swings through still connects) */
			int lat = spt.vz - o->z;
			int limb_hit = fwd >= d - 450 && fwd <= d + 1300 && (lat < 0 ? -lat : lat) < 520 &&
			               spt.vy < 0 && spt.vy > -3900;
			if (mv->height == H_HIGH && spt.vy > -1400) limb_hit = 0;   /* high: above the waist */
			if (mv->height == H_LOW && spt.vy < -1100) limb_hit = 0;    /* low: near the floor */
			int range_hit = mv->limb == LIMB_FOOT ? 0 : d <= mv->reach - 300;
			if ((limb_hit || range_hit) && o->state != FS_KO && o->state != FS_DOWN) {
				f->hit_done = 1;
				f->rehit_at = pct + mv->rehit;
				/* DOGEZA RISK: instant death if hit during dogeza */
				int dogeza_death = (o->state == FS_DOGEZA);
				/* standing guard stops mid / high attacks (chip damage);
				 * low attacks and floor-takers go through */
				int guarded = o->state == FS_GUARD && mv->height != H_LOW && !(mv->flags & MF_KNOCKDOWN);
				int counter = o->state == FS_ATTACK;             /* hit them during their swing */
				int dmg = dogeza_death ? 999 : guarded ? (mv->dmg + 3) / 4 : counter ? (mv->dmg * 3) / 2 : mv->dmg;
				if (guarded && o->hp - dmg <= 0) dmg = o->hp - 1;   /* no chip KO */
				/* training mode: record damage, prevent KO */
				if (fg->training && i == 0) {
					fg->last_dmg = dmg > 999 ? 999 : dmg;
					if (o->hp - dmg < 1) dmg = o->hp - 1;
				}
				o->hp -= dmg;
				if (counter && !guarded) { fg->counter_t = 50; fg->counter_side = 1 - i; }
				/* hit effect where the strike lands: the opponent's body surface
				 * facing the attacker, at the height of the striking limb */
				int hx = o->x - dir * 280;
				if ((spt.vx - hx) * dir > 0) hx = spt.vx;          /* limb already inside: use it */
				for (int k = 0; k < MAX_FX; k++) {
					if (fg->fx[k].active) continue;
					fg->fx[k].active = 1; fg->fx[k].t = guarded ? 8 : 0;   /* guard: small blue-ish burst */
					fg->fx[k].x = hx; fg->fx[k].y = spt.vy; fg->fx[k].z = (spt.vz + o->z) / 2;
					break;
				}
				fg->shake = guarded ? 3 : 10 + mv->stop;
				if (guarded) {
					o->kb = dir * (mv->kb / 3);
					o->guard_t += 6;
					fg->hitstop = 2;
				} else if (o->hp <= 0 && !fg->training) {
					o->hp = 0;
					o->state = FS_KO;
					set_anim(o, o->anim_ko);
				} else if (mv->flags & (MF_KNOCKDOWN | MF_LAUNCH)) {
					o->state = FS_DOWN;                     /* sweeps / launchers floor them */
					o->down_phase = 0;
					set_anim(o, o->anim_fall);
					o->pose.speed = (mv->flags & MF_LAUNCH) ? 384 : 256;
				} else {
					o->state = FS_HIT;
					set_anim(o, o->anim_hit);              /* restarts on every hit */
					o->pose.speed = 512;
				}
				if (!guarded) {
					o->hits_taken++;
					int last = f->combo_i + 1 >= f->combo_len;
					o->kb = dir * (last ? mv->kb : mv->kb / 2);   /* mid-combo: keep them in reach */
					if (mv->flags & MF_LAUNCH) o->kb = dir * (mv->kb + 120);
					fg->hitstop = counter ? mv->stop + 4 : mv->stop;
				}
			} else if (pct > mv->hit_to - 8 && !mv->rehit) {
				f->hit_done = 1;
			}
		}
		/* combos cut the recovery half of each clip: chain right after the hit */
		if (f->pose.loops > 0 || (pct >= 55 && f->combo_i + 1 < f->combo_len)) {
			if (f->combo_i + 1 < f->combo_len && o->state != FS_KO && o->state != FS_DOWN) {
				f->combo_i++;
				start_attack(fg, i, f->combo[f->combo_i]);
			} else {
				f->state = FS_IDLE;
				f->cooldown = f->human ? 6 : 10 + (rnd(fg) % 25);
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
				f->hits_taken = 0;
				f->state = FS_IDLE;
				f->cooldown = 12 + (rnd(fg) % 16);
				set_anim(f, f->anim_idle);
			}
		}
		break;
	case FS_HIT:
		if (f->pose.loops > 0) {
			f->hits_taken = 0;
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
	case FS_DOGEZA: {
		f->dogeza_t += g_step;
		/* hold the pose (slow down animation after first loop) */
		if (f->pose.loops > 0) {
			f->pose.speed = 0;
			f->pose.playing = 0;
		}
		/* speech bubble appears after startup */
		if (f->dogeza_t == DOGEZA_STARTUP) {
			fg->dogeza_bubble_t = DOGEZA_ACTIVE;
			fg->dogeza_bubble_side = i;
		}
		/* check if dogeza is finished */
		if (f->dogeza_t >= DOGEZA_TOTAL) {
			f->state = FS_IDLE;
			f->cooldown = 30;
			set_anim(f, f->anim_idle);
			f->dogeza_t = 0;
		}
		break;
	}
	}
	/* ring out: knocked over the stage edge while flying (disabled in training) */
	if (!fg->training && (f->x < -STAGE_EDGE || f->x > STAGE_EDGE) && f->kb &&
	    (f->state == FS_HIT || f->state == FS_DOWN) && fg->phase == FP_FIGHTING) {
		f->state = FS_KO;
		f->hp = 0;
		fg->ringout = 1;
		set_anim(f, f->anim_fall);
		f->kb = 0;
	}
	if (f->x < -STAGE_CLAMP) f->x = -STAGE_CLAMP;
	if (f->x >  STAGE_CLAMP) f->x =  STAGE_CLAMP;
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
		want_dist = 5600;
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
		/* VF-style distance-based camera: close when fighting, pull back when far apart */
		want_yaw = (fsin(fg->t * 2) * 380) >> 12;        /* gentle sway, ~20 s period */
		/* close range (<2500): tight camera for fighting feel;
		   far range: pull back so both fighters are visible on the wider stage */
		if (sep < 2500) {
			want_dist = 4600;                            /* close combat: original feel */
		} else {
			want_dist = 3200 + sep;                      /* scale with separation */
			if (want_dist > 12000) want_dist = 12000;    /* max pullback for 2x stage */
		}
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
		if (!fg->training) {
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
				fg->match_over = 1;
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
	fg->f[0].in_pressed = fg->f[1].in_pressed = 0;   /* presses live for one update */
	if (fg->hitstop > 0) fg->hitstop -= g_step;
	if (fg->name_t > 0) fg->name_t -= g_step;
	if (fg->counter_t > 0) fg->counter_t -= g_step;
	/* advice popup: random timing during fights (disabled in training) */
	if (fg->phase == FP_FIGHTING && !fg->training) {
		if (fg->advice_t > 0) {
			fg->advice_t -= g_step;
		} else if (fg->advice_cd > 0) {
			fg->advice_cd -= g_step;
		} else {
			fg->advice_idx = rnd(fg) % NUM_ADVICE;
			fg->advice_t = ADVICE_SHOW_FRAMES;
			fg->advice_cd = ADVICE_CD_MIN + (rnd(fg) % (ADVICE_CD_MAX - ADVICE_CD_MIN));
		}
	}

	/* ---- projectile (hadouken) update ------------------------------------ */
	for (int k = 0; k < MAX_PROJECTILES; k++) {
		Projectile *p = &fg->proj[k];
		if (!p->active) continue;

		/* move the projectile */
		p->x += p->vx * g_step;
		p->life -= g_step;

		/* disappear at range limit or stage edge */
		if (p->life <= 0 || p->x < -STAGE_EDGE || p->x > STAGE_EDGE) {
			p->active = 0;
			continue;
		}

		/* hit detection against the opponent */
		Fighter *o = &fg->f[1 - p->owner];
		int dx = p->x - o->x;
		if (dx < 0) dx = -dx;
		int dz = p->z - o->z;
		if (dz < 0) dz = -dz;

		if (dx < HADOUKEN_HITBOX && dz < 300 && o->state != FS_KO && o->state != FS_DOWN) {
			/* check if guarded */
			int guarded = o->state == FS_GUARD;
			int dmg = guarded ? (p->dmg + 3) / 4 : p->dmg;
			if (guarded && o->hp - dmg <= 0) dmg = o->hp - 1;  /* no chip KO */
			/* training mode: record damage, prevent KO */
			if (fg->training && p->owner == 0) {
				fg->last_dmg = dmg > 999 ? 999 : dmg;
				if (o->hp - dmg < 1) dmg = o->hp - 1;
			}
			o->hp -= dmg;

			/* hit effect */
			int dir = p->vx > 0 ? 1 : -1;
			for (int j = 0; j < MAX_FX; j++) {
				if (fg->fx[j].active) continue;
				fg->fx[j].active = 1;
				fg->fx[j].t = guarded ? 8 : 0;
				fg->fx[j].x = o->x - dir * 280;
				fg->fx[j].y = p->y;
				fg->fx[j].z = o->z;
				break;
			}

			fg->shake = guarded ? 2 : 6;
			if (guarded) {
				o->kb = dir * (HADOUKEN_KB / 3);
				o->guard_t += 4;
				fg->hitstop = 2;
			} else if (o->hp <= 0 && !fg->training) {
				o->hp = 0;
				o->state = FS_KO;
				set_anim(o, o->anim_ko);
			} else {
				o->state = FS_HIT;
				set_anim(o, o->anim_hit);
				o->pose.speed = 512;
				o->hits_taken++;
				o->kb = dir * HADOUKEN_KB;
				fg->hitstop = 3;
			}

			p->active = 0;  /* projectile consumed on hit */
		}
	}

	for (int i = 0; i < 2; i++) {
		Fighter *f = &fg->f[i], *o = &fg->f[1 - i];
		if (f->state == FS_ATTACK && (MOVES[f->move].flags & MF_NOLOOK)) continue;   /* inverted: no look-at */
		VECTOR oh = head_world(o);
		/* into this fighter's model space: Ry(-yaw) * (p - pos) */
		SVECTOR r = { 0, (-f->yaw) & 4095, 0, 0 };
		MATRIX m4;
		RotMatrix(&r, &m4);
		VECTOR d = vec(oh.vx - f->x, oh.vy, oh.vz - f->z), local;
		ApplyMatrixLV(&m4, &d, &local);
		ik_look_at_point(f->model, &f->pose, local);
	}

	/* ---- dogeza speech bubble hit detection ----------------------------- */
	if (fg->dogeza_bubble_t > 0) {
		fg->dogeza_bubble_t -= g_step;
		Fighter *f = &fg->f[fg->dogeza_bubble_side];
		Fighter *o = &fg->f[1 - fg->dogeza_bubble_side];

		if (!f->dogeza_hit && o->state != FS_KO && o->state != FS_DOWN) {
			int dir = (o->x > f->x) ? 1 : -1;
			int dx = (o->x - f->x) * dir;
			int bubble_range = 1200;
			if (dx < bubble_range && dx > 200) {
				f->dogeza_hit = 1;
				int dmg = DOGEZA_DMG;
				/* training mode: record damage, prevent KO */
				if (fg->training && fg->dogeza_bubble_side == 0) {
					fg->last_dmg = dmg > 999 ? 999 : dmg;
					if (o->hp - dmg < 1) dmg = o->hp - 1;
				}
				o->hp -= dmg;
				fg->shake = 20;
				fg->hitstop = 12;
				fg->name_t = 60;
				fg->name_move = -1;
				fg->name_side = fg->dogeza_bubble_side;

				for (int k = 0; k < MAX_FX; k++) {
					if (fg->fx[k].active) continue;
					fg->fx[k].active = 1;
					fg->fx[k].t = 0;
					fg->fx[k].x = (f->x + o->x) / 2;
					fg->fx[k].y = -2000;
					fg->fx[k].z = 0;
					break;
				}

				if (o->hp <= 0 && !fg->training) {
					o->hp = 0;
					o->state = FS_KO;
					set_anim(o, o->anim_ko);
				} else {
					o->state = FS_DOWN;
					o->down_phase = 0;
					set_anim(o, o->anim_fall);
					o->pose.speed = 384;
					o->kb = dir * DOGEZA_KB;
				}
			}
		}
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

/* hadouken projectile: glowing energy ball (blue/cyan gradient) */
static char *draw_projectiles(Fight *fg, const Camera *cam, uint32_t *ot, char *nextpri) {
	TILE *t = (TILE *)nextpri;
	gte_SetRotMatrix(&cam->view);
	gte_SetTransMatrix(&cam->view);

	for (int k = 0; k < MAX_PROJECTILES; k++) {
		Projectile *p = &fg->proj[k];
		if (!p->active) continue;

		SVECTOR v = { p->x, p->y, p->z, 0 };
		uint32_t sxy; int32_t sz;
		gte_ldv0(&v); gte_rtps();
		__asm__ volatile("swc2 $14, 0(%0); swc2 $19, 0(%1)" :: "r"(&sxy), "r"(&sz) : "memory");
		if (sz <= 0) continue;

		int cx = (int16_t)(sxy & 0xffff), cy = (int16_t)(sxy >> 16);

		/* outer glow (cyan, semi-transparent look via dithering) */
		setTile(t); setRGB0(t, 60, 180, 220);
		setXY0(t, cx - 10, cy - 10); setWH(t, 20, 20);
		addPrim(ot, t); t++;

		/* middle ring (brighter cyan) */
		setTile(t); setRGB0(t, 100, 220, 255);
		setXY0(t, cx - 7, cy - 7); setWH(t, 14, 14);
		addPrim(ot, t); t++;

		/* inner core (white-ish) */
		setTile(t); setRGB0(t, 200, 255, 255);
		setXY0(t, cx - 4, cy - 4); setWH(t, 8, 8);
		addPrim(ot, t); t++;

		/* bright center */
		setTile(t); setRGB0(t, 255, 255, 255);
		setXY0(t, cx - 2, cy - 2); setWH(t, 4, 4);
		addPrim(ot, t); t++;
	}
	return (char *)t;
}

static char *draw_hud(Fight *fg, uint32_t *ot, char *nextpri) {
	TILE *t = (TILE *)nextpri;
	if (fg->training) {
		/* training mode: "TRAINING" at top center, DMG display at bottom left */
		nextpri = fight_text("TRAINING", CENTERX, 8, 2, 80, 255, 120, ot, nextpri);
		t = (TILE *)nextpri;
		setTile(t); setRGB0(t, 20, 20, 30); setXY0(t, CENTERX - 50, 5); setWH(t, 100, 20); addPrim(ot, t); t++;
		nextpri = (char *)t;
		/* DMG display: "DMG 000" format */
		char dmg_buf[8];
		int d = fg->last_dmg;
		dmg_buf[0] = 'D'; dmg_buf[1] = 'M'; dmg_buf[2] = 'G'; dmg_buf[3] = ' ';
		dmg_buf[4] = '0' + (d / 100) % 10;
		dmg_buf[5] = '0' + (d / 10) % 10;
		dmg_buf[6] = '0' + d % 10;
		dmg_buf[7] = 0;
		nextpri = fight_text(dmg_buf, 50, SCREEN_YRES - 20, 2, 255, 200, 100, ot, nextpri);
	} else {
		/* timer: drawn first in bucket 0 (= on top), on a dark box */
		char buf[4];
		int secs = (fg->timer + 59) / 60;
		buf[0] = '0' + (secs / 10) % 10; buf[1] = '0' + secs % 10; buf[2] = 0;
		nextpri = fight_text(buf, CENTERX, 8, 3, 255, 255, 255, ot, nextpri);
		t = (TILE *)nextpri;
		setTile(t); setRGB0(t, 20, 20, 30); setXY0(t, CENTERX - 20, 5); setWH(t, 40, 28); addPrim(ot, t); t++;
		nextpri = (char *)t;
	}

	t = (TILE *)nextpri;
	/* life bars: 130 px wide, 12 px thick; yellow = remaining, red = lost.
	 * Bucket 0 draws the newest primitive first, so the fill goes in before
	 * the red background. Orange when HP <= 30% (pinch mode). */
	for (int i = 0; i < 2; i++) {
		int x0 = i ? SCREEN_XRES - 16 - 120 : 16;
		int w = (fg->f[i].hp_disp * 120) / 100;
		int pinch = fg->f[i].hp <= DOGEZA_HP_THRESHOLD;
		if (w > 0) {
			if (pinch) {
				setTile(t); setRGB0(t, 255, 140, 40);
			} else {
				setTile(t); setRGB0(t, 255, 220, 40);
			}
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
	/* announcements (skipped in training mode) */
	if (!fg->training) {
		switch (fg->phase) {
		case FP_ROUND: {
			char r[8] = "ROUND 1";
			r[6] = '0' + fg->round;
			if (fg->phase_t > 10) nextpri = fight_text(r, CENTERX, 90, 4, 255, 220, 60, ot, nextpri);
			break;
		}
		case FP_FIGHT:
			nextpri = fight_text("FIGHT!", CENTERX, 88, 5, 255, 80, 60, ot, nextpri);
			break;
		case FP_KO:
			if (fg->phase_t > 5)
				nextpri = fight_text(fg->ringout ? "RING OUT" : fg->timer == 0 ? "TIME UP" : "K.O.", CENTERX, 88,
				                   fg->ringout ? 4 : 5, 255, 60, 60, ot, nextpri);
			break;
		case FP_END: {
			const char *s = fg->winner == 0 ? "P1 WIN" : "P2 WIN";
			nextpri = fight_text(s, CENTERX, 90, 4, 255, 220, 60, ot, nextpri);
			if (fg->winner >= 0 && fg->f[fg->winner].hp == 100)
				nextpri = fight_text("PERFECT", CENTERX, 66, 2, 255, 255, 255, ot, nextpri);
			break;
		}
		default:
			break;
		}
	}
	/* hit counter under the life bar of whoever is being juggled */
	for (int i = 0; i < 2; i++) {
		int n = fg->f[i].hits_taken;
		if (n < 2 || (fg->f[i].state != FS_HIT && fg->f[i].state != FS_DOWN)) continue;
		char hb[8];
		int k = 0;
		if (n >= 10) hb[k++] = '0' + n / 10;
		hb[k++] = '0' + n % 10; hb[k++] = ' '; hb[k++] = 'H'; hb[k++] = 'I'; hb[k++] = 'T'; hb[k++] = 'S'; hb[k] = 0;
		nextpri = fight_text(hb, i ? SCREEN_XRES - 60 : 60, 40, 2, 255, 160, 40, ot, nextpri);
	}
	if (fg->counter_t > 0)
		nextpri = fight_text("COUNTER!", fg->counter_side ? SCREEN_XRES - 70 : 70, 58, 2, 255, 80, 200, ot, nextpri);
	/* move name popup, on the side of the fighter using it */
	if (fg->name_t > 0) {
		char nm[20];
		if (fg->name_move < 0) {
			nm[0] = 'D'; nm[1] = 'O'; nm[2] = 'G'; nm[3] = 'E'; nm[4] = 'Z'; nm[5] = 'A'; nm[6] = '!'; nm[7] = 0;
		} else {
			const char *src = MOVES[fg->name_move].name;
			int k = 0;
			for (; src[k] && k < 19; k++) {
				char c = src[k];
				nm[k] = c == '_' ? ' ' : (c >= 'a' && c <= 'z') ? c - 32 : c;
			}
			nm[k] = 0;
		}
		nextpri = fight_text(nm, fg->name_side ? SCREEN_XRES - 84 : 84, 178, 2,
		                   255, 255, fg->name_t > 40 ? 255 : 120, ot, nextpri);
	}
	/* dogeza speech bubble: SUMIMASEN! appears above the fighter's head */
	if (fg->dogeza_bubble_t > 0) {
		int side = fg->dogeza_bubble_side;
		int cx = side ? SCREEN_XRES - 80 : 80;
		int flash = (fg->dogeza_bubble_t / 4) & 1;
		int r = flash ? 255 : 220;
		int g = flash ? 100 : 60;
		int b = flash ? 100 : 60;
		nextpri = fight_text("SUMIMASEN!", cx, 140, 2, r, g, b, ot, nextpri);
	}
	/* advice popup ("goiken"): appears at random intervals at top center */
	if (fg->advice_t > 0) {
		int fade = fg->advice_t > 150 ? (180 - fg->advice_t) * 255 / 30 :
		           fg->advice_t < 30 ? fg->advice_t * 255 / 30 : 255;
		if (fade > 255) fade = 255;
		if (fade < 0) fade = 0;
		int r = (180 * fade) / 255;
		int g = (220 * fade) / 255;
		int b = (255 * fade) / 255;
		nextpri = fight_text(ADVICE[fg->advice_idx], CENTERX, 200, 1, r, g, b, ot, nextpri);
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
	nextpri = draw_projectiles(fg, cam, ot, nextpri);
	if (fg->demo) return nextpri;
	return draw_hud(fg, ot, nextpri);
}
