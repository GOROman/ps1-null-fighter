#include <psxgpu.h>
#include <psxpad.h>
#include "title.h"
#include "fight.h"
#include "build_info.h"

#define SCREEN_XRES 320
#define SCREEN_YRES 240
#define CENTERX     (SCREEN_XRES >> 1)

static const char *const MENU[NUM_MODES] = { "1P MODE", "VS MODE", "CPU MODE" };

static char build_banner[64];
static int build_banner_init = 0;

/* advice phrases shared with fight.c */
static const char *const TITLE_ADVICE[] = {
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
#define NUM_TITLE_ADVICE ((int)(sizeof(TITLE_ADVICE) / sizeof(TITLE_ADVICE[0])))
#define TITLE_ADVICE_SHOW  180
#define TITLE_ADVICE_CD_MIN 240
#define TITLE_ADVICE_CD_MAX 480

static uint32_t title_rnd(Title *tt) {
	tt->rng = tt->rng * 1664525u + 1013904223u;
	return tt->rng >> 8;
}

void title_init(Title *tt) {
	tt->sel = 0;
	tt->t = 0;
	tt->rng = 0xDEADBEEF;
	tt->advice_t = 0;
	tt->advice_idx = 0;
	tt->advice_cd = TITLE_ADVICE_CD_MIN + (tt->rng % (TITLE_ADVICE_CD_MAX - TITLE_ADVICE_CD_MIN));
	if (!build_banner_init) {
		/* format: "V0.1.0 ABC1234 2026.09.04 10.15" - version commit date time */
		char *p = build_banner;
		const char *v = BUILD_VERSION;
		const char *c = BUILD_COMMIT;
		const char *d = BUILD_DATE;
		/* copy version (uppercase, limit length) */
		int n = 0;
		while (*v && n < 10) {
			char ch = *v++;
			if (ch >= 'a' && ch <= 'z') ch -= 32;
			if (ch == '-') ch = '.';
			*p++ = ch; n++;
		}
		*p++ = ' ';
		/* copy commit (7 chars max) */
		n = 0;
		while (*c && n < 7) {
			char ch = *c++;
			if (ch >= 'a' && ch <= 'z') ch -= 32;
			*p++ = ch; n++;
		}
		*p++ = ' ';
		/* copy date and time: "2026-09-04 10:15" -> "2026.09.04 10.15" */
		n = 0;
		while (*d && n < 16) {
			char ch = *d++;
			if (ch == '-' || ch == ':') ch = '.';
			*p++ = ch; n++;
		}
		*p = 0;
		build_banner_init = 1;
	}
}

int title_update(Title *tt, uint16_t pressed) {
	tt->t++;
	/* advice popup timing */
	if (tt->advice_t > 0) {
		tt->advice_t--;
	} else if (tt->advice_cd > 0) {
		tt->advice_cd--;
	} else {
		tt->advice_idx = title_rnd(tt) % NUM_TITLE_ADVICE;
		tt->advice_t = TITLE_ADVICE_SHOW;
		tt->advice_cd = TITLE_ADVICE_CD_MIN + (title_rnd(tt) % (TITLE_ADVICE_CD_MAX - TITLE_ADVICE_CD_MIN));
	}
	if (pressed & PAD_UP)   tt->sel = (tt->sel + NUM_MODES - 1) % NUM_MODES;
	if (pressed & PAD_DOWN) tt->sel = (tt->sel + 1) % NUM_MODES;
	if (pressed & (PAD_START | PAD_CROSS | PAD_CIRCLE | PAD_TRIANGLE))
		return tt->sel;
	return -1;
}

char *title_draw(const Title *tt, uint32_t *ot, char *nextpri) {
	/* bucket 0 draws the newest primitive first: text, then the dimming
	 * tile, then the tpage that selects 50 % blending for it (added last,
	 * executed first) */
	int pulse = (tt->t >> 3) & 1;
	nextpri = fight_text("NULL FIGHTER", CENTERX, 44, 4, 255, 236, 120, ot, nextpri);
	nextpri = fight_text("PLAYSTATION 3D FIGHTING", CENTERX, 84, 1, 200, 200, 220, ot, nextpri);
	for (int i = 0; i < NUM_MODES; i++) {
		int y = 128 + i * 22;
		int on = i == tt->sel;
		nextpri = fight_text(MENU[i], CENTERX, y, 2, on ? 255 : 150, on ? 240 : 150, on ? 80 : 170, ot, nextpri);
		if (on) {
			/* cursor: a small blinking block on each side of the entry */
			int w = 8 * 6 * 2;                       /* widest entry, "CPU MODE" */
			TILE *t = (TILE *)nextpri;
			int c = pulse ? 255 : 120;
			setTile(t); setRGB0(t, c, c, 60); setXY0(t, CENTERX - w / 2 - 16, y + 4); setWH(t, 6, 6); addPrim(ot, t); t++;
			setTile(t); setRGB0(t, c, c, 60); setXY0(t, CENTERX + w / 2 + 10, y + 4); setWH(t, 6, 6); addPrim(ot, t); t++;
			nextpri = (char *)t;
		}
	}
	/* advice popup ("goiken") */
	if (tt->advice_t > 0) {
		int fade = tt->advice_t > 150 ? (180 - tt->advice_t) * 255 / 30 :
		           tt->advice_t < 30 ? tt->advice_t * 255 / 30 : 255;
		if (fade > 255) fade = 255;
		if (fade < 0) fade = 0;
		int r = (180 * fade) / 255;
		int g = (220 * fade) / 255;
		int b = (255 * fade) / 255;
		nextpri = fight_text(TITLE_ADVICE[tt->advice_idx], CENTERX, 108, 1, r, g, b, ot, nextpri);
	}
	if (pulse || tt->t < 60)
		nextpri = fight_text("PUSH START BUTTON", CENTERX, 206, 1, 230, 230, 230, ot, nextpri);
	nextpri = fight_text("2026 GOROMAN", CENTERX, 218, 1, 140, 140, 160, ot, nextpri);
	/* build metadata banner: version commit date */
	nextpri = fight_text(build_banner, CENTERX, 228, 1, 100, 100, 120, ot, nextpri);
	/* dim the demo fight behind the menu: 50 % black over the whole screen */
	TILE *dim = (TILE *)nextpri;
	setTile(dim);
	setSemiTrans(dim, 1);
	setRGB0(dim, 0, 0, 0);
	setXY0(dim, 0, 0);
	setWH(dim, SCREEN_XRES, SCREEN_YRES);
	addPrim(ot, dim);
	nextpri = (char *)(dim + 1);
	DR_TPAGE *tp = (DR_TPAGE *)nextpri;
	setDrawTPage(tp, 0, 1, getTPage(0, 0, 0, 0));     /* abr 0: B/2 + F/2 */
	addPrim(ot, tp);
	return (char *)(tp + 1);
}
