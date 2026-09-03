/* Frame profiler: hardware timer 1 counts horizontal blanks (263 per NTSC
 * frame).  Stages are recorded as timer deltas and drawn as a colour coded
 * bar on the left of the screen (see prof_draw in main.c). */
#ifndef PROF_H
#define PROF_H

#include <stdint.h>

enum {
	PROF_INPUT = 0,   /* pad, camera */
	PROF_POSE,        /* pose_eval */
	PROF_VERTS,       /* render pass 1: transform + light */
	PROF_PRIMS,       /* render pass 2: primitives */
	PROF_MISC,        /* text, texture preview */
	PROF_GPU,         /* DrawSync: waiting for the GPU */
	PROF_VSYNC,       /* VSync: idle */
	PROF_STAGES
};

#define TIMER1_COUNT (*(volatile uint16_t *)0x1F801110)
#define TIMER1_MODE  (*(volatile uint32_t *)0x1F801114)

extern uint16_t prof_stage[PROF_STAGES];   /* h-blanks per stage, accumulating over the current frame */
extern uint16_t prof_shown[PROF_STAGES];   /* completed previous frame (drawn / printed) */
extern uint16_t prof_last;

static inline void prof_init(void) {
	TIMER1_MODE = 0x0100;                  /* clock source = h-blank, free running */
	prof_last = TIMER1_COUNT;
}

/* close the given stage: everything since the previous mark belongs to it
 * (a stage may be marked several times per frame, e.g. two render passes) */
static inline void prof_mark(int stage) {
	uint16_t now = TIMER1_COUNT;
	prof_stage[stage] += (uint16_t)(now - prof_last);
	prof_last = now;
}

/* frame boundary: publish the finished frame and start accumulating anew */
static inline void prof_frame(void) {
	for (int i = 0; i < PROF_STAGES; i++) {
		prof_shown[i] = prof_stage[i];
		prof_stage[i] = 0;
	}
}

#endif
