#ifndef ANIM_H
#define ANIM_H

#include <psxgte.h>
#include "model.h"

#define MAX_BONES 64

typedef struct {
	const Model *model;
	int anim;              /* current animation index */
	int frame;             /* integer frame */
	int subframe;          /* 0..255 interpolation towards frame+1 */
	int playing;
	int speed;             /* playback speed, 256 = 1.0 */
	int loops;             /* how many times the animation wrapped since pose_set_anim */
	VECTOR look_smooth;    /* head look-at: filtered direction (model space) */
	int look_init;
	int hip_bone;          /* bone that receives hip_offset (-1: none) */
	VECTOR hip_offset;     /* extra local translation for hip_bone (IK mode) */
	MATRIX local[MAX_BONES];   /* bone -> parent space for the current frame */
	MATRIX world[MAX_BONES];   /* bone -> model space (rotation 4096 = 1.0, t in PS1 units) */
} Pose;

void pose_init(Pose *p, const Model *m, int anim);
void pose_set_anim(Pose *p, int anim);
/* advance by one display frame (60 Hz) */
void pose_step(Pose *p, int display_hz);
/* evaluate local + world matrices for the current frame/subframe */
void pose_eval(Pose *p);
/* world = parent_world * local */
void pose_compose(const MATRIX *parent_world, const MATRIX *local, MATRIX *out);
/* after some world matrices were overwritten (modified[b] != 0), recompute
 * every descendant of a modified bone from its local matrix */
void pose_refresh(Pose *p, const uint8_t *modified);

#endif
