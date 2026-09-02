/* IK mode: two-bone (law of cosines) IK for both hands and feet, the hip
 * bone driven by the d-pad, and the head turned towards the camera. */
#ifndef IK_H
#define IK_H

#include <psxgte.h>
#include "model.h"
#include "anim.h"
#include "camera.h"

typedef struct {
	int active;
	VECTOR target[IK_CHAINS];      /* effector positions (model space), may be animated */
	VECTOR base[IK_CHAINS];        /* effector positions captured on entry */
	int has[IK_CHAINS];
} IKState;

/* freeze the current pose, capture the effector positions as targets */
void ik_enter(IKState *s, const Model *m, Pose *pose);
void ik_leave(IKState *s, Pose *pose);
/* call after pose_eval: solve the chains and the head look-at */
void ik_apply(IKState *s, const Model *m, Pose *pose, const Camera *cam);
/* head look-at only (used every frame for characters that always face the camera) */
void ik_look_at(const Model *m, Pose *pose, const Camera *cam);

#endif
