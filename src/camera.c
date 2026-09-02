#include "camera.h"

void camera_init(Camera *c) {
	c->yaw = 0;
	c->pitch = 200;         /* slightly above */
	c->dist = 4000;         /* character (height 4096) fills ~2/3 of the screen */
	c->fov = 160;           /* == SCREEN_XRES/2: 90 degree horizontal FOV */
	c->target.vx = 0;
	c->target.vy = -2100;   /* character centre (y is down, height ~4096) */
	c->target.vz = 0;
}

void camera_orbit(Camera *c, int dyaw, int dpitch, int ddist) {
	c->yaw = (c->yaw + dyaw) & 4095;
	c->pitch += dpitch;
	if (c->pitch >  1000) c->pitch =  1000;
	if (c->pitch < -1000) c->pitch = -1000;
	c->dist += ddist;
	if (c->dist < 1600)  c->dist = 1600;
	if (c->dist > 24000) c->dist = 24000;
}

void camera_pan(Camera *c, int dy) {
	c->target.vy += dy;
	if (c->target.vy < -6000) c->target.vy = -6000;
	if (c->target.vy >  2000) c->target.vy =  2000;
}

void camera_zoom(Camera *c, int dfov) {
	c->fov += dfov;
	if (c->fov < 64)   c->fov = 64;     /* very wide */
	if (c->fov > 1200) c->fov = 1200;   /* telephoto */
}

void camera_update(Camera *c, int model_yaw) {
	/* view = Rx(pitch) * Ry(yaw + model_yaw), then translate so that the
	 * target sits at (0, 0, dist) in front of the camera. */
	SVECTOR r = { c->pitch, (c->yaw + model_yaw) & 4095, 0, 0 };
	VECTOR  t;
	RotMatrix(&r, &c->view);
	ApplyMatrixLV(&c->view, &c->target, &t);
	c->view.t[0] = -t.vx;
	c->view.t[1] = -t.vy;
	c->view.t[2] = -t.vz + c->dist;
}
