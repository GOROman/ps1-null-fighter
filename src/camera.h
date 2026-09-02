#ifndef CAMERA_H
#define CAMERA_H

#include <psxgte.h>

typedef struct {
	int yaw, pitch;        /* 0..4095 = 360 degrees */
	int dist;              /* distance from target, PS1 units */
	int fov;               /* GTE projection distance (h): larger = narrower angle */
	VECTOR target;         /* orbit centre in model space */
	MATRIX view;           /* model space -> camera space */
} Camera;

void camera_init(Camera *c);
void camera_orbit(Camera *c, int dyaw, int dpitch, int ddist);
/* move the orbit centre up/down (dy < 0 = up on screen) and change the FOV */
void camera_pan(Camera *c, int dy);
void camera_zoom(Camera *c, int dfov);
/* rebuild c->view; model_yaw rotates the model about its vertical axis */
void camera_update(Camera *c, int model_yaw);

#endif
