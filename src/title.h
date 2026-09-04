/* Title screen: logo + mode menu drawn over the CPU vs CPU demo fight. */
#ifndef TITLE_H
#define TITLE_H

#include <stdint.h>
#include <psxgpu.h>

enum { MODE_1P = 0, MODE_VS, MODE_CPU, NUM_MODES };

typedef struct {
	int sel;               /* highlighted menu entry */
	int t;                 /* frames on the title */
} Title;

void title_init(Title *tt);
/* d-pad / start / buttons; returns the chosen MODE_* or -1 */
int  title_update(Title *tt, uint16_t pressed);
/* the whole overlay (dims the scene behind); OT bucket 0 */
char *title_draw(const Title *tt, uint32_t *ot, char *nextpri);

#endif
