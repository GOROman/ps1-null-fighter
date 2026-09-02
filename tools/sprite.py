#!/usr/bin/env python3
"""Generate the monkey sprite sheet as a 4bpp TIM.

64x16 texels: frame 0/1 facing right, frame 2/3 facing left (walk cycle).
VRAM: pixels at (896, 0) (third texture page), CLUT at (640, 258).
"""
import struct
import sys

# 16x16 pixel art, characters map to palette indices below
MONKEY = [
    "................",
    ".....BBBBBB.....",
    "....BBBBBBBB....",
    "...BBTTTTTTBB...",
    "..BBTTKTTKTTBB..",
    "..BBTTTTTTTTBB..",
    "...BBTTTTTTBB...",
    "....BBTMMTBB....",
    ".....BBBBBB.....",
    "....BBBBBBBB....",
    "...BBBBBBBBBBB..",
    "...BBBBBBBBB.B..",
    "....BBBBBBBB....",
    "....BBB..BBB....",
    "....BB....BB....",
    "...BBB....BBB...",
]
MONKEY2 = [  # second walk frame: legs together, tail up
    "................",
    ".....BBBBBB.....",
    "....BBBBBBBB....",
    "...BBTTTTTTBB...",
    "..BBTTKTTKTTBB..",
    "..BBTTTTTTTTBB..",
    "...BBTTTTTTBB...",
    "....BBTMMTBB....",
    ".....BBBBBB..B..",
    "....BBBBBBBB.B..",
    "...BBBBBBBBBBB..",
    "...BBBBBBBBBB...",
    "....BBBBBBBB....",
    ".....BBBBBB.....",
    ".....BB..BB.....",
    "....BBB..BBB....",
]
PALETTE = {
    ".": (0, 0, 0, True),        # transparent
    "B": (120, 72, 32, False),   # brown fur
    "T": (232, 190, 150, False), # tan face
    "K": (16, 16, 16, False),    # eyes
    "M": (140, 60, 60, False),   # mouth
}


def tartan(size=128, period=32):
    """Navy tartan with red / white / green lines, tiled every `period` texels."""
    from PIL import Image
    img = Image.new("RGB", (size, size))
    px = img.load()
    base = (34, 40, 78)
    lines = [(0, (200, 40, 50), 3), (12, (220, 220, 220), 1), (20, (60, 140, 80), 2)]
    for y in range(size):
        for x in range(size):
            r, g, b = base
            for (off, col, w) in lines:
                for k in (x, y):
                    if (k - off) % period < w:
                        r, g, b = (r + col[0]) // 2, (g + col[1]) // 2, (b + col[2]) // 2
            # subtle weave
            if (x + y) % 2:
                r, g, b = int(r * 0.92), int(g * 0.92), int(b * 0.92)
            px[x, y] = (r, g, b)
    return img


def main(out_path):
    keys = list(PALETTE)
    frames = [MONKEY, MONKEY2, [r[::-1] for r in MONKEY], [r[::-1] for r in MONKEY2]]
    w, h = 16 * len(frames), 16
    idx = bytearray()
    for y in range(h):
        row = []
        for f in frames:
            row += [keys.index(c) for c in f[y]]
        for i in range(0, w, 2):          # 4bpp: two texels per byte, low nibble first
            idx.append(row[i] | (row[i + 1] << 4))
    clut = []
    for k in keys:
        r, g, b, transparent = PALETTE[k]
        c = 0 if transparent else ((r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10)) or 0x8000
        clut.append(c)
    clut += [0] * (16 - len(clut))
    clut_blob = struct.pack("<16H", *clut)
    out = bytearray()
    out += struct.pack("<II", 0x10, 0x08)                                   # 4bpp + CLUT
    out += struct.pack("<IHHHH", 12 + len(clut_blob), 640, 258, 16, 1) + clut_blob
    out += struct.pack("<IHHHH", 12 + len(idx), 896, 0, w // 4, h) + bytes(idx)  # 4bpp: 4 texels / word
    with open(out_path, "wb") as f:
        f.write(out)
    print("wrote %s: %dx%d 4bpp, %d bytes" % (out_path, w, h, len(out)))


if __name__ == "__main__":
    main(sys.argv[1])
