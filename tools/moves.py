"""Move set for the fighting game: 32 moves.

Two kinds of entries:
  * clip moves  - use an existing animation clip of the rig ("box_01"...)
  * procedural  - a short keyframed motion generated here on top of the bind
                  pose.  Keys are per body role in degrees:
                    (pitch, yaw, roll)  pitch > 0 lifts the limb forward/up,
                    yaw > 0 turns the character right, roll > 0 swings the
                    limb out to the character's left.
                  "hip" keys are translations (left, up, forward) in PS1
                  units (1 m == 4096).
Both carry the gameplay parameters that fight.c reads from the generated
moves_table.h: damage, knock back, hit stop, playback speed, active window,
required limb height, travel, multi-hit re-arm, launcher / sweep flags.

Roles: hip (whole body), spine, head, ual/uar (upper arms), fal/far
(forearms), thl/thr (thighs), cal/car (calves).
"""
import math

import numpy as np

FPS = 15

# limb used for the hit test: 0 hand, 1 foot, 2 body (head / shoulder / knee)
HAND, FOOT, BODY = 0, 1, 2
# required height of the striking limb: 0 any, 1 above the waist (high), 2 below the knee (low)
ANY, HIGH, LOW = 0, 1, 2

MOVES = [
    # ---- punches -------------------------------------------------------
    dict(name="jab", clip="box_01", speed=3.0, hit=(18, 55), dmg=6, kb=150, stop=3, limb=HAND, height=ANY, travel=0, cat="punch", reach=1600),
    dict(name="straight", clip="box_01", speed=2.2, hit=(18, 60), dmg=9, kb=220, stop=5, limb=HAND, height=ANY, travel=30, cat="punch", reach=1700),
    dict(name="hook", dur=0.55, cat="punch", dmg=11, kb=240, stop=6, hit=(35, 70), limb=HAND, height=HIGH, travel=10, reach=1500,
         keys={"uar": [(0, (30, 0, -70)), (0.3, (60, 0, -90)), (0.4, (110, 0, 10)), (0.55, (20, 0, -20))],
               "far": [(0, (80, 0, 0)), (0.55, (80, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.25, (0, -25, 0)), (0.45, (0, 30, 0)), (0.55, (0, 0, 0))]}),
    dict(name="uppercut", dur=0.55, cat="punch", dmg=12, kb=200, stop=7, hit=(35, 65), limb=HAND, height=HIGH, travel=15, reach=1400, launch=1,
         keys={"uar": [(0, (20, 0, 0)), (0.25, (-20, 0, 0)), (0.45, (150, 0, 0)), (0.55, (60, 0, 0))],
               "far": [(0, (90, 0, 0)), (0.3, (100, 0, 0)), (0.45, (40, 0, 0)), (0.55, (60, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.25, (25, 0, 0)), (0.45, (-15, 0, 0)), (0.55, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.25, (0, -250, 0)), (0.45, (0, 200, 300)), (0.55, (0, 0, 0))]}),
    dict(name="backfist", dur=0.6, cat="punch", dmg=10, kb=260, stop=6, hit=(45, 75), limb=HAND, height=HIGH, travel=10, reach=1600,
         keys={"body": [(0, (0, 0, 0)), (0.3, (0, 120, 0)), (0.5, (0, 330, 0)), (0.6, (0, 360, 0))],
               "uar": [(0, (20, 0, 0)), (0.3, (60, 0, -40)), (0.5, (90, 0, -80)), (0.6, (20, 0, 0))],
               "far": [(0, (60, 0, 0)), (0.5, (0, 0, 0)), (0.6, (40, 0, 0))]}),
    dict(name="elbow", dur=0.45, cat="punch", dmg=10, kb=200, stop=6, hit=(35, 65), limb=HAND, height=HIGH, travel=25, reach=1200,
         keys={"uar": [(0, (20, 0, 0)), (0.2, (60, 0, -60)), (0.35, (100, 0, 40)), (0.45, (20, 0, 0))],
               "far": [(0, (60, 0, 0)), (0.2, (150, 0, 0)), (0.45, (150, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.2, (0, -30, 0)), (0.35, (0, 25, 0)), (0.45, (0, 0, 0))]}),
    dict(name="palm", dur=0.6, cat="punch", dmg=12, kb=340, stop=6, hit=(40, 70), limb=HAND, height=ANY, travel=25, reach=1500,
         keys={"ual": [(0, (0, 0, 0)), (0.3, (20, 0, 20)), (0.45, (95, 0, 0)), (0.6, (0, 0, 0))],
               "uar": [(0, (0, 0, 0)), (0.3, (20, 0, -20)), (0.45, (95, 0, 0)), (0.6, (0, 0, 0))],
               "fal": [(0, (0, 0, 0)), (0.3, (110, 0, 0)), (0.45, (10, 0, 0)), (0.6, (0, 0, 0))],
               "far": [(0, (0, 0, 0)), (0.3, (110, 0, 0)), (0.45, (10, 0, 0)), (0.6, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.3, (0, 0, -150)), (0.45, (0, 0, 250)), (0.6, (0, 0, 0))]}),
    dict(name="hammer", dur=0.6, cat="punch", dmg=13, kb=180, stop=7, hit=(45, 70), limb=HAND, height=ANY, travel=15, reach=1300, knockdown=1,
         keys={"uar": [(0, (0, 0, 0)), (0.3, (170, 0, -20)), (0.5, (60, 0, 0)), (0.6, (0, 0, 0))],
               "far": [(0, (0, 0, 0)), (0.3, (60, 0, 0)), (0.5, (30, 0, 0)), (0.6, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.3, (-15, 0, 0)), (0.5, (30, 0, 0)), (0.6, (0, 0, 0))]}),
    # ---- kicks ---------------------------------------------------------
    dict(name="front_kick", clip="front_kick_01", speed=2.0, hit=(28, 75), dmg=13, kb=240, stop=8, limb=FOOT, height=HIGH, travel=0, cat="kick", reach=1900),
    dict(name="high_kick", clip="front_kick_01", speed=2.6, hit=(30, 70), dmg=14, kb=200, stop=8, limb=FOOT, height=HIGH, travel=20, cat="kick", reach=1900),
    dict(name="roundhouse", dur=0.7, cat="kick", dmg=15, kb=280, stop=9, hit=(40, 75), limb=FOOT, height=HIGH, travel=15, reach=2000,
         keys={"thr": [(0, (0, 0, 0)), (0.25, (30, 0, -40)), (0.45, (95, 0, -60)), (0.6, (100, 0, 30)), (0.7, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.25, (-90, 0, 0)), (0.45, (-40, 0, 0)), (0.6, (0, 0, 0)), (0.7, (0, 0, 0))],
               "body": [(0, (0, 0, 0)), (0.3, (0, -30, 0)), (0.6, (0, 60, 0)), (0.7, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.45, (-20, 0, 20)), (0.7, (0, 0, 0))],
               "ual": [(0, (0, 0, 0)), (0.45, (40, 0, 40)), (0.7, (0, 0, 0))]}),
    dict(name="sweep", dur=0.6, cat="kick", dmg=10, kb=180, stop=7, hit=(35, 70), limb=FOOT, height=LOW, travel=20, reach=1700, knockdown=1,
         keys={"thr": [(0, (0, 0, 0)), (0.25, (60, 0, -50)), (0.45, (80, 0, 40)), (0.6, (0, 0, 0))],
               "thl": [(0, (0, 0, 0)), (0.2, (110, 0, 0)), (0.6, (0, 0, 0))],
               "cal": [(0, (0, 0, 0)), (0.2, (-120, 0, 0)), (0.6, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.2, (0, -1500, 0)), (0.5, (0, -1500, 0)), (0.6, (0, 0, 0))],
               "body": [(0, (0, 0, 0)), (0.25, (0, -40, 0)), (0.45, (0, 60, 0)), (0.6, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.2, (40, 0, 0)), (0.6, (0, 0, 0))]}),
    dict(name="knee", dur=0.45, cat="kick", dmg=11, kb=120, stop=6, hit=(30, 65), limb=BODY, height=ANY, travel=35, reach=1200,
         keys={"thr": [(0, (0, 0, 0)), (0.3, (120, 0, 0)), (0.45, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.3, (-130, 0, 0)), (0.45, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.3, (0, 150, 200)), (0.45, (0, 0, 0))],
               "ual": [(0, (0, 0, 0)), (0.3, (60, 0, 0)), (0.45, (0, 0, 0))],
               "uar": [(0, (0, 0, 0)), (0.3, (60, 0, 0)), (0.45, (0, 0, 0))]}),
    dict(name="axe_kick", dur=0.75, cat="kick", dmg=16, kb=140, stop=9, hit=(50, 75), limb=FOOT, height=ANY, travel=20, reach=1700, knockdown=1,
         keys={"thr": [(0, (0, 0, 0)), (0.35, (150, 0, 15)), (0.55, (60, 0, 0)), (0.75, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.2, (-60, 0, 0)), (0.35, (0, 0, 0)), (0.75, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.35, (-20, 0, 0)), (0.6, (25, 0, 0)), (0.75, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.35, (0, 200, 0)), (0.55, (0, -100, 150)), (0.75, (0, 0, 0))]}),
    dict(name="side_kick", dur=0.55, cat="kick", dmg=14, kb=300, stop=8, hit=(35, 65), limb=FOOT, height=ANY, travel=25, reach=2000,
         keys={"body": [(0, (0, 0, 0)), (0.2, (0, -80, 0)), (0.55, (0, 0, 0))],
               "thl": [(0, (0, 0, 0)), (0.2, (60, 0, 60)), (0.4, (60, 0, 100)), (0.55, (0, 0, 0))],
               "cal": [(0, (0, 0, 0)), (0.2, (-100, 0, 0)), (0.4, (0, 0, 0)), (0.55, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.4, (0, 0, -25)), (0.55, (0, 0, 0))]}),
    dict(name="back_kick", dur=0.65, cat="kick", dmg=15, kb=320, stop=9, hit=(45, 70), limb=FOOT, height=ANY, travel=10, reach=2000,
         keys={"body": [(0, (0, 0, 0)), (0.3, (0, 180, 0)), (0.65, (0, 360, 0))],
               "thl": [(0, (0, 0, 0)), (0.3, (-30, 0, 0)), (0.5, (-110, 0, 0)), (0.65, (0, 0, 0))],
               "cal": [(0, (0, 0, 0)), (0.3, (-90, 0, 0)), (0.5, (0, 0, 0)), (0.65, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.5, (45, 0, 0)), (0.65, (0, 0, 0))]}),
    dict(name="spin_kick", dur=0.8, cat="kick", dmg=17, kb=340, stop=10, hit=(55, 80), limb=FOOT, height=HIGH, travel=15, reach=2100,
         keys={"body": [(0, (0, 0, 0)), (0.35, (0, 200, 0)), (0.65, (0, 360, 0)), (0.8, (0, 360, 0))],
               "thl": [(0, (0, 0, 0)), (0.35, (20, 0, 20)), (0.55, (100, 0, -60)), (0.7, (110, 0, 40)), (0.8, (0, 0, 0))],
               "cal": [(0, (0, 0, 0)), (0.35, (-90, 0, 0)), (0.55, (-20, 0, 0)), (0.8, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.55, (-25, 0, 20)), (0.8, (0, 0, 0))]}),
    dict(name="jump_kick", dur=0.8, cat="kick", dmg=15, kb=280, stop=9, hit=(40, 70), limb=FOOT, height=ANY, travel=40, reach=1900,
         keys={"hip": [(0, (0, 0, 0)), (0.3, (0, 1200, 0)), (0.55, (0, 1300, 0)), (0.8, (0, 0, 0))],
               "thr": [(0, (0, 0, 0)), (0.3, (60, 0, 0)), (0.5, (110, 0, 0)), (0.8, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.3, (-110, 0, 0)), (0.5, (0, 0, 0)), (0.8, (0, 0, 0))],
               "thl": [(0, (0, 0, 0)), (0.3, (40, 0, 0)), (0.5, (30, 0, 0)), (0.8, (0, 0, 0))],
               "cal": [(0, (0, 0, 0)), (0.3, (-120, 0, 0)), (0.8, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.5, (-15, 0, 0)), (0.8, (0, 0, 0))]}),
    dict(name="double_kick", dur=0.8, cat="kick", dmg=9, kb=120, stop=6, hit=(25, 80), limb=FOOT, height=ANY, travel=15, reach=1800, rehit=35,
         keys={"thr": [(0, (0, 0, 0)), (0.2, (110, 0, 0)), (0.35, (30, 0, 0)), (0.55, (130, 0, 0)), (0.8, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.1, (-110, 0, 0)), (0.2, (0, 0, 0)), (0.42, (-110, 0, 0)), (0.55, (0, 0, 0)), (0.8, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.2, (-10, 0, 0)), (0.55, (-15, 0, 0)), (0.8, (0, 0, 0))]}),
    dict(name="crescent", dur=0.7, cat="kick", dmg=14, kb=260, stop=8, hit=(40, 70), limb=FOOT, height=HIGH, travel=15, reach=1900,
         keys={"thr": [(0, (0, 0, 0)), (0.3, (90, 0, -70)), (0.5, (130, 0, 0)), (0.6, (90, 0, 60)), (0.7, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.7, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.5, (-20, 0, 0)), (0.7, (0, 0, 0))]}),
    # ---- specials ------------------------------------------------------
    dict(name="sbk", clip="sbk", speed=1.0, hit=(20, 88), dmg=6, kb=40, stop=2, limb=FOOT, height=ANY, travel=30, cat="special", reach=1900, rehit=22, nolook=1),
    dict(name="hurricane", dur=1.2, cat="special", dmg=7, kb=60, stop=2, hit=(20, 90), limb=FOOT, height=HIGH, travel=28, reach=1900, rehit=25, nolook=1,
         keys={"body": [(0, (0, 0, 0)), (0.15, (0, 0, 0)), (1.05, (0, 1080, 0)), (1.2, (0, 1080, 0))],
               "thl": [(0, (0, 0, 0)), (0.15, (95, 0, 60)), (1.05, (95, 0, 60)), (1.2, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.15, (0, 700, 0)), (1.05, (0, 900, 0)), (1.2, (0, 0, 0))],
               "ual": [(0, (0, 0, 0)), (0.15, (60, 0, 80)), (1.2, (0, 0, 0))],
               "uar": [(0, (0, 0, 0)), (0.15, (60, 0, -80)), (1.2, (0, 0, 0))]}),
    dict(name="rising_upper", dur=0.9, cat="special", dmg=18, kb=200, stop=10, hit=(25, 60), limb=HAND, height=ANY, travel=20, reach=1500, launch=1, nolook=1,
         keys={"uar": [(0, (0, 0, 0)), (0.15, (-30, 0, 0)), (0.35, (170, 0, 0)), (0.7, (170, 0, 0)), (0.9, (0, 0, 0))],
               "far": [(0, (60, 0, 0)), (0.35, (20, 0, 0)), (0.9, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.15, (0, -300, 0)), (0.5, (0, 1600, 300)), (0.75, (0, 800, 400)), (0.9, (0, 0, 0))],
               "thl": [(0, (0, 0, 0)), (0.35, (100, 0, 0)), (0.75, (60, 0, 0)), (0.9, (0, 0, 0))],
               "cal": [(0, (0, 0, 0)), (0.35, (-120, 0, 0)), (0.9, (0, 0, 0))],
               "body": [(0, (0, 0, 0)), (0.5, (0, 180, 0)), (0.9, (0, 360, 0))]}),
    dict(name="somersault", dur=1.0, cat="special", dmg=16, kb=260, stop=9, hit=(35, 70), limb=FOOT, height=ANY, travel=5, reach=1700, launch=1, nolook=1,
         keys={"body": [(0, (0, 0, 0)), (0.3, (60, 0, 0)), (0.6, (300, 0, 0)), (0.85, (360, 0, 0)), (1.0, (360, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.3, (0, 1200, 0)), (0.6, (0, 1800, 0)), (0.85, (0, 300, 0)), (1.0, (0, 0, 0))],
               "thr": [(0, (0, 0, 0)), (0.3, (150, 0, 0)), (0.6, (60, 0, 0)), (1.0, (0, 0, 0))],
               "thl": [(0, (0, 0, 0)), (0.3, (60, 0, 0)), (0.6, (40, 0, 0)), (1.0, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.3, (-40, 0, 0)), (1.0, (0, 0, 0))],
               "ual": [(0, (0, 0, 0)), (0.3, (150, 0, 0)), (1.0, (0, 0, 0))],
               "uar": [(0, (0, 0, 0)), (0.3, (150, 0, 0)), (1.0, (0, 0, 0))]}),
    dict(name="flying_knee", dur=0.8, cat="special", dmg=15, kb=240, stop=8, hit=(30, 70), limb=BODY, height=ANY, travel=45, reach=1300, nolook=1,
         keys={"hip": [(0, (0, 0, 0)), (0.3, (0, 1100, 0)), (0.6, (0, 900, 0)), (0.8, (0, 0, 0))],
               "thr": [(0, (0, 0, 0)), (0.3, (130, 0, 0)), (0.6, (120, 0, 0)), (0.8, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.3, (-140, 0, 0)), (0.8, (0, 0, 0))],
               "thl": [(0, (0, 0, 0)), (0.3, (-30, 0, 0)), (0.8, (0, 0, 0))],
               "ual": [(0, (0, 0, 0)), (0.3, (-40, 0, 0)), (0.8, (0, 0, 0))],
               "uar": [(0, (0, 0, 0)), (0.3, (-40, 0, 0)), (0.8, (0, 0, 0))]}),
    dict(name="slide", dur=0.8, cat="special", dmg=12, kb=200, stop=7, hit=(25, 75), limb=FOOT, height=LOW, travel=50, reach=1800, knockdown=1, nolook=1,
         keys={"hip": [(0, (0, 0, 0)), (0.2, (0, -1800, 0)), (0.6, (0, -2000, 0)), (0.8, (0, 0, 0))],
               "body": [(0, (0, 0, 0)), (0.2, (-60, 0, 0)), (0.6, (-70, 0, 0)), (0.8, (0, 0, 0))],
               "thr": [(0, (0, 0, 0)), (0.2, (80, 0, 0)), (0.6, (80, 0, 0)), (0.8, (0, 0, 0))],
               "thl": [(0, (0, 0, 0)), (0.2, (100, 0, 0)), (0.6, (100, 0, 0)), (0.8, (0, 0, 0))],
               "cal": [(0, (0, 0, 0)), (0.2, (-110, 0, 0)), (0.8, (0, 0, 0))]}),
    dict(name="tackle", dur=0.7, cat="special", dmg=14, kb=380, stop=9, hit=(30, 70), limb=BODY, height=ANY, travel=60, reach=1200, knockdown=1, nolook=1,
         keys={"body": [(0, (0, 0, 0)), (0.3, (0, -50, 0)), (0.7, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.3, (50, 0, 0)), (0.7, (0, 0, 0))],
               "ual": [(0, (0, 0, 0)), (0.3, (-40, 0, 0)), (0.7, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.3, (0, -500, 0)), (0.7, (0, 0, 0))]}),
    dict(name="headbutt", dur=0.5, cat="special", dmg=12, kb=180, stop=8, hit=(35, 65), limb=BODY, height=HIGH, travel=25, reach=1100,
         keys={"spine": [(0, (0, 0, 0)), (0.2, (-30, 0, 0)), (0.4, (50, 0, 0)), (0.5, (0, 0, 0))],
               "head": [(0, (0, 0, 0)), (0.2, (-30, 0, 0)), (0.4, (40, 0, 0)), (0.5, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.4, (0, 0, 200)), (0.5, (0, 0, 0))]}),
    dict(name="lightning", dur=1.1, cat="special", dmg=4, kb=30, stop=1, hit=(15, 92), limb=FOOT, height=ANY, travel=8, reach=1800, rehit=12,
         keys={"thr": [(0, (0, 0, 0)), (0.12, (90, 0, 0)), (0.24, (20, 0, 0)), (0.36, (110, 0, 0)), (0.48, (20, 0, 0)), (0.6, (100, 0, 0)), (0.72, (20, 0, 0)), (0.84, (120, 0, 0)), (0.96, (20, 0, 0)), (1.1, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.06, (-100, 0, 0)), (0.12, (0, 0, 0)), (0.3, (-100, 0, 0)), (0.36, (0, 0, 0)), (0.54, (-100, 0, 0)), (0.6, (0, 0, 0)), (0.78, (-100, 0, 0)), (0.84, (0, 0, 0)), (1.1, (0, 0, 0))],
               "ual": [(0, (0, 0, 0)), (0.12, (40, 0, 60)), (1.1, (0, 0, 0))],
               "uar": [(0, (0, 0, 0)), (0.12, (40, 0, -60)), (1.1, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.12, (-10, 0, 0)), (1.1, (0, 0, 0))]}),
    dict(name="stomp", dur=0.9, cat="special", dmg=15, kb=100, stop=9, hit=(45, 75), limb=FOOT, height=ANY, travel=30, reach=1400, knockdown=1, nolook=1,
         keys={"hip": [(0, (0, 0, 0)), (0.35, (0, 1900, 200)), (0.6, (0, 400, 400)), (0.75, (0, -300, 400)), (0.9, (0, 0, 0))],
               "thr": [(0, (0, 0, 0)), (0.35, (110, 0, 0)), (0.6, (60, 0, 0)), (0.9, (0, 0, 0))],
               "thl": [(0, (0, 0, 0)), (0.35, (110, 0, 0)), (0.6, (60, 0, 0)), (0.9, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.35, (-120, 0, 0)), (0.6, (-20, 0, 0)), (0.9, (0, 0, 0))],
               "cal": [(0, (0, 0, 0)), (0.35, (-120, 0, 0)), (0.6, (-20, 0, 0)), (0.9, (0, 0, 0))],
               "ual": [(0, (0, 0, 0)), (0.35, (170, 0, 0)), (0.9, (0, 0, 0))],
               "uar": [(0, (0, 0, 0)), (0.35, (170, 0, 0)), (0.9, (0, 0, 0))]}),
    dict(name="twin_fist", dur=0.9, cat="special", dmg=9, kb=120, stop=6, hit=(25, 85), limb=HAND, height=HIGH, travel=15, reach=1600, rehit=30,
         keys={"body": [(0, (0, 0, 0)), (0.45, (0, 360, 0)), (0.9, (0, 720, 0))],
               "ual": [(0, (0, 0, 0)), (0.2, (90, 0, 60)), (0.9, (90, 0, 60))],
               "uar": [(0, (0, 0, 0)), (0.2, (90, 0, -60)), (0.9, (90, 0, -60))],
               "spine": [(0, (0, 0, 0)), (0.2, (-10, 0, 0)), (0.9, (0, 0, 0))]}),
    dict(name="dragon_kick", dur=1.0, cat="special", dmg=20, kb=360, stop=12, hit=(35, 65), limb=FOOT, height=HIGH, travel=25, reach=2000, launch=1, nolook=1,
         keys={"hip": [(0, (0, 0, 0)), (0.2, (0, -300, 0)), (0.5, (0, 1700, 300)), (0.8, (0, 500, 400)), (1.0, (0, 0, 0))],
               "thr": [(0, (0, 0, 0)), (0.2, (20, 0, 0)), (0.5, (140, 0, 0)), (0.8, (60, 0, 0)), (1.0, (0, 0, 0))],
               "car": [(0, (0, 0, 0)), (0.2, (-100, 0, 0)), (0.5, (0, 0, 0)), (1.0, (0, 0, 0))],
               "thl": [(0, (0, 0, 0)), (0.5, (40, 0, 0)), (1.0, (0, 0, 0))],
               "cal": [(0, (0, 0, 0)), (0.5, (-90, 0, 0)), (1.0, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.5, (-25, 0, 0)), (1.0, (0, 0, 0))],
               "ual": [(0, (0, 0, 0)), (0.5, (120, 0, 30)), (1.0, (0, 0, 0))],
               "uar": [(0, (0, 0, 0)), (0.5, (120, 0, -30)), (1.0, (0, 0, 0))]}),
    # ---- hadouken (projectile) - conservative balance: slow, short range, heavy recovery
    # hit window is just for the animation; actual projectile spawns at ~50% and travels separately
    dict(name="hadouken", dur=0.9, cat="special", dmg=5, kb=80, stop=4, hit=(45, 55), limb=HAND, height=ANY, travel=0, reach=800, nolook=1,
         keys={"ual": [(0, (0, 0, 0)), (0.25, (40, 0, 60)), (0.5, (90, 0, 30)), (0.7, (90, 0, 30)), (0.9, (0, 0, 0))],
               "uar": [(0, (0, 0, 0)), (0.25, (40, 0, -60)), (0.5, (90, 0, -30)), (0.7, (90, 0, -30)), (0.9, (0, 0, 0))],
               "fal": [(0, (0, 0, 0)), (0.25, (100, 0, 0)), (0.5, (20, 0, 0)), (0.7, (20, 0, 0)), (0.9, (0, 0, 0))],
               "far": [(0, (0, 0, 0)), (0.25, (100, 0, 0)), (0.5, (20, 0, 0)), (0.7, (20, 0, 0)), (0.9, (0, 0, 0))],
               "spine": [(0, (0, 0, 0)), (0.25, (0, -20, 0)), (0.5, (0, 10, 0)), (0.9, (0, 0, 0))],
               "hip": [(0, (0, 0, 0)), (0.25, (0, 0, -100)), (0.5, (0, 0, 50)), (0.9, (0, 0, 0))]}),
]

# non-attack clips generated with the same machinery (guard pose)
EXTRAS = [
    dict(name="guard", dur=0.3,
         keys={"ual": [(0, (30, 0, 25)), (0.3, (30, 0, 25))], "uar": [(0, (30, 0, -25)), (0.3, (30, 0, -25))],
               "fal": [(0, (65, 0, 0)), (0.3, (65, 0, 0))], "far": [(0, (65, 0, 0)), (0.3, (65, 0, 0))],
               "spine": [(0, (8, 0, 0)), (0.3, (8, 0, 0))]}),
]
assert len(MOVES) == 33, len(MOVES)


# --------------------------------------------------------------------------
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def spec_rot(pitch, yaw, roll, left_sign):
    """(pitch, yaw, roll) in degrees -> 3x3 in PS1 world axes (y down, -z forward).
    pitch > 0 lifts a hanging limb forward / up, roll > 0 swings it to the
    character's left, yaw > 0 turns to the right."""
    return _ry(math.radians(yaw)) @ _rx(math.radians(-pitch)) @ _rz(math.radians(-roll * left_sign))


def interp(keys, t):
    if t <= keys[0][0]:
        return np.array(keys[0][1], dtype=np.float64)
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)                     # smoothstep
            return np.array(v0, dtype=np.float64) * (1 - u) + np.array(v1, dtype=np.float64) * u
    return np.array(keys[-1][1], dtype=np.float64)


def role_bones(bone_names, ik):
    """map roles -> bone indices from the IK table and bone names"""
    low = [n.lower() for n in bone_names]
    def find(*keys):
        for i, n in enumerate(low):
            if any(k in n for k in keys):
                return i
        return -1
    c = ik["chains"]
    r = {
        "hip": ik["hip"], "head": ik["head"],
        "ual": c[0]["upper"] if c[0] else -1, "fal": c[0]["lower"] if c[0] else -1,
        "uar": c[1]["upper"] if c[1] else -1, "far": c[1]["lower"] if c[1] else -1,
        "thl": c[2]["upper"] if c[2] else -1, "cal": c[2]["lower"] if c[2] else -1,
        "thr": c[3]["upper"] if c[3] else -1, "car": c[3]["lower"] if c[3] else -1,
    }
    spine = find("spine02", "spine2", "spine")
    r["spine"] = spine if spine >= 0 else ik["hip"]
    r["body"] = ik["hip"]
    return r


def synth_move(move, nb, parents, bind_local, bone_names, ik, unit_scale, sample_fps, mat_to_quat, verbose=True):
    """keyframed move -> {"name", "nframes", "q", "t"} in PS1 space"""
    roles = role_bones(bone_names, ik)
    nframes = max(2, int(round(move["dur"] * sample_fps)) + 1)
    bind_world = [None] * nb
    for b in range(nb):
        m = bind_local[b].copy()
        m[:3, 3] *= unit_scale
        p = parents[b]
        bind_world[b] = m if p < 0 else bind_world[p] @ m
    tl, tr = roles["thl"], roles["thr"]
    left_sign = 1.0 if (tl >= 0 and tr >= 0 and bind_world[tl][0, 3] >= bind_world[tr][0, 3]) else -1.0
    quats = np.zeros((nframes, nb, 4))
    trans = np.zeros((nframes, nb, 3))
    keys = move["keys"]
    for f in range(nframes):
        t = move["dur"] * f / (nframes - 1)
        spec = {}                                   # bone -> extra rotation (bind-world axes)
        # fighting stance on top of the T-pose bind: arms down and bent
        angles = {"ual": np.array([25.0, 0.0, -70.0]), "uar": np.array([25.0, 0.0, 70.0]),
                  "fal": np.array([70.0, 0.0, 0.0]), "far": np.array([70.0, 0.0, 0.0])}
        for role, kf in keys.items():
            if role == "hip":
                continue
            angles[role] = angles.get(role, np.zeros(3)) + interp(kf, t)
        for role, ang in angles.items():
            b = roles.get(role, -1)
            if b < 0:
                continue
            p_, y_, r_ = ang
            spec[b] = spec_rot(p_, y_, r_, left_sign) @ spec.get(b, np.eye(3))
        hip_off = np.zeros(3)
        if "hip" in keys:
            l_, u_, fw_ = interp(keys["hip"], t)
            hip_off = np.array([l_ * left_sign, -u_, -fw_])    # left, up, forward -> PS1 (y down, -z forward)
        world = [None] * nb
        for b in range(nb):
            p = parents[b]
            pw = np.eye(4) if p < 0 else world[p]
            pb = np.eye(4) if p < 0 else bind_world[p]
            # bone relative to the parent = bind relation, plus the spec
            # rotation applied in bind-world axes about the joint
            rel = np.linalg.inv(pb) @ bind_world[b]
            if b in spec:
                rel_rot = np.linalg.inv(pb[:3, :3]) @ spec[b] @ bind_world[b][:3, :3]
                rel = rel.copy()
                rel[:3, :3] = rel_rot
            if b == roles["hip"]:
                rel = rel.copy()
                rel[:3, 3] = rel[:3, 3] + np.linalg.inv(pw[:3, :3]) @ hip_off
            world[b] = pw @ rel
            quats[f, b] = mat_to_quat(rel)
            trans[f, b] = rel[:3, 3]
    for f in range(1, nframes):
        for b in range(nb):
            if np.dot(quats[f, b], quats[f - 1, b]) < 0:
                quats[f, b] = -quats[f, b]
    if verbose:
        print("synth move %-12s %d frames" % (move["name"], nframes))
    return {"name": move["name"], "nframes": nframes, "q": quats, "t": trans}


def write_table(path):
    """C table for the runtime (src/moves_table.h)"""
    cats = {"punch": 0, "kick": 1, "special": 2}
    lines = ["/* generated by tools/moves.py - do not edit */",
             "#define NUM_MOVES %d" % len(MOVES),
             "static const MoveDef MOVES[NUM_MOVES] = {"]
    for m in MOVES:
        clip = m.get("clip", m["name"])
        lines.append('\t{ "%s", "%s", %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d },' % (
            m["name"], clip, cats[m["cat"]], int(m.get("speed", 1.0) * 256), m["hit"][0], m["hit"][1],
            m["dmg"], m["kb"], m["stop"], m["limb"], m["height"], m["travel"], m["reach"],
            m.get("rehit", 0), (1 if m.get("launch") else 0) | (2 if m.get("knockdown") else 0) | (4 if m.get("nolook") else 0),
            0))
    lines.append("};")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s: %d moves" % (path, len(MOVES)))


if __name__ == "__main__":
    import sys
    write_table(sys.argv[1] if len(sys.argv) > 1 else "moves_table.h")
