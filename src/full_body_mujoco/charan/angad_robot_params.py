"""
Angad Humanoid Robot — Inertial & Kinematic Parameters
======================================================
Extracted from XP_robot_walking.xml via MuJoCo.
Used by angad_crouch_final.py and all downstream controllers.

This file contains:
  1. Body masses, local COM positions, and diagonal inertias
  2. Joint axes (local & global at default pose), ranges
  3. Actuator gear ratios and torque limits
  4. Verified crouch IK solutions
  5. Axis mirroring rules for left↔right symmetry
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  1.  BODY MASS & INERTIA
# ═══════════════════════════════════════════════════════════════════════

TOTAL_MASS_KG = 64.94

# fmt: off
BODY_PARAMS = {
    # name              mass(kg)   local_COM [x, y, z]                 diag_inertia [Ixx, Iyy, Izz]
    'pelvis':         { 'mass': 24.406, 'com': [ 0.0005, -0.0224,  0.0282], 'inertia': [0.1648, 0.1197, 0.1568] },
    'hip_pitch_r':    { 'mass':  1.008, 'com': [-0.0001,  0.0035,  0.0728], 'inertia': [0.0042, 0.0059, 0.0035] },
    'hip_pitch_l':    { 'mass':  1.008, 'com': [-0.0001,  0.0035,  0.0728], 'inertia': [0.0042, 0.0059, 0.0035] },
    'hip_roll_r':     { 'mass':  1.319, 'com': [ 0.0000, -0.0000,  0.0397], 'inertia': [0.0087, 0.0104, 0.0072] },
    'hip_roll_l':     { 'mass':  1.319, 'com': [ 0.0000, -0.0000,  0.0397], 'inertia': [0.0087, 0.0104, 0.0072] },
    'thigh_r':        { 'mass':  2.264, 'com': [ 0.0144, -0.1346, -0.0424], 'inertia': [0.0456, 0.0488, 0.0082] },
    'thigh_l':        { 'mass':  2.264, 'com': [ 0.0144, -0.1346, -0.0424], 'inertia': [0.0456, 0.0488, 0.0082] },
    'leg_shank_r':    { 'mass':  1.446, 'com': [-0.1487, -0.0041, -0.5012], 'inertia': [0.0230, 0.0243, 0.0026] },
    'leg_shank_l':    { 'mass':  1.446, 'com': [-0.1487, -0.0041, -0.5012], 'inertia': [0.0230, 0.0243, 0.0026] },
    'foot_r':         { 'mass':  0.253, 'com': [-0.0230,  0.0000,  0.0007], 'inertia': [0.0001, 0.0003, 0.0003] },
    'foot_l':         { 'mass':  0.253, 'com': [-0.0230,  0.0000,  0.0007], 'inertia': [0.0001, 0.0003, 0.0003] },
    'torso':          { 'mass': 22.100, 'com': [ 0.0002,  0.0066,  0.1940], 'inertia': [0.3051, 0.2534, 0.1207] },
}
# fmt: on


# ═══════════════════════════════════════════════════════════════════════
#  2.  JOINT AXES & LIMITS
# ═══════════════════════════════════════════════════════════════════════

# Global axes at the default standing pose (rounded).
# These determine the mirroring rules.
JOINT_AXES_GLOBAL = {
    'pelvis_hip_pitch_l':      np.array([ 0.97,  0.00, -0.26]),  # 15° oblique!
    'pelvis_hip_pitch_r':      np.array([ 0.97,  0.00, +0.26]),  # mirrored Z
    'hip_pitch_l_hip_roll_l':  np.array([ 0.00, -1.00,  0.00]),
    'hip_pitch_r_hip_roll_r':  np.array([ 0.00, -1.00,  0.00]),
    'hip_roll_l_thigh_yaw_l':  np.array([ 0.00,  0.00, -1.00]),
    'hip_roll_r_thigh_yaw_r':  np.array([ 0.00,  0.00, -1.00]),
    'thigh_l_knee_l':          np.array([-1.00,  0.00,  0.00]),
    'thigh_r_knee_r':          np.array([-1.00,  0.00,  0.00]),
    'leg_shank_l_ankle_pitch_l': np.array([ 1.00,  0.00,  0.00]),
    'leg_shank_r_ankle_pitch_r': np.array([ 1.00,  0.00,  0.00]),
    'ankle_pitch_l_ankle_roll_l': np.array([ 1.00,  0.00,  0.00]),
    'uj_r_ankle_roll_r':          np.array([ 1.00,  0.00,  0.00]),
}

# Joint limits in radians
JOINT_LIMITS_RAD = {
    'pelvis_hip_pitch':  (-3.1416, 3.1416),
    'hip_roll':          (-2.35,   0.292),   # R: (-134.6°, 16.7°)  L: (-16.7°, 134.6°) — swapped
    'thigh_yaw':         (-3.1416, 3.1416),
    'knee':              (-1.5708, 0.0),     # -90° to 0°
    'ankle_pitch':       (-0.6981, 0.6981),  # ±40°
    'ankle_roll':        (-0.3491, 0.3491),  # ±20°
}


# ═══════════════════════════════════════════════════════════════════════
#  3.  ACTUATOR GEAR RATIOS & TORQUE LIMITS
# ═══════════════════════════════════════════════════════════════════════

ACTUATOR_PARAMS = {
    # actuator_name:  (joint_name,                      gear_ratio, max_torque_Nm)
    'hip_pitch_r':    ('pelvis_hip_pitch_r',              80,  80),
    'hip_roll_r':     ('hip_pitch_r_hip_roll_r',          60,  60),
    'thigh_yaw_r':    ('hip_roll_r_thigh_yaw_r',          60,  60),
    'knee_r':         ('thigh_r_knee_r',                 120, 120),
    'ankle_pitch_r':  ('leg_shank_r_ankle_pitch_r',       80,  80),
    'ankle_roll_r':   ('uj_r_ankle_roll_r',               80,  80),
    'hip_pitch_l':    ('pelvis_hip_pitch_l',              80,  80),
    'hip_roll_l':     ('hip_pitch_l_hip_roll_l',          60,  60),
    'thigh_yaw_l':    ('hip_roll_l_thigh_yaw_l',          60,  60),
    'knee_l':         ('thigh_l_knee_l',                 120, 120),
    'ankle_pitch_l':  ('leg_shank_l_ankle_pitch_l',       80,  80),
    'ankle_roll_l':   ('uj_l_ankle_roll_l',               80,  80),
    'torso_yaw':      ('pelvis_torso_yaw',                90,  90),
    'arm_pitch_r':    ('torso_arm_pitch_r',               60,  60),
    'arm_roll_r':     ('shoulder_pitch_r_arm_roll_r',     60,  60),
    'arm_yaw_r':      ('shoulder_roll_r_arm_yaw_r',       17,  17),
    'elbow_r':        ('bicep_r_elbow_r',                 17,  17),
    'arm_pitch_l':    ('torso_arm_pitch_l',               60,  60),
    'arm_roll_l':     ('shoulder_pitch_l_arm_roll_l',     60,  60),
    'arm_yaw_l':      ('shoulder_roll_l_arm_yaw_l',       17,  17),
    'elbow_l':        ('bicep_l_elbow_l',                 17,  17),
}

GEAR_RATIOS = {name: params[1] for name, params in ACTUATOR_PARAMS.items()}


# ═══════════════════════════════════════════════════════════════════════
#  4.  LEFT↔RIGHT MIRRORING RULES
# ═══════════════════════════════════════════════════════════════════════
#
#  The hip pitch axes are 15° oblique with opposite Z signs.
#  Empirically verified mirroring (used in angad_crouch_final.py):
#
#    hip_pitch_r  =  hip_pitch_l     (SAME sign — NOT mirrored)
#    hip_roll_r   = -hip_roll_l      (mirrored)
#    thigh_yaw_r  = -thigh_yaw_l     (mirrored)
#    knee_r       =  knee_l          (SAME sign)
#    ankle_pitch_r=  ankle_pitch_l   (SAME sign)
#    ankle_roll_r = -ankle_roll_l    (mirrored)
#

MIRROR_SIGN = {
    'hip_pitch':   +1,  # NOT mirrored
    'hip_roll':    -1,  # mirrored
    'thigh_yaw':   -1,  # mirrored
    'knee':        +1,  # NOT mirrored
    'ankle_pitch': +1,  # NOT mirrored
    'ankle_roll':  -1,  # mirrored
}


# ═══════════════════════════════════════════════════════════════════════
#  5.  VERIFIED CROUCH IK SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════
#
#  Computed by scipy.optimize.minimize (SLSQP) against MuJoCo FK.
#  Stability verified by headless simulation (5s, no fall).
#
#  Order: [hip_pitch, hip_roll, thigh_yaw, knee, ankle_pitch, ankle_roll]
#

CROUCH_IK_SOLUTIONS = {
    # drop_cm: (left_leg_angles, stable?, knee_ctrl_pct)
     7: ([-0.4177, +0.0260, -0.0108, -0.8731, -0.4968, 0.0], True,  1.00),
    10: ([-0.5010, +0.0357, -0.0067, -1.0482, -0.5919, 0.0], True,  0.77),
    11: ([-0.5282, +0.0391, -0.0050, -1.1021, -0.5975, 0.0], True,  0.63),
    12: ([-0.5543, +0.0425, -0.0032, -1.1537, -0.6024, 0.0], True,  0.67),  # ← RECOMMENDED
    15: ([-0.6269, +0.0523, +0.0028, -1.2967, -0.6129, 0.0], False, 1.00),  # ← exceeds torque
}

# The recommended operating point for walking
RECOMMENDED_CROUCH_CM = 12
RECOMMENDED_CROUCH_ANGLES = CROUCH_IK_SOLUTIONS[12][0]


# ═══════════════════════════════════════════════════════════════════════
#  6.  STANDING POSE & BALANCE GAINS
# ═══════════════════════════════════════════════════════════════════════

STANDING_HEIGHT_M = 0.746  # default qpos[2] for upright standing

# PD gains proven stable across all tested controllers
PD_GAINS = {
    'knee':        {'kp': 2000, 'kd': 80},
    'hip':         {'kp': 1500, 'kd': 60},
    'thigh_yaw':   {'kp': 1500, 'kd': 60},
    'ankle':       {'kp':  800, 'kd': 30},
    'torso':       {'kp': 1000, 'kd': 50},
    'arm':         {'kp':  200, 'kd': 10},
}

# IMU balance feedback gains
BALANCE_GAINS = {
    'pitch': {'kp': 2000, 'kd': 300},
    'roll':  {'kp': 1500, 'kd': 200},
}

# Balance torque distribution
BALANCE_DISTRIBUTION = {
    'ankle_pitch_share': 1.0,   # 100% of pitch correction → ankle
    'ankle_roll_share':  0.4,   # 40% of roll correction → ankle
    'hip_roll_share':    0.6,   # 60% of roll correction → hip
}


# ═══════════════════════════════════════════════════════════════════════
#  7.  KINEMATIC CHAIN LINK LENGTHS (approximate, from body positions)
# ═══════════════════════════════════════════════════════════════════════

LINK_LENGTHS_M = {
    'pelvis_to_hip_pitch':  0.115,   # vertical from pelvis origin to hip_pitch
    'hip_pitch_to_hip_roll': 0.143,  # along the oblique axis
    'hip_roll_to_thigh':    0.169,   # to the thigh yaw joint
    'thigh_to_knee':        0.266,   # upper leg length
    'knee_to_ankle':        0.350,   # shank length (approximate)
    'ankle_to_foot':        0.031,   # foot height
    'foot_half_width':      0.040,   # lateral foot half-width
    'foot_half_length':     0.080,   # forward foot half-length
    'track_width':          0.299,   # distance between feet (X direction)
}


# ═══════════════════════════════════════════════════════════════════════
#  Utility: build gain/target arrays from MuJoCo model
# ═══════════════════════════════════════════════════════════════════════

def build_gain_arrays(m):
    """Return (kp, kd, gear) arrays sized to m.nu, using PD_GAINS."""
    nu = m.nu
    kp = np.zeros(nu);  kd = np.zeros(nu)
    gear = np.array([m.actuator_gear[i][0] for i in range(nu)])
    import mujoco
    for i in range(nu):
        name = m.actuator(i).name
        if 'knee' in name:
            kp[i] = PD_GAINS['knee']['kp'];  kd[i] = PD_GAINS['knee']['kd']
        elif 'hip' in name or 'thigh' in name:
            kp[i] = PD_GAINS['hip']['kp'];   kd[i] = PD_GAINS['hip']['kd']
        elif 'ankle' in name:
            kp[i] = PD_GAINS['ankle']['kp']; kd[i] = PD_GAINS['ankle']['kd']
        elif 'torso' in name:
            kp[i] = PD_GAINS['torso']['kp']; kd[i] = PD_GAINS['torso']['kd']
        else:
            kp[i] = PD_GAINS['arm']['kp'];   kd[i] = PD_GAINS['arm']['kd']
    return kp, kd, gear


def build_crouch_targets(m, drop_cm=12):
    """Return target array for a crouch at given depth."""
    import mujoco
    nu = m.nu
    act_names = [m.actuator(i).name for i in range(nu)]
    IDX = {n: i for i, n in enumerate(act_names)}
    target = np.zeros(nu)

    angles = CROUCH_IK_SOLUTIONS[drop_cm][0]
    # Left leg
    target[IDX['hip_pitch_l']]   = angles[0]
    target[IDX['hip_roll_l']]    = angles[1]
    target[IDX['thigh_yaw_l']]   = angles[2]
    target[IDX['knee_l']]        = angles[3]
    target[IDX['ankle_pitch_l']] = angles[4]
    target[IDX['ankle_roll_l']]  = angles[5]
    # Right leg (mirrored)
    target[IDX['hip_pitch_r']]   = angles[0] * MIRROR_SIGN['hip_pitch']
    target[IDX['hip_roll_r']]    = angles[1] * MIRROR_SIGN['hip_roll']
    target[IDX['thigh_yaw_r']]   = angles[2] * MIRROR_SIGN['thigh_yaw']
    target[IDX['knee_r']]        = angles[3] * MIRROR_SIGN['knee']
    target[IDX['ankle_pitch_r']] = angles[4] * MIRROR_SIGN['ankle_pitch']
    target[IDX['ankle_roll_r']]  = angles[5] * MIRROR_SIGN['ankle_roll']

    return target, IDX
