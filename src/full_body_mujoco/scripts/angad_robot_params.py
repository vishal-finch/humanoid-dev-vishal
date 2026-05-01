"""
Angad Humanoid Robot — Verified Crouch Parameters
==================================================
IK solutions computed by scipy SLSQP, verified stable 5+ seconds in MuJoCo.
Joint names use angad_full_body.xml convention.

Mirroring rules (from global-axis analysis):
  hip_pitch:   SAME sign    (NOT mirrored)
  hip_roll:    mirrored
  thigh_yaw:   mirrored
  knee_pitch:  SAME sign
  ankle_pitch: SAME sign
  ankle_roll:  mirrored
"""

# ── Joint name order for the lower_body_controller ────────────────────
LOWER_BODY_JOINTS = [
    'hip_pitch_r', 'hip_roll_r', 'thigh_yaw_r',
    'knee_pitch_r', 'ankle_pitch_r', 'ankle_roll_r',
    'hip_pitch_l', 'hip_roll_l', 'thigh_yaw_l',
    'knee_pitch_l', 'ankle_pitch_l', 'ankle_roll_l',
]

# ── Mirroring sign: left → right ─────────────────────────────────────
MIRROR_SIGN = {
    'hip_pitch':   +1,
    'hip_roll':    -1,
    'thigh_yaw':   -1,
    'knee_pitch':  +1,
    'ankle_pitch': +1,
    'ankle_roll':  -1,
}

# ── Verified IK crouch solutions ─────────────────────────────────────
# Order: [hip_pitch, hip_roll, thigh_yaw, knee_pitch, ankle_pitch, ankle_roll]
# All angles are LEFT leg values; right leg = left * MIRROR_SIGN
CROUCH_IK = {
    #  depth_cm: (left_leg_angles,                                     stable?)
     7: ([-0.4177, +0.0260, -0.0108, -0.8731, -0.4968, 0.0],         True),
    10: ([-0.5010, +0.0357, -0.0067, -1.0482, -0.5919, 0.0],         True),
    11: ([-0.5282, +0.0391, -0.0050, -1.1021, -0.5975, 0.0],         True),
    12: ([-0.5543, +0.0425, -0.0032, -1.1537, -0.6024, 0.0],         True),   # RECOMMENDED
    15: ([-0.6269, +0.0523, +0.0028, -1.2967, -0.6129, 0.0],         False),  # exceeds torque
}

RECOMMENDED_DEPTH_CM = 12

# ── PD Gains (verified stable by charan) ──────────────────────────────
PD_GAINS = {
    'knee':    {'kp': 2000, 'kd': 80},
    'hip':     {'kp': 1500, 'kd': 60},
    'ankle':   {'kp':  800, 'kd': 30},
}

# ── IMU Balance Gains ─────────────────────────────────────────────────
BALANCE_GAINS = {
    'pitch': {'kp': 2000, 'kd': 300},
    'roll':  {'kp': 1500, 'kd': 200},
}

# Balance torque distribution
ANKLE_PITCH_SHARE = 1.0   # 100% pitch correction → ankle
ANKLE_ROLL_SHARE  = 0.4   # 40% roll → ankle
HIP_ROLL_SHARE    = 0.6   # 60% roll → hip


def get_crouch_targets(depth_cm=12):
    """
    Return full 12-element position array for the lower_body_controller.
    Order matches LOWER_BODY_JOINTS.
    """
    left_angles = CROUCH_IK[depth_cm][0]
    joint_keys = ['hip_pitch', 'hip_roll', 'thigh_yaw',
                  'knee_pitch', 'ankle_pitch', 'ankle_roll']

    # Right leg = left * mirror
    right = [a * MIRROR_SIGN[k] for a, k in zip(left_angles, joint_keys)]
    left = list(left_angles)

    return right + left   # [R_hip, R_roll, ..., L_hip, L_roll, ...]
