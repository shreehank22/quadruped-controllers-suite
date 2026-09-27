import numpy as np
import mujoco
import pinocchio as pin
from robot_descriptions import go2_mj_description

mj_model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)
mj_data = mujoco.MjData(mj_model)

pin_model, _ = pin.buildModelAndLegacyConstraintsFromMJCF(
    go2_mj_description.MJCF_PATH, pin.JointModelFreeFlyer(), "root_joint"
)
pin_data = pin_model.createData()

# --- 1. set a known, non-trivial base state in MuJoCo ---
# 90 deg rotation about world z-axis, MuJoCo quat order is (w, x, y, z)
angle = np.pi / 2
mj_data.qpos[3:7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
mj_data.qpos[0:3] = [0.1, 0.2, 0.3]          # arbitrary base position
mj_data.qpos[7:] = 0.05                       # small nonzero angle on every leg joint

# free joint qvel convention: [0:3] linear vel of body origin in WORLD frame,
#                              [3:6] angular vel in BODY (local) frame
v_world_lin = np.array([1.0, 0.5, 0.0])
w_body = np.array([0.0, 0.0, 0.5])
mj_data.qvel[0:3] = v_world_lin
mj_data.qvel[3:6] = w_body
mj_data.qvel[6:] = 0.1                        # small nonzero joint velocities

mujoco.mj_forward(mj_model, mj_data)

# --- 2. build the corresponding Pinocchio q, v by hand, per documented convention ---
w, x, y, z = mj_data.qpos[3:7]                # MuJoCo: wxyz
quat_xyzw = [x, y, z, w]                      # Pinocchio: xyzw

q = np.zeros(pin_model.nq)
q[0:3] = mj_data.qpos[0:3]
q[3:7] = quat_xyzw
q[7:] = mj_data.qpos[7:]

R = pin.Quaternion(w, x, y, z).toRotationMatrix()   # world_R_body
v_body_lin = R.T @ v_world_lin                       # Pinocchio free-flyer: linear vel in LOCAL frame

v = np.zeros(pin_model.nv)
v[0:3] = v_body_lin
v[3:6] = w_body                               # angular already local/body on both sides
v[6:] = mj_data.qvel[6:]

pin.forwardKinematics(pin_model, pin_data, q, v)
pin.updateFramePlacements(pin_model, pin_data)

# --- 3. compare a foot's position and velocity between the two ---
# --- 3. compare a link's position and velocity between the two ---
link_name = "FL_calf"

foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
print("MuJoCo link pos: ", mj_data.xpos[foot_id])

frame_id = pin_model.getFrameId(link_name)
print("Pinocchio link pos:", pin_data.oMf[frame_id].translation)
print("Pinocchio frame names available:", [f.name for f in pin_model.frames][:10], "...")

# --- 4. compare frame velocity ---
mj_vel = np.zeros(6)
mujoco.mj_objectVelocity(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, foot_id, mj_vel, 0)  # flg_local=0 -> world-aligned
mj_lin_world, mj_ang_world = mj_vel[3:6], mj_vel[0:3]   # MuJoCo returns [ang; lin]

pin_vel = pin.getFrameVelocity(pin_model, pin_data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

print("\nMuJoCo link lin vel (world):   ", mj_lin_world)
print("Pinocchio link lin vel (world):", pin_vel.linear)
print("MuJoCo link ang vel (world):   ", mj_ang_world)
print("Pinocchio link ang vel (world):", pin_vel.angular)

# --- 5. ground-truth check via finite difference ---
qpos0 = mj_data.qpos.copy()
qvel0 = mj_data.qvel.copy()

dt = 1e-7

qpos_pert = qpos0.copy()
mujoco.mj_integratePos(mj_model, qpos_pert, qvel0, dt)

mj_data.qpos[:] = qpos_pert
mujoco.mj_kinematics(mj_model, mj_data)
foot_pos_pert = mj_data.xpos[foot_id].copy()

mj_data.qpos[:] = qpos0
mujoco.mj_kinematics(mj_model, mj_data)
foot_pos0 = mj_data.xpos[foot_id].copy()

fd_vel = (foot_pos_pert - foot_pos0) / dt

print("\nFinite-difference foot linear vel (world):", fd_vel)
print("mj_objectVelocity foot linear vel (world): ", mj_lin_world)
print("Pinocchio foot linear vel (world):         ", pin_vel.linear)