import mujoco
import pinocchio as pin
from robot_descriptions import go2_mj_description

mj_model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)

pin_model, constraint_models = pin.buildModelAndLegacyConstraintsFromMJCF(
    go2_mj_description.MJCF_PATH,
    pin.JointModelFreeFlyer(),
    "root_joint",
)

print("nq, nv:", pin_model.nq, pin_model.nv)          # expect 19, 18
print("root joint type:", pin_model.joints[1])         # expect JointModelFreeFlyer
print("num legacy constraints:", len(constraint_models))

pin_mass = pin.computeTotalMass(pin_model)
mj_mass = mj_model.body_mass.sum()
print("total mass — pinocchio:", pin_mass, " mujoco:", mj_mass)

print("armature — pinocchio:", pin_model.armature)
print("armature — mujoco:", mj_model.dof_armature)

print("constraint models:", constraint_models)

print("pinocchio joint names:", [pin_model.names[i] for i in range(pin_model.njoints)])
print("mujoco joint names:", [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)])

print("\nnum actuators:", mj_model.nu)
print("actuator names:", [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(mj_model.nu)])

# actuator_trnid[:,0] gives the joint id each actuator drives (for joint-type transmissions)
print("actuator -> joint id:", mj_model.actuator_trnid[:, 0])
print("joint names by id:", [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in mj_model.actuator_trnid[:, 0]])
