import argparse
import pinocchio as pin
import numpy as np
try:
    from pinocchio.visualize import MeshcatVisualizer
except ImportError as e:
    raise ImportError("Pinocchio not found, try ``pip install pin``") from e

from robot_descriptions.loaders.pinocchio import load_robot_description


import math
import sys
import time
from pathlib import Path



# loadd robot model into pinocchio
try:
    from pinocchio.visualize import MeshcatVisualizer
except ImportError as e:
    raise ImportError("Pinocchio not found, try ``pip install pin``") from e

from robot_descriptions.loaders.pinocchio import load_robot_description


robot = load_robot_description("g1_mj_description")


robot.setVisualizer(MeshcatVisualizer())
robot.initViewer(open=True)
robot.loadViewerModel("pinocchio")
#robot.display(robot.q0)



model = robot.model
data = robot.data



q0 = np.array(
    [
      0 ,0, -0.01,
      0, 0, 0, 1,
      0, 0, 0, 1, 0, 0,
      0, 0.2, 0, 0, 0.0, -0.2,
      0, 
      0, 0.3, 0, 1.28 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 
      0,-1 ,0 ,1.28 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0
    ]
)

 
  
frame_id = model.getFrameId("right_ankle_roll_link") 
            
q_ref = pin.integrate(model, q0, 0.03* np.random.rand(model.nv))

robot.display(q0)

v0 = np.zeros(model.nv)
v_ref = v0.copy()
    
data_sim = model.createData()
data_control = model.createData()


contact_models = []
contact_datas = [] 

frame = model.frames[frame_id]

contact_model = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D, model, frame.parentJoint, frame.placement
        )

contact_models.append(contact_model)
contact_datas.append(contact_model.createData())


num_constraints=1
contact_dim = 6 * num_constraints

pin.initConstraintDynamics(model, data_sim, contact_models)


t = 0
dt = 5e-3

S = np.zeros((model.nv - 6, model.nv))
S.T[6:, :] = np.eye(model.nv - 6)
Kp_posture = 100.0
Kv_posture = 0.05 * math.sqrt(Kp_posture)

q = q0.copy()
v = v0.copy()
tau = np.zeros(model.nv)

T = 5


while t <= T:
    print("t:", t)
    t += dt

    tic = time.time()
    J_constraint = np.zeros((contact_dim, model.nv))
    pin.computeJointJacobians(model, data_control, q)
    

    J_constraint[ :6, :] = pin.getFrameJacobian(
                model,
                data_control,
                contact_model.joint1_id,
                contact_model.joint1_placement,
                contact_model.reference_frame,
            )


    A = np.vstack((S, J_constraint))
    b = pin.rnea(model, data_control, q, v, np.zeros(model.nv))

    sol = np.linalg.lstsq(A.T, b, rcond=None)[0]
    tau = np.concatenate((np.zeros((6)), sol[: model.nv - 6]))

    tau[6:] += (
            -Kp_posture * (pin.difference(model, q_ref, q))[6:]
            - Kv_posture * (v - v_ref)[6:]
        )

    prox_settings = pin.ProximalSettings(1e-12, 1e-12, 10)
    a = pin.constraintDynamics(
            model, data_sim, q, v, tau, contact_models, contact_datas, prox_settings
        )
    print("a:", a.T)
    print("v:", v.T)
    print("constraint:", np.linalg.norm(J_constraint @ a))
    print("iter:", prox_settings.iter)

    v += a * dt
    q = pin.integrate(model, q, v * dt)

    robot.display(q)
    elapsed_time = time.time() - tic

    time.sleep(max(0, dt - elapsed_time))
    # input()


