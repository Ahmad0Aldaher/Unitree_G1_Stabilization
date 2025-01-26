import pinocchio as pin
import numpy as np
import math
import sys
import time

from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from pathlib import Path




#-------- Task 1.1 ------------
# Part 1 : Find contact forces and joint torques for the robot standing on one foot. Depending on chosen configuration the robot.
# Part 2 : check whether given configuration is stable or not


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

# input("Press Enter to close MeshCat and terminate... ")


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




v0 = np.zeros(model.nv)
a0 = np.zeros(model.nv)


#-------- Part 1  ------------

# We want to find the contact forces and torques required to stand still at a
# configuration 'q0'.
# We assume 6D contacts at the standing foot
#
# The dynamic equation would look like:
#
# M*q_ddot + g(q) + C(q, q_dot) = tau + J^T*lambda --> (for the static case) --> g(q) =
# tau + Jc^T*lambda (1).

# ----- SOLVING STRATEGY ------

# Split the equation between the base link (_bl) joint and the rest of the joints (_j).
# That is,
#
#  | g_bl |   |  0  |   | Jc__foot_bl.T |   
#  | g_j  | = | tau | + | Jc__foot_j.T  | *  | l |     (2)
#                                           
#                                           |

# First, find the contact wrench (these are 6 dimensional) by solving for
# the first 6 rows of (2).
# That is,
#
# g_bl   = Jc__foot_bl.T * | l |
#                              (3)
#                          
# Thus we find the contact froces by computing the jacobian pseudoinverse,
#
# | l | = pinv(Jc__foott_bl.T) * g_bl  (4)
#
# Now, we can find the necessary torques using the bottom rows in (2). That is,
#
#                             
#  tau = g_j - Jc__foot_j.T * | l |    (5)
#                            
#                             

# 1. GRAVITY TERM

# We compute the gravity terms by using the ID at desired configuration q0, with
# velocity and acceleration being 0. I.e., ID with a = v = 0.
g_grav = pin.rnea(model, data, q0, v0, a0)

g_bl = g_grav[:6]
g_j = g_grav[6:]


# get the id of the contacting link 
foot_id = model.getFrameId("right_ankle_roll_link")

# get the id of the body link (pelvis)
bl_id=model.getFrameId("pelvis")

# number of contact points
ncontact = 1

# Now, we need to find the contact Jacobians appearing in (1).
# These are the Jacobians that relate the joint velocity  to the velocity of the standing foot

Js__foot_q = np.copy(pin.computeFrameJacobian(model, data, q0, foot_id, pin.LOCAL))

# get the jacobian between contact foot and body link
Js__foot_bl = np.copy(Js__foot_q[:6, :6]) 

Jc__foot_bl_T = np.zeros([6, 6 * ncontact])

# transpot
Jc__foot_bl_T[:, :] = np.vstack(Js__foot_bl).T


# Now I only need to do the pinv to compute the contact forces
ls = np.linalg.pinv(Jc__foot_bl_T) @ g_bl # This is (3)


# Contact forces at local coordinates 
print("ls: ",ls)
 
pin.framesForwardKinematics(model, data, q0)


# Contact forces at base link frame
l_sp = pin.Force(ls)
l_sp__bl = data.oMf[bl_id].actInv(data.oMf[foot_id].act(l_sp))


print("\n--- CONTACT FORCES ---")

print(f"Contact force at foot  expressed at the BL is: {ls}")

# Notice that if we add all the contact forces are equal to the g_grav
print(
    "Error between contact forces and gravity at base link: "
    f"{np.linalg.norm(g_bl - l_sp__bl)}"
)


# 3. FIND TAU

Js_foot_j = np.copy(Js__foot_q[:6, 6:])
Jc__foot_j_T = np.zeros([37, 6 * ncontact])
Jc__foot_j_T[:, :] = np.vstack(Js_foot_j).T

tau = g_j - Jc__foot_j_T @ ls


# 4. CROSS CHECKS

# INVERSE DYNAMICS
# We can compare this torques with the ones one would obtain when computing the ID
# considering the external forces in ls.
pin.framesForwardKinematics(model, data, q0)


joint_ids = model.getJointId("right_ankle_roll_joint") 

fs_ext = [pin.Force(np.zeros(6)) for _ in range(len(model.joints))]
for idx, joint in enumerate(model.joints):
    if joint.id == joint_ids:
        fext__bl = pin.Force(l_sp__bl)
        fs_ext[idx] = data.oMi[joint.id].actInv(data.oMf[bl_id].act(fext__bl))

tau_rnea = pin.rnea(model, data, q0, v0, a0, fs_ext)

print("\n--- ID: JOINT TORQUES ---")
print(f"Tau from RNEA:         {tau_rnea}")
print(f"Tau computed manually: {np.append(np.zeros(6), tau)}")
print(f"Tau error: {np.linalg.norm(np.append(np.zeros(6), tau) - tau_rnea)}")


# FORWARD DYNAMICS
# We can also check the results using FD. FD with the tau we got, q0 and v0, should give
# 0 acceleration and the contact forces.
Js_feet6d_q = np.copy(Js__foot_q[:6, :])
acc = pin.forwardDynamics(
    model,
    data,
    q0,
    v0,
    np.append(np.zeros(6), tau),
    np.vstack(Js_feet6d_q),
    np.zeros(6),
)

print("\n--- FD: ACC. & CONTACT FORCES ---")
print(f"Norm of the FD acceleration: {np.linalg.norm(acc)}")
print(f"Contact forces manually: {ls}")
print(f"Contact forces FD: {data.lambda_c}")
print(f"Contact forces error: {np.linalg.norm(data.lambda_c - ls)}")



#
robot.display(q0)
# input("Press Enter to close MeshCat and terminate... ")


#########################################################


# #-------- Part 2  ------------
# # Check whether given configuration is stable or not
# # We will do that by finding center of mass CoM of the robot in the given configuration q0 
# # and then check if it's projection into supprot polygon genrated by the standing foot
# # (Note In case of static configuration  the Zero Moment Point (ZMP) coincides with the projection of the CoM onto the ground plane )


## find Com
# ----- Compute the CoM -----
com_position = pin.centerOfMass(model, data, q0)
com_projection = com_position[:2]  # Project CoM to the ground (x, y)

print("\n Center of Mass (CoM) position:", com_position)
print("Projected CoM on the ground (x, y):", com_projection)


# -----finding supporting polygon-----
# We consider four point inside the foot surface that form a rectangular.

# The coordinates of the rectangular's corneres at the bottom of the standing foot
#  (these coordinates are relative to the frame of the foot in our case "right_ankle_roll_link".  Note: this done using STL file of this link)
  
foot_contact_points_local = np.array([
    [0.124, 0.0275, -0.0354],   
    [0.124, -0.0275, -0.0354],  
    [-0.036, 0.0275, -0.0354],  
    [-0.036, -0.0275, -0.0354]  
])

# Compute foot pose in world frame
pin.framesForwardKinematics(model, data, q0)
foot_pose_world = data.oMf[foot_id]

# Transform contact points to the world frame
foot_contact_points_world = np.array([
    foot_pose_world.act(p) for p in foot_contact_points_local
])

# Project contact points onto the ground (x, y)
foot_contact_points_ground = foot_contact_points_world[:, :2]

# Compute the convex hull for the support polygon
hull = ConvexHull(foot_contact_points_ground)
support_polygon_points = foot_contact_points_ground[hull.vertices]

# ----- Check if CoM is Inside the Support Polygon -----
# Create a Shapely polygon from the support polygon points
support_polygon = Polygon(support_polygon_points)
com_point = Point(com_projection)

# Check if the CoM projection is inside the support polygon
is_com_inside = support_polygon.contains(com_point)

print("\nSupport Polygon Vertices (x, y):")
for vertex in support_polygon_points:
    print(vertex)

print(f"Is CoM inside the support polygon? {'Yes' if is_com_inside else 'No'}")

# input("\n Press Enter to close MeshCat and terminate... ")




