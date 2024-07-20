from .urdf_parser_py.urdf import URDF, Mesh, Cylinder, Box, Sphere
from . import frame
from . import chain
import torch
from . import transforms as tf
# has better RPY to quaternion transformation
import transformations as tf2

JOINT_TYPE_MAP = {
    "revolute": "revolute",
    "continuous": "revolute",
    "prismatic": "prismatic",
    "fixed": "fixed"
}


def _convert_transform(origin):
  if origin is None:
    return tf.Transform3d()
  else:
    return tf.Transform3d(
        rot=torch.tensor(
            tf2.quaternion_from_euler(*origin.rpy, "sxyz"),
            dtype=torch.float32),
        pos=origin.xyz)


def _convert_visual(visual):
  if visual is None or visual.geometry is None:
    return frame.Visual()
  else:
    v_tf = _convert_transform(visual.origin)
    if isinstance(visual.geometry, Mesh):
      g_type = "mesh"
      g_param = visual.geometry.filename
    elif isinstance(visual.geometry, Cylinder):
      g_type = "cylinder"
      g_param = (visual.geometry.radius, visual.geometry.length)
    elif isinstance(visual.geometry, Box):
      g_type = "box"
      g_param = visual.geometry.size
    elif isinstance(visual.geometry, Sphere):
      g_type = "sphere"
      g_param = visual.geometry.radius
    else:
      g_type = None
      g_param = None
    return frame.Visual(v_tf, g_type, g_param)


def _build_chain_recurse(root_frame, lmap, joints):
  children = []
  for j in joints:
    if j.parent == root_frame.link.name:
      child_frame = frame.Frame(j.child + "_frame")
      child_frame.joint = frame.Joint(
          j.name,
          offset=_convert_transform(j.origin),
          joint_type=JOINT_TYPE_MAP[j.type],
          axis=j.axis)
      link = lmap[j.child]
      child_frame.link = frame.Link(
          link.name,
          offset=_convert_transform(link.origin),
          visuals=[_convert_visual(link.visual)])
      child_frame.children = _build_chain_recurse(child_frame, lmap, joints)
      children.append(child_frame)
  return children


def build_chain_from_urdf(data):
  """
    Build a Chain object from URDF data.

    Parameters
    ----------
    data : str
        URDF string data.

    Returns
    -------
    chain.Chain
        Chain object created from URDF.

    Example
    -------
    >>> import pytorch_kinematics as pk
    >>> data = '''<robot name="test_robot">
    ... <link name="link1" />
    ... <link name="link2" />
    ... <joint name="joint1" type="revolute">
    ...   <parent link="link1"/>
    ...   <child link="link2"/>
    ... </joint>
    ... </robot>'''
    >>> chain = pk.build_chain_from_urdf(data)
    >>> print(chain)
    link1_frame
        link2_frame

    """
  robot = URDF.from_xml_string(data)
  lmap = robot.link_map
  joints = robot.joints
  n_joints = len(joints)
  has_root = [True for _ in range(len(joints))]
  for i in range(n_joints):
    for j in range(i + 1, n_joints):
      if joints[i].parent == joints[j].child:
        has_root[i] = False
      elif joints[j].parent == joints[i].child:
        has_root[j] = False
  for i in range(n_joints):
    if has_root[i]:
      root_link = lmap[joints[i].parent]
      break
  root_frame = frame.Frame(root_link.name + "_frame")
  root_frame.joint = frame.Joint()
  root_frame.link = frame.Link(root_link.name,
                               _convert_transform(root_link.origin),
                               [_convert_visual(root_link.visual)])
  root_frame.children = _build_chain_recurse(root_frame, lmap, joints)
  return chain.Chain(root_frame)


def build_serial_chain_from_urdf(data, end_link_name, root_link_name=""):
  """
    Build a SerialChain object from urdf data.

    Parameters
    ----------
    data : str
        URDF string data.
    end_link_name : str
        The name of the link that is the end effector.
    root_link_name : str, optional
        The name of the root link.

    Returns
    -------
    chain.SerialChain
        SerialChain object created from URDF.
    """
  urdf_chain = build_chain_from_urdf(data)
  return chain.SerialChain(
      urdf_chain, end_link_name + "_frame",
      "" if root_link_name == "" else root_link_name + "_frame")
