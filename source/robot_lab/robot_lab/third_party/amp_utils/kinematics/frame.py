import torch
from . import transforms as tf


class Visual(object):
  TYPES = ['box', 'cylinder', 'sphere', 'capsule', 'mesh']

  def __init__(self, offset=tf.Transform3d(), geom_type=None, geom_param=None):
    self.offset = offset
    self.geom_type = geom_type
    self.geom_param = geom_param

  def __repr__(self):
    return "Visual(offset={0}, geom_type='{1}', geom_param={2})".format(
        self.offset, self.geom_type, self.geom_param)


class Link(object):

  def __init__(self, name=None, offset=tf.Transform3d(), visuals=()):
    self.name = name
    self.offset = offset
    self.visuals = visuals

  def to(self, *args, **kwargs):
    self.offset = self.offset.to(*args, **kwargs)
    return self

  def __repr__(self):
    return "Link(name='{0}', offset={1}, visuals={2})".format(
        self.name, self.offset, self.visuals)


class Joint(object):
  TYPES = ['fixed', 'revolute', 'prismatic']

  def __init__(self,
               name=None,
               offset=tf.Transform3d(),
               joint_type='fixed',
               axis=(0.0, 0.0, 1.0),
               dtype=torch.float32,
               device='cpu'):
    self.name = name
    self.offset = offset
    if joint_type not in self.TYPES:
      raise RuntimeError(
          'joint specified as {} type not, but we only support {}'.format(
              joint_type, self.TYPES))
    self.joint_type = joint_type
    if axis is None:
      self.axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    else:
      if torch.is_tensor(axis):
        self.axis = axis.clone().detach().to(dtype=dtype, device=device)
      else:
        self.axis = torch.tensor(axis, dtype=dtype, device=device)
    # normalize axis to have norm 1 (needed for correct representation scaling with theta)
    self.axis = self.axis / self.axis.norm()

  def to(self, *args, **kwargs):
    self.axis = self.axis.to(*args, **kwargs)
    self.offset = self.offset.to(*args, **kwargs)
    return self

  def __repr__(self):
    return "Joint(name='{0}', offset={1}, joint_type='{2}', axis={3})".format(
        self.name, self.offset, self.joint_type, self.axis)


class Frame(object):

  def __init__(self, name=None, link=Link(), joint=Joint(), children=()):
    self.name = 'None' if name is None else name
    self.link = link
    self.joint = joint
    self.children = children

  def __str__(self, level=0):
    ret = ' \t' * level + self.name + '\n'
    for child in self.children:
      ret += child.__str__(level + 1)
    return ret

  def to(self, *args, **kwargs):
    self.joint = self.joint.to(*args, **kwargs)
    self.link = self.link.to(*args, **kwargs)
    self.children = [c.to(*args, **kwargs) for c in self.children]
    return self

  def add_child(self, child):
    self.children.append(child)

  def is_end(self):
    return (len(self.children) == 0)

  def get_transform(self, theta):
    dtype = self.joint.axis.dtype
    d = self.joint.axis.device
    if self.joint.joint_type == 'revolute':
      t = tf.Transform3d(
          rot=tf.axis_angle_to_quaternion(theta * self.joint.axis),
          dtype=dtype,
          device=d)
    elif self.joint.joint_type == 'prismatic':
      t = tf.Transform3d(pos=theta * self.joint.axis, dtype=dtype, device=d)
    elif self.joint.joint_type == 'fixed':
      t = tf.Transform3d(
          default_batch_size=theta.shape[0], dtype=dtype, device=d)
    else:
      raise ValueError('Unsupported joint type %s.' % self.joint.joint_type)
    return self.joint.offset.compose(t)
