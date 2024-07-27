"""Special orthogonal group."""
import warnings
from typing import Tuple
import torch

from .math import acos_linear_extrapolation

HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5


def so3_relative_angle(R1, R2, cos_angle: bool = False):
  """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1
    R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than the
          angle itself. This can avoid the unstable calculation of `acos`.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
  R12 = torch.bmm(R1, R2.permute(0, 2, 1))
  return so3_rotation_angle(R12, cos_angle=cos_angle)


def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
  """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.
    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than the
          angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to [-1 + cos_bound, 1
          - cos_bound] to avoid non-finite outputs/gradients of the `acos` call.
          Note that the non-finite outputs/gradients are returned when the angle
          is requested (i.e. `cos_angle==False`) and the rotation angle is close
          to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.
    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

  N, dim1, dim2 = R.shape
  if dim1 != 3 or dim2 != 3:
    raise ValueError("Input has to be a batch of 3x3 Tensors.")

  rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

  if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
    raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

  # phi ... rotation angle
  phi_cos = (rot_trace - 1.0) * 0.5

  if cos_angle:
    return phi_cos
  else:
    if cos_bound > 0.0:
      return acos_linear_extrapolation(phi_cos, 1.0 - cos_bound)
    else:
      return torch.acos(phi_cos)


def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
  """
    Convert a batch of logarithmic representations of rotation matrices
    `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].
    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation
    respectively.
    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.
    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.
    Raises:
        ValueError if `log_rot` is of incorrect shape.
    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
  return _so3_exp_map(log_rot, eps=eps)[0]


def _so3_exp_map(
    log_rot: torch.Tensor,
    eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
  _, dim = log_rot.shape
  if dim != 3:
    raise ValueError("Input tensor shape has to be Nx3.")

  nrms = (log_rot * log_rot).sum(1)
  # phis ... rotation angles
  rot_angles = torch.clamp(nrms, eps).sqrt()
  rot_angles_inv = 1.0 / rot_angles
  fac1 = rot_angles_inv * rot_angles.sin()
  fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
  skews = hat(log_rot)
  skews_square = torch.bmm(skews, skews)

  R = (
      # pyre-fixme[16]: `float` has no attribute `__getitem__`.
      fac1[:, None, None] * skews + fac2[:, None, None] * skews_square +
      torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None])

  return R, rot_angles, skews, skews_square


def so3_log_map(R, eps: float = 0.0001):
  """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)` which is handled
    by clamping controlled with the `eps` argument.

    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

  N, dim1, dim2 = R.shape
  if dim1 != 3 or dim2 != 3:
    raise ValueError("Input has to be a batch of 3x3 Tensors.")

  phi = so3_rotation_angle(R)

  phi_sin = phi.sin()

  phi_denom = (
      torch.clamp(phi_sin.abs(), eps) * phi_sin.sign() +
      (phi_sin == 0).type_as(phi) * eps)

  log_rot_hat = (phi / (2.0 * phi_denom))[:, None, None] * (
      R - R.permute(0, 2, 1))
  log_rot = hat_inv(log_rot_hat)

  return log_rot


def hat_inv(h):
  """
    Compute the inverse Hat operator [1] of a batch of 3x3 matrices.

    Args:
        h: Batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`.

    Returns:
        Batch of 3d vectors of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `h` is of incorrect shape.
        ValueError if `h` not skew-symmetric.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

  N, dim1, dim2 = h.shape
  if dim1 != 3 or dim2 != 3:
    raise ValueError("Input has to be a batch of 3x3 Tensors.")

  ss_diff = (h + h.permute(0, 2, 1)).abs().max()
  if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
    raise ValueError("One of input matrices not skew-symmetric.")

  x = h[:, 2, 1]
  y = h[:, 0, 2]
  z = h[:, 1, 0]

  v = torch.stack((x, y, z), dim=1)

  return v


def hat(v):
  """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

  N, dim = v.shape
  if dim != 3:
    raise ValueError("Input vectors have to be 3-dimensional.")

  h = v.new_zeros(N, 3, 3)

  x, y, z = v.unbind(1)

  h[:, 0, 1] = -z
  h[:, 0, 2] = y
  h[:, 1, 0] = z
  h[:, 1, 2] = -x
  h[:, 2, 0] = -y
  h[:, 2, 1] = x

  return h
