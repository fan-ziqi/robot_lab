import torch

from . import transforms as tf


def ensure_2d_tensor(th, dtype, device):
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=dtype, device=device)
    if len(th.shape) <= 1:
        N = 1
        th = th.view(1, -1)
    else:
        N = th.shape[0]
    return th, N


class Chain:
    def __init__(self, root_frame, dtype=torch.float32, device="cpu"):
        self._root = root_frame
        self.dtype = dtype
        self.device = device

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)
        return self

    def __str__(self):
        return str(self._root)

    @staticmethod
    def _find_frame_recursive(name, frame):
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_frame(self, name):
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name, frame):
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name):
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    @staticmethod
    def _get_joint_parameter_names(frame, exclude_fixed=True):
        joint_names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.joint.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parameter_names(child, exclude_fixed))
        return joint_names

    def get_joint_parameter_names(self, exclude_fixed=True):
        names = self._get_joint_parameter_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    def add_frame(self, frame, parent_name):
        frame = self.find_frame(parent_name)
        if not frame is None:
            frame.add_child(frame)

    @staticmethod
    def _forward_kinematics(root, th_dict, world=tf.Transform3d()):
        link_transforms = {}

        th, N = ensure_2d_tensor(th_dict.get(root.joint.name, 0.0), world.dtype, world.device)

        trans = world.compose(root.get_transform(th.view(N, 1)))
        link_transforms[root.link.name] = trans.compose(root.link.offset)
        for child in root.children:
            link_transforms.update(Chain._forward_kinematics(child, th_dict, trans))
        return link_transforms

    def forward_kinematics(self, th, world=tf.Transform3d()):
        if not isinstance(th, dict):
            jn = self.get_joint_parameter_names()
            assert len(jn) == len(th)
            th_dict = {j: th[i] for i, j in enumerate(jn)}
        else:
            th_dict = th
        if world.dtype != self.dtype or world.device != self.device:
            world = world.to(dtype=self.dtype, device=self.device, copy=True)
        return self._forward_kinematics(self._root, th_dict, world)


class SerialChain(Chain):
    def __init__(self, chain, end_frame_name, root_frame_name="", **kwargs):
        if root_frame_name == "":
            super().__init__(chain._root, **kwargs)
        else:
            super().__init__(chain.find_frame(root_frame_name), **kwargs)
            if self._root is None:
                raise ValueError("Invalid root frame name %s." % root_frame_name)
        self._serial_frames = self._generate_serial_chain_recurse(self._root, end_frame_name)
        if self._serial_frames is None:
            raise ValueError("Invalid end frame name %s." % end_frame_name)

    @staticmethod
    def _generate_serial_chain_recurse(root_frame, end_frame_name):
        for child in root_frame.children:
            if child.name == end_frame_name:
                return [child]
            else:
                frames = SerialChain._generate_serial_chain_recurse(child, end_frame_name)
                if not frames is None:
                    return [child] + frames
        return None

    def get_joint_parameter_names(self, exclude_fixed=True):
        names = []
        for f in self._serial_frames:
            if exclude_fixed and f.joint.joint_type == "fixed":
                continue
            names.append(f.joint.name)
        return names

    def forward_kinematics(self, th, world=tf.Transform3d(), end_only=True):
        if world.dtype != self.dtype or world.device != self.device:
            world = world.to(dtype=self.dtype, device=self.device, copy=True)
        th, N = ensure_2d_tensor(th, self.dtype, self.device)

        cnt = 0
        link_transforms = {}
        trans = tf.Transform3d(matrix=world.get_matrix().repeat(N, 1, 1))
        for f in self._serial_frames:
            if f.joint.joint_type == "fixed":  # If fixed
                trans = trans.compose(
                    f.get_transform(th[:, 0].view(N, 1))
                )  # Use th[0] because the value is not relevant
            else:
                trans = trans.compose(f.get_transform(th[:, cnt].view(N, 1)))
                cnt += 1
            link_transforms[f.link.name] = trans.compose(f.link.offset)

        return link_transforms[self._serial_frames[-1].link.name] if end_only else link_transforms
