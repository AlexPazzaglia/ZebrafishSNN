"""Animat options"""

from typing import List, Dict, Union
from functools import partial

import numpy as np

from farms_core.options import Options
from farms_core.model.options import (
    AnimatOptions,
    MorphologyOptions,
    LinkOptions,
    JointOptions,
    SpawnOptions,
    ControlOptions,
    MotorOptions,
    SensorsOptions,
    WaterOptions,
    ArenaOptions,
)
from .convention import SpikingConvention

# pylint: disable=too-many-lines,too-many-arguments,
# pylint: disable=too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-instance-attributes


def options_kwargs_float_keys():
    """Options kwargs float keys"""
    return [
        'kinematics_sampling', 'kinematics_start', 'kinematics_end',
        'muscle_alpha', 'muscle_beta', 'muscle_gamma', 'muscle_delta',
    ]


def options_kwargs_float_list_keys():
    """Options kwargs float list keys"""
    return ['drives_init', 'solref']


def options_kwargs_int_keys():
    """Options kwargs int keys"""
    return ['kinematics_time_index']


def options_kwargs_int_list_keys():
    """Options kwargs int list keys"""
    return ['kinematics_indices']


def options_kwargs_str_keys():
    """Options kwargs string keys"""
    return ['drive_contact_type', 'kinematics_file']


def options_kwargs_str_list_keys():
    """Options kwargs str list keys"""
    return ['collisions_list']


def options_kwargs_bool_keys():
    """Options kwargs bool keys"""
    return ['inanimate', 'kinematics_invert', 'kinematics_degrees']


def options_kwargs_animat_keys():
    """Options kwargs animat keys"""
    return (
        options_kwargs_float_keys()
        + options_kwargs_float_list_keys()
        + options_kwargs_int_keys()
        + options_kwargs_int_list_keys()
        + options_kwargs_str_keys()
        + options_kwargs_str_list_keys()
        + options_kwargs_bool_keys()
    )

def options_kwargs_all_keys():
    """Options kwargs all keys"""
    return (
        options_kwargs_animat_keys()
    )

# --------------------- [ SPIKING ] ---------------------
class SpikingOptions(AnimatOptions):
    """Simulation options"""

    def __init__(self, sdf: str, **kwargs):
        super().__init__(
            sdf=sdf,
            spawn=SpawnOptions(**kwargs.pop('spawn')),
            morphology=AmphibiousMorphologyOptions(**kwargs.pop('morphology')),
            control=(
                KinematicsControlOptions(**kwargs.pop('control'))
                if 'kinematics_file' in kwargs['control']
                else SpikingControlOptions(**kwargs.pop('control'))
            ),
        )
        self.name = kwargs.pop('name')
        self.show_xfrc = kwargs.pop('show_xfrc')
        self.scale_xfrc = kwargs.pop('scale_xfrc')
        self.mujoco = kwargs.pop('mujoco')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def default(cls):
        """Deafault options"""
        return cls.from_options({})

    @classmethod
    def from_options(cls, kwargs=None):
        """From options"""
        options = {}
        options['sdf'] = kwargs.pop('sdf_path')
        options['name'] = kwargs.pop('name', 'Animat')
        options['morphology'] = kwargs.pop(
            'morphology',
            AmphibiousMorphologyOptions.from_options(kwargs),
        )
        convention = SpikingConvention.from_morphology(
            morphology=options['morphology'],
            **{
                key: kwargs.get(key, False)
                for key in ('single_osc_body', 'single_osc_legs')
                if key in kwargs
            },
        )
        options['spawn'] = kwargs.pop(
            'spawn',
            SpawnOptions.from_options(kwargs)
        )
        options['mujoco'] = kwargs.pop('mujoco', {})
        if 'solref' in kwargs:
            options['mujoco']['solref'] = kwargs.pop('solref')
        kinematics_file = kwargs.get('kinematics_file', None)
        if 'control' in kwargs:
            options['control'] = kwargs.pop('control')
        elif kinematics_file is not None:
            # Kinematics controller
            options['control'] = KinematicsControlOptions.from_options(kwargs)
            options['control'].defaults_from_convention(convention, kwargs)
        else:
            # Spiking controller
            options['control'] = SpikingControlOptions.from_options(kwargs)
            options['control'].defaults_from_convention(convention, kwargs)
        options['show_xfrc'] = kwargs.pop('show_xfrc', False)
        options['scale_xfrc'] = kwargs.pop('scale_xfrc', 1)
        assert not kwargs, f'Unknown kwargs: {kwargs}'
        return cls(**options)

    def state_init(self):
        """Initial states"""
        return (
            [ 0 for osc in self.control.network.oscillators ] +
            [ joint.initial[0] for joint in self.morphology.joints ]
        )

# \-------------------- [ SPIKING ] ---------------------

class AmphibiousMorphologyOptions(MorphologyOptions):
    """Amphibious morphology options"""

    def __init__(self, **kwargs):
        super().__init__(
            links=[
                AmphibiousLinkOptions(**link)
                for link in kwargs.pop('links')
            ],
            self_collisions=kwargs.pop('self_collisions'),
            joints=[
                JointOptions(**joint)
                for joint in kwargs.pop('joints')
            ],
        )
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        for kwarg in [
                'n_joints_body', 'n_dof_legs', 'n_legs',
                'links_names', 'joints_names',
        ]:
            if kwarg in kwargs.copy():
                options[kwarg] = kwargs.pop(kwarg)
        convention = SpikingConvention(**options)
        default_lateral_friction = kwargs.pop('default_lateral_friction', 1)
        # Feet handling
        feet_links = kwargs.pop('feet_links', None)
        if feet_links is None:
            feet_links = convention.feet_links_names()
        else:  # Feet defined and to be attributed
            feet_indices = [
                convention.leglink2index(
                    leg_i=leg_i,
                    side_i=side_i,
                    joint_i=convention.n_dof_legs-1,
                )
                for leg_i in range(convention.n_legs//2)
                for side_i in range(2)
            ]
            assert len(feet_indices) == len(feet_links), (
                f'len({feet_indices}) != len({feet_links})'
            )
            for index, name in zip(feet_indices, feet_links):
                convention.links_names[index] = name
        # Links and joints
        links_names = convention.links_names
        joints_names = convention.joints_names
        options.pop('links_names', None)
        options.pop('joints_names', None)
        # Feet friction
        feet_friction = kwargs.pop('feet_friction', None)
        if feet_friction is None:
            feet_friction = default_lateral_friction
        if isinstance(feet_friction, (float, int)):
            feet_friction = [feet_friction]*convention.n_legs_pair()
        elif len(feet_friction) < convention.n_legs_pair():
            feet_friction += [feet_friction[0]]*(
                convention.n_legs_pair() - len(feet_friction)
            )
        links_friction_lateral = kwargs.pop(
            'links_friction_lateral',
            [
                feet_friction[feet_links.index(link)//2]
                if link in feet_links
                else default_lateral_friction
                for link in links_names
            ],
        )
        links_friction_spinning = kwargs.pop(
            'links_friction_spinning',
            [0 for link in links_names],
        )
        links_friction_rolling = kwargs.pop(
            'links_friction_rolling',
            [0 for link in links_names],
        )
        links_no_collisions = kwargs.pop('links_no_collisions', (
            [
                convention.bodylink2name(body_i)
                for body_i in range(1, options['n_joints_body'])
            ] + [
                convention.leglink2name(leg_i, side_i, joint_i)
                for leg_i in range(options['n_legs']//2)
                for side_i in range(2)
                for joint_i in range(options['n_dof_legs']-1)
            ] if kwargs.pop('reduced_collisions', False) else []
        ))
        links_linear_damping = kwargs.pop(
            'links_linear_damping',
            [0 for link in links_names],
        )
        links_angular_damping = kwargs.pop(
            'links_angular_damping',
            [0 for link in links_names],
        )
        default_restitution = kwargs.pop('default_restitution', 0)
        links_restitution = kwargs.pop(
            'links_restitution',
            [default_restitution for link in links_names],
        )
        links_density = kwargs.pop('density', None)
        links_swimming = kwargs.pop('links_swimming', links_names)
        links_mass_multiplier = kwargs.pop('mass_multiplier', 1)
        drag_coefficients = kwargs.pop(
            'drag_coefficients',
            [None for name in links_names],
        )
        assert (
            len(links_names)
            == len(links_friction_lateral)
            == len(links_friction_spinning)
            == len(links_friction_rolling)
            == len(drag_coefficients)
        ), (
            'links_name,'
            ' links_friction_lateral,'
            ' links_friction_spinning,'
            ' links_friction_rolling,'
            ' drag_coefficients',
            np.shape(links_names),
            np.shape(links_friction_lateral),
            np.shape(links_friction_spinning),
            np.shape(links_friction_rolling),
            np.shape(drag_coefficients),
            links_names,
        )
        options['links'] = kwargs.pop(
            'links',
            [
                AmphibiousLinkOptions(
                    name=name,
                    collisions=name not in links_no_collisions,
                    density=links_density,
                    mass_multiplier=links_mass_multiplier,
                    swimming=name in links_swimming,
                    drag_coefficients=drag,
                    friction=[lateral, spin, roll],
                    extras={
                        'restitution': restitution,
                        'linearDamping': linear,
                        'angularDamping': angular,
                    },
                )
                for (
                        name, lateral, spin, roll,
                        drag, linear, angular, restitution
                ) in zip(
                    links_names,
                    links_friction_lateral,
                    links_friction_spinning,
                    links_friction_rolling,
                    drag_coefficients,
                    links_linear_damping,
                    links_angular_damping,
                    links_restitution,
                )
            ]
        )
        options['self_collisions'] = kwargs.pop('self_collisions', [])
        joints_positions = kwargs.pop(
            'joints_positions',
            [0 for name in joints_names]
        )
        joints_velocities = kwargs.pop(
            'joints_velocities',
            [0 for name in joints_names]
        )
        joints_stiffness = kwargs.pop(
            'joints_stiffness',
            [0 for name in joints_names]
        )
        joints_damping = kwargs.pop(
            'joints_damping',
            [0 for name in joints_names]
        )
        max_velocity = kwargs.pop('max_velocity', np.inf)
        if 'joints' not in kwargs:
            assert all(len(element) == len(joints_names) for element in (
                joints_positions,
                joints_velocities,
                joints_stiffness,
                joints_damping,
            )), (
                'Not all same size:'
                f' position: {len(joints_positions)},'
                f' velocity: {len(joints_velocities)},'
                f' stiffness: {len(joints_stiffness)}'
                f' damping: {len(joints_damping)}'
            )
        options['joints'] = kwargs.pop(
            'joints',
            [
                JointOptions(
                    name=name,
                    initial=[position, velocity],
                    stiffness=stiffness,
                    springref=0,
                    damping=damping,
                    limits=[
                        [-np.inf, np.inf],
                        [-max_velocity, max_velocity],
                    ],
                    extras={},
                )
                for name, position, velocity, stiffness, damping in zip(
                    joints_names,
                    joints_positions,
                    joints_velocities,
                    joints_stiffness,
                    joints_damping,
                )
            ]
        )
        morphology = cls(**options)
        if kwargs.pop('use_self_collisions', False):
            convention = SpikingConvention.from_morphology(morphology)
            morphology.self_collisions += [
                # Body-body collisions
                [
                    convention.bodylink2name(body0),
                    convention.bodylink2name(body1),
                ]
                for body0 in range(options['n_joints_body']+1)
                for body1 in range(options['n_joints_body']+1)
                if abs(body1 - body0) > 2  # Avoid neighbouring collisions
            ] + [
                # Body-leg collisions
                [
                    convention.bodylink2name(body0),
                    convention.leglink2name(leg_i, side_i, joint_i),
                ]
                for body0 in range(options['n_joints_body']+1)
                for leg_i in range(options['n_legs']//2)
                for side_i in range(2)
                for joint_i in [options['n_dof_legs']-1]  # End-effector
            ] + [
                # Leg-leg collisions
                [
                    convention.leglink2name(leg0, side0, joint0),
                    convention.leglink2name(leg1, side1, joint1),
                ]
                for leg0 in range(options['n_legs']//2)
                for leg1 in range(options['n_legs']//2)
                for side0 in range(2)
                for side1 in range(2)
                for joint0 in [options['n_dof_legs']-1]
                for joint1 in [options['n_dof_legs']-1]
                if leg0 != leg1 or side0 != side1 or joint0 != joint1
            ]
            for links in morphology.self_collisions:
                assert links[0] != links[1], f'Collision to self: {links}'
        collisions_list = kwargs.pop('collisions_list', [])
        if collisions_list:
            morphology.self_collisions += [
                [
                    collisions_list[2*i+0],
                    collisions_list[2*i+1],
                ]
                for i in range(len(collisions_list)//2)
            ]
        return morphology

    def n_joints_legs(self):
        """Number of legs joints"""
        return self.n_legs*self.n_dof_legs


class AmphibiousLinkOptions(LinkOptions):
    """Amphibious link options"""

    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.pop('name'),
            collisions=kwargs.pop('collisions'),
            friction=kwargs.pop('friction'),
            extras=kwargs.pop('extras', {}),
        )
        self.density = kwargs.pop('density')
        self.swimming = kwargs.pop('swimming')
        self.drag_coefficients = kwargs.pop('drag_coefficients')
        self.mass_multiplier: float = kwargs.pop('mass_multiplier')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


# --------------------- [ SPIKING ] ---------------------
class SpikingControlOptions(ControlOptions):
    """Spiking control options"""

    def __init__(self, **kwargs):
        super().__init__(
            sensors=(AmphibiousSensorsOptions(**kwargs.pop('sensors'))),
            motors=[
                AmphibiousMotorOptions(**motor)
                for motor in kwargs.pop('motors')
            ],
        )
        network_options = kwargs.pop('network', None)
        self.network = (
            SpikingNetworkOptions(**network_options)
            if network_options is not None
            and 'oscillators' in network_options
            else None
        )
        self.muscles = [
            AmphibiousMuscleSetOptions(**muscle)
            for muscle in kwargs.pop('muscles')
        ]
        self.hill_muscles = kwargs.pop('hill_muscles', [])
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def options_from_kwargs(cls, kwargs):
        """Options from kwargs"""
        options = super(cls, cls).options_from_kwargs({
            'sensors': kwargs.pop(
                'sensors',
                AmphibiousSensorsOptions.options_from_kwargs(kwargs),
            ),
            'motors': kwargs.pop('motors', {}),
        })
        options['network'] = kwargs.pop(
            'network',
            SpikingNetworkOptions.from_options(kwargs).to_dict()
        )
        options['muscles'] = kwargs.pop('muscles', [])
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""
        self.sensors.defaults_from_convention(convention, kwargs)
        self.network.defaults_from_convention(convention, kwargs)

        # Joints
        n_joints = convention.n_joints()
        offsets = [None]*n_joints

        # Motor gains
        motor_gains = kwargs.pop('motor_gains', [[0]]*n_joints)

        # Turning body
        for joint_i in range(convention.n_joints_body):
            for side_i in range(2):
                offsets[convention.bodyjoint2index(joint_i=joint_i)] = (
                    AmphibiousMotorOffsetOptions(
                        gain=0,
                        bias=0,
                        low=1,
                        high=5,
                        saturation=0,
                        rate=2,
                    )
                )

        # Turning legs
        legs_offsets_walking = kwargs.pop(
            'legs_offsets_walking',
            [0]*convention.n_dof_legs
        )
        legs_offsets_swimming = kwargs.pop(
            'legs_offsets_swimming',
            [0]*convention.n_dof_legs
        )
        leg_turn_gain = kwargs.pop(
            'leg_turn_gain',
            [0, 0]
            if convention.n_legs == 4
            else (-np.ones(convention.n_legs_pair())).tolist()
        )
        leg_side_turn_gain = kwargs.pop(
            'leg_side_turn_gain',
            [0, 0]
        )
        leg_joint_turn_gain = kwargs.pop(
            'leg_joint_turn_gain',
            [0]*convention.n_dof_legs
        )

        # Augment parameters
        repeat = partial(np.repeat, repeats=convention.n_legs_pair(), axis=0)
        if np.ndim(legs_offsets_walking) == 1:
            legs_offsets_walking = repeat([legs_offsets_walking]).tolist()
        if np.ndim(legs_offsets_swimming) == 1:
            legs_offsets_swimming = repeat([legs_offsets_swimming]).tolist()
        if np.ndim(leg_side_turn_gain) == 1:
            leg_side_turn_gain = repeat([leg_side_turn_gain]).tolist()
        if np.ndim(leg_joint_turn_gain) == 1:
            leg_joint_turn_gain = repeat([leg_joint_turn_gain]).tolist()

        # Motors offsets for walking and swimming
        for leg_i in range(convention.n_legs_pair()):
            for side_i in range(2):
                for joint_i in range(convention.n_dof_legs):
                    offsets[convention.legjoint2index(
                        leg_i=leg_i,
                        side_i=side_i,
                        joint_i=joint_i,
                    )] = AmphibiousMotorOffsetOptions(
                        gain=(
                            leg_turn_gain[leg_i]
                            * leg_side_turn_gain[leg_i][side_i]
                            * leg_joint_turn_gain[leg_i][joint_i]
                        ),
                        bias=legs_offsets_walking[leg_i][joint_i],
                        low=1,
                        high=3,
                        saturation=legs_offsets_swimming[leg_i][joint_i],
                        rate=2,
                    )

        # Amphibious joints control
        if not self.motors:
            self.motors = [
                AmphibiousMotorOptions(
                    joint_name=None,
                    control_types=[],
                    limits_torque=None,
                    gains=None,
                    equation=None,
                    transform=AmphibiousMotorTransformOptions(
                        gain=None,
                        bias=None,
                    ),
                    offsets=AmphibiousMotorOffsetOptions(
                        gain=None,
                        bias=None,
                        low=None,
                        high=None,
                        saturation=None,
                        rate=None,
                    ),
                    passive=AmphibiousPassiveJointOptions(
                        is_passive=False,
                        stiffness_coefficient=0,
                        damping_coefficient=0,
                        friction_coefficient=0,
                    ),
                )
                for joint in range(n_joints)
            ]
        joints_names = kwargs.pop(
            'joints_control_names',
            convention.joints_names,
        )
        transform_gain = kwargs.pop(
            'transform_gain',
            {joint_name: 1 for joint_name in joints_names},
        )
        transform_bias = kwargs.pop(
            'transform_bias',
            {joint_name: 0 for joint_name in joints_names},
        )
        default_max_torque = kwargs.pop('default_max_torque', np.inf)
        max_torques = kwargs.pop(
            'max_torques',
            {joint_name: default_max_torque for joint_name in joints_names},
        )
        default_equation = kwargs.pop('default_equation', 'position')
        equations = kwargs.pop(
            'equations',
            {
                joint_name: (
                    'phase'
                    if convention.single_osc_body
                    and joint_i < convention.n_joints_body
                    or convention.single_osc_legs
                    and joint_i >= convention.n_joints_body
                    else default_equation
                )
                for joint_i, joint_name in enumerate(joints_names)
            },
        )
        for motor_i, motor in enumerate(self.motors):

            # Control
            if motor.joint_name is None:
                motor.joint_name = joints_names[motor_i]
            if motor.equation is None:
                motor.equation = equations[motor.joint_name]
            if not motor.control_types:
                motor.control_types = {
                    'position': ['position'],
                    'phase': ['position'],
                    'ekeberg_muscle': ['velocity', 'torque'],
                    'ekeberg_muscle_explicit': ['torque'],
                    'passive': ['velocity', 'torque'],
                    'passive_explicit': ['torque'],
                }[motor.equation]
            if motor.limits_torque is None:
                motor.limits_torque = [
                    -max_torques[motor.joint_name],
                    +max_torques[motor.joint_name],
                ]
            if motor.gains is None:
                motor.gains = motor_gains[motor_i]

            # Transform
            if motor.transform.gain is None:
                motor.transform.gain = transform_gain[motor.joint_name]
            if motor.transform.bias is None:
                motor.transform.bias = transform_bias[motor.joint_name]

            # Offset
            if motor.offsets.gain is None:
                motor.offsets.gain = offsets[motor_i]['gain']
            if motor.offsets.bias is None:
                motor.offsets.bias = offsets[motor_i]['bias']
            if motor.offsets.low is None:
                motor.offsets.low = offsets[motor_i]['low']
            if motor.offsets.high is None:
                motor.offsets.high = offsets[motor_i]['high']
            if motor.offsets.saturation is None:
                motor.offsets.saturation = offsets[motor_i]['saturation']
            if motor.offsets.rate is None:
                motor.offsets.rate = offsets[motor_i]['rate']

        # Passive
        joints_passive = kwargs.pop('joints_passive', [])
        self.sensors.joints += [name for name, *_ in joints_passive]
        self.motors += [
            AmphibiousMotorOptions(
                joint_name=joint_name,
                control_types=['velocity', 'torque'],
                limits_torque=[-default_max_torque, default_max_torque],
                gains=None,
                equation='passive',
                transform=AmphibiousMotorTransformOptions(
                    gain=1,
                    bias=0,
                ),
                offsets=None,
                passive=AmphibiousPassiveJointOptions(
                    is_passive=True,
                    stiffness_coefficient=stiffness,
                    damping_coefficient=damping,
                    friction_coefficient=friction,
                ),
            )
            for joint_name, stiffness, damping, friction in joints_passive
        ]

        # Muscles
        if not self.muscles:
            self.muscles = [
                AmphibiousMuscleSetOptions(
                    joint_name=None,
                    osc1=None,
                    osc2=None,
                    alpha=None,
                    beta=None,
                    gamma=None,
                    delta=None,
                    epsilon=None,
                )
                for joint_i in range(n_joints)
            ]
        default_alpha = kwargs.pop('muscle_alpha', 0)
        default_beta = kwargs.pop('muscle_beta', 0)
        default_gamma = kwargs.pop('muscle_gamma', 0)
        default_delta = kwargs.pop('muscle_delta', 0)
        default_epsilon = kwargs.pop('muscle_epsilon', 0)
        for joint_i, muscle in enumerate(self.muscles):
            if muscle.joint_name is None:
                muscle.joint_name = joints_names[joint_i]
            if muscle.osc1 is None or muscle.osc2 is None:
                osc_idx = convention.osc_indices(joint_i)
                assert osc_idx[0] < len(self.network.oscillators), (
                    f'{joint_i}: '
                    f'{osc_idx[0]} !< {len(self.network.oscillators)}'
                )
                muscle.osc1 = self.network.oscillators[osc_idx[0]].name
                if len(osc_idx) > 1:
                    assert osc_idx[1] < len(self.network.oscillators), (
                        f'{joint_i}: '
                        f'{osc_idx[1]} !< {len(self.network.oscillators)}'
                    )
                    muscle.osc2 = self.network.oscillators[osc_idx[1]].name
            if muscle.alpha is None:
                muscle.alpha = default_alpha
            if muscle.beta is None:
                muscle.beta = default_beta
            if muscle.gamma is None:
                muscle.gamma = default_gamma
            if muscle.delta is None:
                muscle.delta = default_delta
            if muscle.epsilon is None:
                muscle.epsilon = default_epsilon

    def motors_offsets(self):
        """Motors offsets"""
        return [
            {
                key: getattr(motor.offsets, key)
                for key in ['gain', 'bias', 'low', 'high', 'saturation']
            }
            for motor in self.motors
            if motor.offsets is not None
        ]

    def motors_offset_rates(self):
        """Motors rates"""
        return [
            motor.offsets.rate
            for motor in self.motors
            if motor.offsets is not None
        ]

    def motors_transform_gain(self):
        """Motors gain amplitudes"""
        return [motor.transform.gain for motor in self.motors]

    def motors_transform_bias(self):
        """Motors offset bias"""
        return [motor.transform.bias for motor in self.motors]

    def drives_contacts_indices(self):
        """Drives contacts indices"""
        for drive in self.network.drives:
            for contact in drive.contacts:
                assert contact in self.sensors.contacts, (
                    f'{contact=} not in {self.sensors.contacts=}'
                )
        return [
            [self.sensors.contacts.index(contact) for contact in drive.contacts]
            for drive in self.network.drives
        ]

# \-------------------- [ SPIKING ] ---------------------

class KinematicsControlOptions(ControlOptions):
    """Amphibious kinematics control options"""

    def __init__(self, **kwargs):
        super().__init__(
            sensors=(AmphibiousSensorsOptions(**kwargs.pop('sensors'))),
            motors=[
                AmphibiousMotorOptions(**motor)
                for motor in kwargs.pop('motors')
            ],
            # muscles=kwargs.pop('muscles', []),
        )
        self.hill_muscles = kwargs.pop('hill_muscles', [])
        self.kinematics_file = kwargs.pop('kinematics_file')
        self.kinematics_sampling = kwargs.pop('kinematics_sampling')
        self.kinematics_indices = kwargs.pop('kinematics_indices')
        self.kinematics_time_index = kwargs.pop('kinematics_time_index')
        self.kinematics_invert = kwargs.pop('kinematics_invert')
        self.kinematics_degrees = kwargs.pop('kinematics_degrees')
        self.kinematics_start = kwargs.pop('kinematics_start')
        self.kinematics_end = kwargs.pop('kinematics_end')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def options_from_kwargs(cls, kwargs):
        """Options from kwargs"""
        options = super(cls, cls).options_from_kwargs({
            'sensors': kwargs.pop(
                'sensors',
                AmphibiousSensorsOptions.options_from_kwargs(kwargs),
            ),
            'motors': kwargs.pop('motors', {}),
        })
        options['kinematics_file'] = kwargs.pop('kinematics_file', '')
        options['kinematics_sampling'] = kwargs.pop('kinematics_sampling', 0)
        options['kinematics_indices'] = kwargs.pop('kinematics_indices', None)
        options['kinematics_time_index'] = kwargs.pop('kinematics_time_index', None)
        options['kinematics_degrees'] = kwargs.pop('kinematics_degrees', False)
        options['kinematics_invert'] = kwargs.pop('kinematics_invert', False)
        options['kinematics_start'] = kwargs.pop('kinematics_start', 0)
        options['kinematics_end'] = kwargs.pop('kinematics_end', 0)
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""
        self.sensors.defaults_from_convention(convention, kwargs)

        # Joints
        n_joints = convention.n_joints()
        offsets = [None]*n_joints

        # Motor gains
        motor_gains = kwargs.pop('motor_gains', [[0]]*n_joints)

        # Turning body
        for joint_i in range(convention.n_joints_body):
            for side_i in range(2):
                offsets[convention.bodyjoint2index(joint_i=joint_i)] = (
                    AmphibiousMotorOffsetOptions(
                        gain=0,
                        bias=0,
                        low=1,
                        high=5,
                        saturation=0,
                        rate=2,
                    )
                )

        # Turning legs
        legs_offsets_walking = kwargs.pop(
            'legs_offsets_walking',
            [0]*convention.n_dof_legs,
        )
        legs_offsets_swimming = kwargs.pop(
            'legs_offsets_swimming',
            [0]*convention.n_dof_legs,
        )
        leg_turn_gain = kwargs.pop(
            'leg_turn_gain',
            [0, 0]
            if convention.n_legs == 4
            else (-np.ones(convention.n_legs_pair())).tolist(),
        )
        leg_side_turn_gain = kwargs.pop(
            'leg_side_turn_gain',
            [0, 0],
        )
        leg_joint_turn_gain = kwargs.pop(
            'leg_joint_turn_gain',
            [0]*convention.n_dof_legs,
        )

        # Augment parameters
        repeat = partial(np.repeat, repeats=convention.n_legs_pair(), axis=0)
        if np.ndim(legs_offsets_walking) == 1:
            legs_offsets_walking = repeat([legs_offsets_walking]).tolist()
        if np.ndim(legs_offsets_swimming) == 1:
            legs_offsets_swimming = repeat([legs_offsets_swimming]).tolist()
        if np.ndim(leg_side_turn_gain) == 1:
            leg_side_turn_gain = repeat([leg_side_turn_gain]).tolist()
        if np.ndim(leg_joint_turn_gain) == 1:
            leg_joint_turn_gain = repeat([leg_joint_turn_gain]).tolist()

        # Motors offsets for walking and swimming
        for leg_i in range(convention.n_legs_pair()):
            for side_i in range(2):
                for joint_i in range(convention.n_dof_legs):
                    offsets[convention.legjoint2index(
                        leg_i=leg_i,
                        side_i=side_i,
                        joint_i=joint_i,
                    )] = AmphibiousMotorOffsetOptions(
                        gain=(
                            leg_turn_gain[leg_i]
                            * leg_side_turn_gain[leg_i][side_i]
                            * leg_joint_turn_gain[leg_i][joint_i]
                        ),
                        bias=legs_offsets_walking[leg_i][joint_i],
                        low=1,
                        high=3,
                        saturation=legs_offsets_swimming[leg_i][joint_i],
                        rate=2,
                    )

        # Amphibious joints control
        if not self.motors:
            self.motors = [
                AmphibiousMotorOptions(
                    joint_name=None,
                    control_types=[],
                    limits_torque=None,
                    gains=None,
                    equation=None,
                    transform=AmphibiousMotorTransformOptions(
                        gain=None,
                        bias=None,
                    ),
                    offsets=AmphibiousMotorOffsetOptions(
                        gain=None,
                        bias=None,
                        low=None,
                        high=None,
                        saturation=None,
                        rate=None,
                    ),
                    passive=AmphibiousPassiveJointOptions(
                        is_passive=False,
                        stiffness_coefficient=0,
                        damping_coefficient=0,
                        friction_coefficient=0,
                    ),
                )
                for joint in range(n_joints)
            ]
        joints_names = kwargs.pop(
            'joints_control_names',
            convention.joints_names,
        )
        transform_gain = kwargs.pop(
            'transform_gain',
            {joint_name: 1 for joint_name in joints_names},
        )
        transform_bias = kwargs.pop(
            'transform_bias',
            {joint_name: 0 for joint_name in joints_names},
        )
        default_max_torque = kwargs.pop('default_max_torque', np.inf)
        max_torques = kwargs.pop(
            'max_torques',
            {joint_name: default_max_torque for joint_name in joints_names},
        )
        default_equation = kwargs.pop('default_equation', 'position')
        equations = kwargs.pop(
            'equations',
            {
                joint_name: (
                    'phase'
                    if convention.single_osc_body
                    and joint_i < convention.n_joints_body
                    or convention.single_osc_legs
                    and joint_i >= convention.n_joints_body
                    else default_equation
                )
                for joint_i, joint_name in enumerate(joints_names)
            },
        )
        for motor_i, motor in enumerate(self.motors):

            # Control
            if motor.joint_name is None:
                motor.joint_name = joints_names[motor_i]
            if motor.equation is None:
                motor.equation = equations[motor.joint_name]
            if not motor.control_types:
                motor.control_types = {
                    'position': ['position'],
                }[motor.equation]
            if motor.limits_torque is None:
                motor.limits_torque = [
                    -max_torques[motor.joint_name],
                    +max_torques[motor.joint_name],
                ]
            if motor.gains is None:
                motor.gains = motor_gains[motor_i]

            # Transform
            if motor.transform.gain is None:
                motor.transform.gain = transform_gain[motor.joint_name]
            if motor.transform.bias is None:
                motor.transform.bias = transform_bias[motor.joint_name]

            # Offset
            if motor.offsets.gain is None:
                motor.offsets.gain = offsets[motor_i]['gain']
            if motor.offsets.bias is None:
                motor.offsets.bias = offsets[motor_i]['bias']
            if motor.offsets.low is None:
                motor.offsets.low = offsets[motor_i]['low']
            if motor.offsets.high is None:
                motor.offsets.high = offsets[motor_i]['high']
            if motor.offsets.saturation is None:
                motor.offsets.saturation = offsets[motor_i]['saturation']
            if motor.offsets.rate is None:
                motor.offsets.rate = offsets[motor_i]['rate']

        # Passive
        joints_passive = kwargs.pop('joints_passive', [])
        self.sensors.joints += [name for name, *_ in joints_passive]
        self.motors += [
            AmphibiousMotorOptions(
                joint_name=joint_name,
                control_types=['velocity', 'torque'],
                limits_torque=[-default_max_torque, default_max_torque],
                equation='passive',
                transform=AmphibiousMotorTransformOptions(
                    gain=1,
                    bias=0,
                ),
                offsets=None,
                passive=AmphibiousPassiveJointOptions(
                    is_passive=True,
                    stiffness_coefficient=stiffness,
                    damping_coefficient=damping,
                    friction_coefficient=friction,
                ),
            )
            for joint_name, stiffness, damping, friction in joints_passive
        ]

    def motors_transform_gain(self):
        """Motors gain amplitudes"""
        return [motor.transform.gain for motor in self.motors]

    def motors_transform_bias(self):
        """Motors offset bias"""
        return [motor.transform.bias for motor in self.motors]


class AmphibiousMotorOptions(MotorOptions):
    """Amphibious motor options"""

    def __init__(self, **kwargs):
        super().__init__(
            joint_name=kwargs.pop('joint_name'),
            control_types=kwargs.pop('control_types'),
            limits_torque=kwargs.pop('limits_torque'),
            gains=kwargs.pop('gains'),
        )
        self.equation: str = kwargs.pop('equation')
        transform = kwargs.pop('transform')
        self.transform: AmphibiousMotorTransformOptions = (
            AmphibiousMotorTransformOptions(**transform)
            if transform is not None
            else None
        )
        offsets = kwargs.pop('offsets')
        self.offsets: AmphibiousMotorOffsetOptions = (
            AmphibiousMotorOffsetOptions(**offsets)
            if offsets is not None
            else None
        )
        passive = kwargs.pop('passive')
        self.passive: AmphibiousPassiveJointOptions = (
            AmphibiousPassiveJointOptions(**passive)
            if passive is not None
            else None
        )
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousMotorTransformOptions(Options):
    """Amphibious motor options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain: float = kwargs.pop('gain')
        self.bias: float = kwargs.pop('bias')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousMotorOffsetOptions(Options):
    """Amphibious motor options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain: float = kwargs.pop('gain')
        self.bias: float = kwargs.pop('bias')
        self.low: float = kwargs.pop('low')
        self.high: float = kwargs.pop('high')
        self.saturation: float = kwargs.pop('saturation')
        self.rate: float = kwargs.pop('rate')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousSensorsOptions(SensorsOptions):
    """Amphibious sensors options"""

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""
        self.links = kwargs.pop('sensors_links', convention.links_names)
        self.joints = kwargs.pop('sensors_joints', convention.joints_names)
        self.contacts = kwargs.pop('sensors_contacts', None)
        self.xfrc = kwargs.pop('sensors_xfrc', convention.links_names)
        if self.contacts is None:
            self.contacts = convention.feet_links_names()


# --------------------- [ SPIKING ] ---------------------
class SpikingNetworkOptions(Options):
    """Spiking network options"""

    def __init__(self, **kwargs):
        super().__init__()

        # Oscillators
        self.oscillators: List[SpikingOscillatorOptions] = [
            SpikingOscillatorOptions(**oscillator)
            for oscillator in kwargs.pop('oscillators')
        ]

        # Kwargs
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['oscillators'] = kwargs.pop('oscillators', [])
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""

        # Oscillators
        n_oscillators = convention.n_osc()
        if not self.oscillators:
            self.oscillators = [
                SpikingOscillatorOptions(name=None)
                for osc_i in range(n_oscillators)
            ]

        state_init = np.concatenate([
            np.zeros(convention.n_osc()),       # Neural activities
            np.zeros(convention.n_joints()),    # Joints
        ])
        assert len(state_init) == convention.n_states()

        for osc_i, osc in enumerate(self.oscillators):
            if osc.name is None:
                osc.name = convention.oscindex2name(osc_i)

    def n_oscillators(self):
        """Number of oscillators"""
        return len(self.oscillators)

    def osc_names(self):
        """Oscillator names"""
        return [osc.name for osc in self.oscillators]

# \-------------------- [ SPIKING ] ---------------------

# --------------------- [ SPIKING ] ---------------------
class SpikingOscillatorOptions(Options):
    """Amphibious oscillator options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name = kwargs.pop('name')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

# \-------------------- [ SPIKING ] ---------------------


class AmphibiousMuscleSetOptions(Options):
    """Amphibious muscle options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.joint_name: str = kwargs.pop('joint_name')
        self.osc1: str = kwargs.pop('osc1')
        self.osc2: str = kwargs.pop('osc2')
        self.alpha: float = kwargs.pop('alpha')  # Gain
        self.beta: float = kwargs.pop('beta')  # Stiffness gain
        self.gamma: float = kwargs.pop('gamma')  # Tonic gain
        self.delta: float = kwargs.pop('delta')  # Damping coefficient
        self.epsilon: float = kwargs.pop('epsilon')  # Friction coefficient
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousPassiveJointOptions(Options):
    """Amphibious passive joint options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.is_passive: bool = kwargs.pop('is_passive')
        self.stiffness_coefficient: float = kwargs.pop('stiffness_coefficient')
        self.damping_coefficient: float = kwargs.pop('damping_coefficient')
        self.friction_coefficient: float = kwargs.pop('friction_coefficient')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

class AmphibiousArenaOptions(ArenaOptions):
    """Amphibious arena options"""

    def __init__(
            self,
            sdf: str,
            spawn: Union[SpawnOptions, Dict],
            water: Union[WaterOptions, Dict],
            ground_height: float,
    ):
        super().__init__(
            sdf=sdf,
            spawn=spawn,
            water=(
                water
                if isinstance(water, WaterOptions)
                else WaterOptions(**water)
            ),
            ground_height=ground_height,
        )
