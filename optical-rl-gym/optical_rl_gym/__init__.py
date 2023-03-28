from gym.envs.registration import register

register(
    id='RMSA-v0',
    entry_point='optical_rl_gym.envs:RMSAEnv',
)

register(
    id='DeepRMSA-v0',
    entry_point='optical_rl_gym.envs:DeepRMSAEnv',
)

register(
    id='RWA-v0',
    entry_point='optical_rl_gym.envs:RWAEnv',
)

register(
    id='QoSConstrainedRA-v0',
    entry_point='optical_rl_gym.envs:QoSConstrainedRA',
)

register(
    id='DeepRMSAKSP-v0',
    entry_point='optical_rl_gym.envs:DeepRMSAKSPEnv',
)

register(
    id='DeepRMSA-v1',
    entry_point='optical_rl_gym.envs:DeepRMSAEnv1',
)

register(
    id='DeepRMSAKSP-v1',
    entry_point='optical_rl_gym.envs:DeepRMSAKSPEnv1',
)

register(
    id='DeepRMSA-v2',
    entry_point='optical_rl_gym.envs:DeepRMSAEnv2',
)