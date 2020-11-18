import torch


class Retrace:
    def __init__(self):
        super(Retrace, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __call__(self,
                 Q,
                 expected_target_Q,
                 target_Q,
                 rewards,
                 target_policy_probs,
                 behaviour_policy_probs,
                 gamma=0.99,
                 logger=None):
        """
        Implementation of Retrace loss ((http://arxiv.org/abs/1606.02647)) in PyTorch.
        Args:
            Q: State-Action values.
            Torch tensor with shape `[B, T]`
            expected_target_Q: 𝔼_π Q(s_t,.) (from the fixed critic network)
            Torch tensor with shape `[B, T]`
            target_Q: State-Action values from target network.
            Torch tensor with shape `[B, T]`
            rewards: Holds rewards for taking an action in the environment.
            Torch tensor with shape `[B, T]`
            target_policy_probs: Probability of target policy π(a|s)
            Torch tensor with shape `[B, T]`
            behaviour_policy_probs: Probability of behaviour policy b(a|s)
            Torch tensor with shape `[B, T]`
            gamma: Discount factor

            where B is the minibatch size and T is the rollout length
        Returns:
            Computes the retrace loss recursively according to
            L = 𝔼_τ[(Q_t - Q_ret_t)^2]
            Q_ret_t = r_t + γ * (𝔼_π_target [Q(s_t+1,•)] + c_t+1 * Q_π_target(s_t+1,a_t+1)) + γ * c_t+1 * Q_ret_t+1
            with trajectory τ = {(s_0, a_0, r_0),..,(s_k, a_k, r_k)}

        The recursive form for calculating Q_ret can be found in 'Sample Efficient Actor-Critic with Experience Replay'
        (https://arxiv.org/abs/1611.01224)
        """
        # Check dimension of inputs
        assert Q.shape == target_Q.shape == expected_target_Q.shape == rewards.shape == target_policy_probs.shape
        T = Q.shape[1]  # total number of time steps in the trajectory

        with torch.no_grad():
            # We don't want gradients from computing Q_ret, since:
            # ∇φ (Q - Q_ret)^2 ∝ (Q - Q_ret) * ∇φ Q
            c_ret = self.calc_retrace_weights(target_policy_probs, behaviour_policy_probs)
            Q_ret = torch.zeros_like(Q, dtype=torch.float)  # (B,T)

            Q_ret[:, -1] = target_Q[:, -1]
            for t in reversed(range(1, T)):
                Q_ret[:, t - 1] = rewards[:, t - 1] + gamma * c_ret[:, t] * (Q_ret[:, t] - target_Q[:, t]) + \
                                  gamma * expected_target_Q[:, t]

        return ((Q - Q_ret) ** 2).mean()

    @staticmethod
    def calc_retrace_weights(target_policy_logprob, behaviour_policy_logprob):
        """
        Calculates the retrace weights (truncated importance weights) c according to:
        c_t = min(1, π_target(a_t|s_t) / b(a_t|s_t)) where:
        π_target: target policy probabilities
        b: behaviour policy probabilities

        Calculation in log space for increased numerical stability.
        Args:
            target_policy_logprob: log π_target(a_t|s_t)
            behaviour_policy_logprob: log b(a_t|s_t)
        Returns:
            retrace weights c
        """
        log_retrace_weights = (target_policy_logprob - behaviour_policy_logprob).clamp(max=0)
        retrace_weights = log_retrace_weights.exp()
        assert not torch.isnan(log_retrace_weights).any(), "Error, a least one NaN value found in retrace weights."
        return retrace_weights
