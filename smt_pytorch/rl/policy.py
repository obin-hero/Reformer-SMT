from .distributions import Categorical, DiagGaussian
import torch
import torch.nn as nn

class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError


    def act(self, inputs, states, masks, deterministic=False):
        ''' TODO: find out what these do '''
        raise NotImplementedError
        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError
        return value

    def evaluate_actions(self, observations, internal_states, masks, action):
        ''' TODO: find out what these do '''
        raise NotImplementedError
        return value, action_log_probs, dist_entropy, states

class PolicyWithBase(BasePolicy):
    def __init__(self, base, action_space, num_stack=4):
        '''
            Args:
                base: A unit which of type ActorCriticModule
        '''
        super().__init__()
        self.base = base
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()


    def forward(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError

    def act(self, observations, model_states, masks, deterministic=False):
        ''' TODO: find out what these do

            inputs: Observations?
            states: Model state?
            masks: ???
            deterministic: Boolean, True if the policy is deterministic
        '''

        value, actor_features, states = self.base(observations, model_states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            # Select MAP/MLE estimate (depending on if we are feeling Bayesian)
            action = dist.mode()
        else:
            # Sample from trained posterior distribution
            action = dist.sample()


        self.probs = dist.probs
        action_log_probs = dist.log_probs(action)
        self.entropy = dist.entropy().mean()
        self.perplexity = torch.exp(self.entropy)

        # if action == 0 and self.base.nav_mode == 'room':
        #    self.base.nav_mode == 'object'

        # used to force agent to one action in training env (not in all envs!) useful for debugging
        # from habitat.sims.habitat_simulator import SimulatorActions
        # action[0][0] = SimulatorActions.FORWARD.value

        # apply takeover
        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks, *args):
        ''' TODO: find out what these do '''
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action, cache={}, *args):
        ''' TODO: find out what these do '''
        value, actor_features, states = self.base(inputs, states, masks, cache)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states

    def compute_intrinsic_losses(self, intrinsic_losses, inputs, states, masks, action, cache):
        losses = {}
        for intrinsic_loss in intrinsic_losses:
            if intrinsic_loss == 'activation_l2':
                assert 'residual' in cache, f'cache does not contain residual. it contains {cache.keys()}'  # (8*4) x 16 x 16
                diff = self.l2(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'activation_l1':
                assert 'residual' in cache, f'cache does not contain residual. it contains {cache.keys()}'
                diff = self.l1(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'perceptual_l1':  # only L1 since decoder
                assert 'residual' in cache, f'cache does not contain residual. it contains {cache.keys()}'
                act_teacher = self.decoder(inputs['taskonomy'])
                act_student = self.decoder(
                    cache['residual'])  # this uses a lot of memory... make sure that ppo_num_epoch=16
                diff = self.l1(act_teacher, act_student)
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'weight':
                pass
        return losses
