import numpy as np
from ai_economist.foundation.base.base_component import BaseComponent, component_registry

@component_registry.add
class ChangeMinerSelectionPolicy(BaseComponent):
    name = "ChangeMinerSelectionPolicy"
    required_entities = ["TotalScore", "GreenScore", "ReliabilityScore"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    def __init__(
        self,
        *base_component_args,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # this defines the maximum importance of the green score (20 is equal to 100%, 1 to 5%)
        self.green_score_importance = 20
        self.policy_interval = 24

        self.default_planner_action_mask = [1 for _ in range(self.green_score_importance)]
        self.no_op_planner_action_mask = [0 for _ in range(self.green_score_importance)]

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicPlanner":
            # from 0 to 1 with a step of 0.05
            return self.green_score_importance
        return None

     # def generate_masks(self, completions=0):
    #     if self.is_first_step:
    #         self.is_first_step = False
    #         if self.mask_first_step:
    #             return self.common_mask_off

    #     return self.common_mask_on

    def generate_masks(self, completions=0):
        masks = {}
        if self.world.timestep % self.policy_interval == 0:
            masks[self.world.planner.idx] = self.default_planner_action_mask
        else:
            masks[self.world.planner.idx] = self.no_op_planner_action_mask
        return masks

    def component_step(self):
        planner_action = self.world.planner.get_component_action(self.name)
        if 0 <= planner_action <= self.green_score_importance:
            if self.world.timestep % self.policy_interval == 0:
                green_score_importance = planner_action * 0.05
                reliability_score_importance = 1 - green_score_importance
                for agent in self.world.get_random_order_agents():
                    agent.state["endogenous"]["TotalScore"] = (green_score_importance * agent.state["endogenous"]["GreenScore"]) + (reliability_score_importance * agent.state["endogenous"]["ReliabilityScore"])

        else: # We only declared 20 actions for this agent type, so action > 20 is an error.
            raise ValueError

    def generate_observations(self):
        obs = {}
        return obs