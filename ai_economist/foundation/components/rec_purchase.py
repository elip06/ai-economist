import numpy as np
from ai_economist.foundation.base.base_component import BaseComponent, component_registry

@component_registry.add
class BuyRECFromVirtualStore(BaseComponent):
    name = "BuyRecFromVirtualStore"
    required_entities = ["ConsumedEnergy", "GreenScoresLastDay", "GreenScore", "RECsPrice"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # TODO: determine actual price of RECs
        self.rec_price = 0.007

        # this defines the maximum # of rec packages (each 5% of their energy consumption) an agent can purchase
        self.rec_packages = 20

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            # from 0 to 1 with a step of 0.05
            return self.rec_packages
        return None

     # def generate_masks(self, completions=0):
    #     if self.is_first_step:
    #         self.is_first_step = False
    #         if self.mask_first_step:
    #             return self.common_mask_off

    #     return self.common_mask_on

    def generate_masks(self, completions=0):
        masks = {}
        for agent in self.world.agents:
            masks[agent.idx] = np.ones(int(self.rec_packages))

        return masks
        
    def component_step(self):
        for agent in self.world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            if 0 <= action <= self.rec_packages: # Agent wants to buy none/some RECs.
                recs_percentage = action * 0.05
                new_green_score = min([1.0, agent.state["endogenous"]["InitialGreenScore"] + recs_percentage])
                agent.state["endogenous"]["GreenScoresLastDay"] = agent.state["endogenous"]["GreenScoresLastDay"][1:]
                agent.state["endogenous"]["GreenScoresLastDay"]= np.append(agent.state["endogenous"]["GreenScoresLastDay"], new_green_score)
                agent.state["endogenous"]["GreenScore"] = np.mean(agent.state["endogenous"]["GreenScoresLastDay"])
                agent.state["endogenous"]["RECsPrice"] = self.rec_price * agent.state["endogenous"]["ConsumedEnergy"] * recs_percentage
                green_score_importance = self.world.planner.state["GreenScoreImportance"]
                reliability_score_importance = 1 - green_score_importance
                agent.state["endogenous"]["TotalScore"] = (green_score_importance * agent.state["endogenous"]["GreenScore"]) + (reliability_score_importance * agent.state["endogenous"]["ReliabilityScore"])

            else: # We only declared 20 actions for this agent type, so action > 20 is an error.
                raise ValueError

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "rec_price": self.rec_price
            }

        return obs_dict