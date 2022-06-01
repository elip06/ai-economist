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
        green_score_importance=0.0,
        static=False,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.rec_price = 0.025
        
        # if static is true, green_score_importance doesn't change
        self.green_score_importance = green_score_importance
        self.static = static
        
        # this defines the maximum # of rec packages (each 5% of their energy consumption) an agent can purchase
        self.rec_packages = 21

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            # from 0 to 1 with a step of 0.05
            return self.rec_packages
        return None

    def generate_masks(self, completions=0):
        masks = {}
        for agent in self.world.agents:
            masks[agent.idx] = np.ones(int(self.rec_packages))

        return masks
        
    def component_step(self):
        for agent in self.world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            if 0 <= action <= self.rec_packages: # Agent wants to buy none/some RECs.
                recs_percentage = (action-1) * 0.05
                new_green_score = min([1.0, agent.state["endogenous"]["InitialGreenScore"] + recs_percentage])
                agent.state["endogenous"]["GreenScoresLastDay"] = agent.state["endogenous"]["GreenScoresLastDay"][1:]
                agent.state["endogenous"]["GreenScoresLastDay"]= np.append(agent.state["endogenous"]["GreenScoresLastDay"], new_green_score)
                if np.sum(agent.state["endogenous"]["ConsumedEnergy"]) > 0.0:
                    agent.state["endogenous"]["GreenScore"] = np.sum(agent.state["endogenous"]["GreenScoresLastDay"] * agent.state["endogenous"]["ConsumedEnergy"]) / np.sum(agent.state["endogenous"]["ConsumedEnergy"])
                else:
                    agent.state["endogenous"]["GreenScore"] = np.mean(agent.state["endogenous"]["GreenScoresLastDay"])
                agent.state["endogenous"]["RECsPrice"] = self.rec_price * agent.state["endogenous"]["ConsumedEnergy"][-1] * recs_percentage
                green_score_importance = self.green_score_importance
                if not self.static:
                    green_score_importance = self.world.planner.state["GreenScoreImportance"]
                reliability_score_importance = 1 - green_score_importance
                agent.state["endogenous"]["TotalScore"] = (green_score_importance * agent.state["endogenous"]["GreenScore"]) + (reliability_score_importance * agent.state["endogenous"]["ReliabilityScore"])

            else: # We only declared 21 actions for this agent type, so action > 21 is an error.
                raise ValueError

    def generate_observations(self):
        obs_dict = {}

        return obs_dict