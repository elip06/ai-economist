# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from ai_economist.foundation.base.registrar import Registry


class Endogenous:
    """Base class for endogenous entity classes.

    Endogenous entities are those that, conceptually, describe the internal state
    of an agent. This provides a convenient way to separate physical entities (which
    may exist in the world, be exchanged among agents, or are otherwise in principal
    observable by others) from endogenous entities (such as the amount of labor
    effort an agent has experienced).

    Endogenous entities are registered in the "endogenous" portion of an agent's
    state and should only be observable by the agent itself.
    """

    name = None

    def __init__(self):
        assert self.name is not None


endogenous_registry = Registry(Endogenous)


@endogenous_registry.add
class Labor(Endogenous):
    """Labor accumulated through working. Included in all environments by default."""

    name = "Labor"


@endogenous_registry.add
class NewData(Endogenous):
    """The new data or lack thereof that a miner gets to store in the current timestep"""

    name = "NewData"

@endogenous_registry.add
class TotalData(Endogenous):
    """The total data capacity a miner is storing"""

    name = "TotalData"

@endogenous_registry.add
class EnergyPrice(Endogenous):
    """The electricity costs a miner has to pay per kWh"""

    name = "EnergyPrice"

@endogenous_registry.add
class RECsPrice(Endogenous):
    """The cost of the RECs a miner has decided to purchase"""

    name = "RECsPrice"

@endogenous_registry.add
class GreenScore(Endogenous):
    """The current green score of a miner, calculated from their scores the last 24h"""

    name = "GreenScore"

@endogenous_registry.add
class GreenScoresLastDay(Endogenous):
    """The green scores of a miner from the past 24h"""

    name = "GreenScoresLastDay"

@endogenous_registry.add
class InitialScore(Endogenous):
    """The initial green score of a miner, based on the energy mix at their location"""

    name = "InitialGreenScore"

@endogenous_registry.add
class ReliabilityScore(Endogenous):
    """The reliability score of a miner"""

    name = "ReliabilityScore"

@endogenous_registry.add
class TotalScore(Endogenous):
    """The score of a miner which determines the miner's likelihood of getting to store new data"""

    name = "TotalScore"

@endogenous_registry.add
class ConsumedEnergy(Endogenous):
    """The energy amount a miner has consumed in the current timestep"""

    name = "ConsumedEnergy"

