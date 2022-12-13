"""Base Scenario Handling."""

from dataclasses import dataclass, frozen
import abc


class Simulation(dataclass):
    pass


class BaseScenarioBuilder(abc.ABC):
    """Base scenario builder object."""


class CustomScenarioBuilder(BaseScenarioBuilder):
    pass
