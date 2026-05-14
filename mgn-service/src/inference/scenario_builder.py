"""Scenario builder compatibility module.

In the commercial version, scenario metadata lives in datasets/<dataset_id>/scenario.yaml.
This module is kept for extension points and UI imports.
"""
from src.data.scenario import default_scenario, merged_scenario, scenario_feature_vector, source_center_radius_temperature
