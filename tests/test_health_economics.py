"""Unit tests for Health Economics & Health Technology Assessment (HTA / NICE) Engine.
"""

import unittest
import math
from src.health_economics.hta_engine import (
    calculate_icer_model,
    generate_icer_dataset,
    calculate_nmb_model,
    generate_nmb_dataset,
    calculate_markov_model,
    generate_markov_dataset,
    calculate_psa_model,
    generate_psa_dataset,
    calculate_bia_model,
    generate_bia_dataset
)
from src.health_economics.hta_schema import HTA_METHODS, HTA_SCENARIOS


class TestHealthEconomicsEngine(unittest.TestCase):
    """Verifies mathematical correctness of HTA calculation models."""

    def test_icer_calculation_tradeoff_ne(self):
        """Tests standard North-East quadrant cost-effectiveness calculation."""
        res = calculate_icer_model(
            cost_intervention=28400.0,
            cost_comparator=14200.0,
            qaly_intervention=6.85,
            qaly_comparator=6.15,
            wtp_threshold=25000.0
        )
        self.assertEqual(res["delta_cost"], 14200.0)
        self.assertAlmostEqual(res["delta_qaly"], 0.70, places=4)
        expected_icer = 14200.0 / 0.70
        self.assertAlmostEqual(res["icer"], expected_icer, places=2)
        self.assertEqual(res["quadrant_code"], "NE")
        self.assertTrue(res["is_cost_effective"])  # ICER ~ £20,285 <= £25,000
        # NMB = 25000 * 0.70 - 14200 = 17500 - 14200 = 3300
        self.assertAlmostEqual(res["nmb"], 3300.0, places=2)

    def test_icer_dominance_cases(self):
        """Tests dominant (SE) and dominated (NW) quadrants."""
        # Dominant: more effective and cheaper
        dom = calculate_icer_model(cost_intervention=10000.0, cost_comparator=15000.0, qaly_intervention=5.0, qaly_comparator=4.0)
        self.assertEqual(dom["quadrant_code"], "SE")
        self.assertTrue(dom["is_cost_effective"])
        self.assertIn("Dominant", dom["verdict"])

        # Dominated: less effective and more expensive
        dom_ed = calculate_icer_model(cost_intervention=20000.0, cost_comparator=15000.0, qaly_intervention=3.5, qaly_comparator=4.0)
        self.assertEqual(dom_ed["quadrant_code"], "NW")
        self.assertFalse(dom_ed["is_cost_effective"])
        self.assertIn("Dominated", dom_ed["verdict"])

    def test_nmb_multi_arm_ranking(self):
        """Verifies multi-arm NMB ranking identifies highest net benefit strategy."""
        treatments = [
            {"name": "Standard Care", "cost": 8500.0, "qalys": 4.20},
            {"name": "ARNI", "cost": 15800.0, "qalys": 4.62},
            {"name": "SGLT2i", "cost": 19400.0, "qalys": 4.88}
        ]
        res = calculate_nmb_model(treatments, wtp_threshold=30000.0)
        # SGLT2i: 30000 * 4.88 - 19400 = 146400 - 19400 = 127000
        # ARNI:   30000 * 4.62 - 15800 = 138600 - 15800 = 122800
        # Standard: 30000 * 4.20 - 8500 = 126000 - 8500 = 117500
        self.assertEqual(res["optimal_treatment"], "SGLT2i")
        self.assertAlmostEqual(res["ranked_by_nmb"][0]["nmb"], 127000.0, places=1)
        self.assertEqual(res["ranked_by_nmb"][1]["name"], "ARNI")
        self.assertEqual(res["ranked_by_nmb"][2]["name"], "Standard Care")

    def test_markov_cohort_simulation_discounting(self):
        """Verifies Markov state transitions, half-cycle correction, and 3.5% discounting."""
        res = calculate_markov_model(cycles=5, cycle_years=1.0, discount_rate=0.035)
        self.assertEqual(len(res["chemo_records"]), 5)
        self.assertEqual(len(res["immuno_records"]), 5)
        # Immuno should yield higher discounted QALYs than chemo
        self.assertGreater(res["immuno_disc_qaly"], res["chemo_disc_qaly"])
        # Immuno should cost more discounted
        self.assertGreater(res["immuno_disc_cost"], res["chemo_disc_cost"])
        self.assertGreater(res["icer"], 0.0)

    def test_psa_ceac_simulation(self):
        """Verifies PSA Monte Carlo simulation output dimensions and CEAC probabilities."""
        res = calculate_psa_model(iterations=200, seed=123)
        self.assertEqual(res["iterations"], 200)
        self.assertGreaterEqual(res["prob_ce_at_wtp"], 0.0)
        self.assertLessEqual(res["prob_ce_at_wtp"], 1.0)
        self.assertTrue(len(res["ceac_curve"]) > 5)
        # CEAC curve should generally increase with threshold
        prob_first = res["ceac_curve"][0]["prob_ce"]
        prob_last = res["ceac_curve"][-1]["prob_ce"]
        self.assertGreaterEqual(prob_last, prob_first)

    def test_bia_5yr_financial_forecasting(self):
        """Verifies 5-year budget impact totals and net financial change."""
        res = calculate_bia_model(eligible_population=100000)
        self.assertEqual(len(res["yearly_results"]), 5)
        # Uptake increases over years, so net budget impact should increase
        yr1_impact = res["yearly_results"][0]["net_budget_impact"]
        yr5_impact = res["yearly_results"][4]["net_budget_impact"]
        self.assertGreater(yr5_impact, yr1_impact)
        self.assertAlmostEqual(
            res["cumulative_5yr_net_budget_impact"],
            sum(y["net_budget_impact"] for y in res["yearly_results"]),
            places=2
        )


if __name__ == "__main__":
    unittest.main()
