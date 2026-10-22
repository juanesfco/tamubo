import unittest
import numpy as np

from autoia import autoia, Interval, ConfigurationPolicy, AutoIAConfiguredResult


class TestAutomaticConfigurationController(unittest.TestCase):
    def test_fixed_policy_preserves_single_run_endpoints(self):
        def model(p):
            x = p[0]
            return np.exp(x) + x*x

        box = [Interval(-0.5, 0.7)]
        a = autoia(model, box, order=4, representation="polynomial")
        b = autoia(
            model, box, order=4, representation="polynomial",
            configuration_policy=ConfigurationPolicy(mode="fixed"),
        )
        self.assertTrue(np.array_equal(a.lower.view(np.uint64), b.lower.view(np.uint64)))
        self.assertTrue(np.array_equal(a.upper.view(np.uint64), b.upper.view(np.uint64)))

    def test_reference_portfolio_intersects_independent_certificates(self):
        def model(p):
            x = p[0]
            y = x*x
            return np.array([y, y-y], dtype=object)

        result = autoia(
            model, [Interval(-1.0, 1.0)], order=1,
            configuration_policy=ConfigurationPolicy(
                mode="reference", extra_orders=(), extra_representations=("generator",), max_attempts=2
            ),
        )
        self.assertIsInstance(result, AutoIAConfiguredResult)
        self.assertEqual(result.lower[1], 0.0)
        self.assertEqual(result.upper[1], 0.0)
        self.assertTrue(result.machine_rigorous)
        self.assertEqual(len(result.attempt_results), 2)
        self.assertEqual(result.configuration_diagnostics["nondominated_attempt_indices"], (0, 1))

    def test_reference_portfolio_uses_higher_order_when_it_tightens(self):
        def model(p):
            return np.exp(p[0])

        baseline = autoia(model, [Interval(-1.0, 1.0)], order=4)
        result = autoia(
            model, [Interval(-1.0, 1.0)], order=4,
            configuration_policy=ConfigurationPolicy(
                mode="reference", extra_orders=(5,), extra_representations=(), max_attempts=2
            ),
        )
        self.assertLess(result.width[0], baseline.width[0])
        self.assertGreaterEqual(result.lower[0], baseline.lower[0])
        self.assertLessEqual(result.upper[0], baseline.upper[0])

    def test_failed_optional_attempt_does_not_destroy_valid_baseline(self):
        def model(p):
            # Interval representation cannot evaluate all polynomial semantics as
            # sharply, but the baseline polynomial run is independently valid.
            return p[0] * p[0] + 1.0

        result = autoia(
            model, [Interval(-1.0, 1.0)], order=2,
            configuration_policy=ConfigurationPolicy(
                extra_orders=(), extra_representations=("interval",), max_attempts=2
            ),
        )
        self.assertTrue(result.machine_rigorous)
        self.assertLessEqual(result.lower[0], 1.0)
        self.assertGreaterEqual(result.upper[0], 2.0)


    def test_auto_policy_skips_unjustified_retry(self):
        def model(p):
            return np.exp(p[0])

        result = autoia(
            model, [Interval(-1.0, 1.0)],
            configuration_policy=ConfigurationPolicy(mode="auto"),
        )
        attempts = result.configuration_diagnostics["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["reason"], "baseline")

    def test_auto_policy_uses_higher_order_when_remainder_dominates(self):
        def model(p):
            return np.exp(3.0*p[0])

        result = autoia(
            model, [Interval(-1.0, 1.0)],
            configuration_policy=ConfigurationPolicy(mode="auto"),
        )
        attempts = result.configuration_diagnostics["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[1]["representation"], "polynomial")
        self.assertEqual(attempts[1]["order"], 5)
        self.assertEqual(attempts[1]["reason"], "remainder-dominates-retained-variation")

    def test_auto_policy_generator_trigger_uses_recombination_signal(self):
        def model(p):
            x = p[0]
            y = np.exp(x)
            return y-y

        result = autoia(
            model, [Interval(-1.0, 1.0)],
            configuration_policy=ConfigurationPolicy(mode="auto"),
        )
        attempts = result.configuration_diagnostics["attempts"]
        self.assertEqual(attempts[1]["representation"], "generator")
        self.assertEqual(attempts[1]["reason"], "shared-error-recombination")
        self.assertEqual(result.lower[0], 0.0)
        self.assertEqual(result.upper[0], 0.0)

    def test_prepared_auto_configuration_lazily_reuses_candidate_sessions(self):
        from autoia import prepare, PreparedConfiguredAutoIA

        def model(p):
            x = p[0]
            y = x*x
            return np.array([y, y-y], dtype=object)

        session = prepare(
            model, order=1,
            configuration_policy=ConfigurationPolicy(mode="auto"),
        )
        self.assertIsInstance(session, PreparedConfiguredAutoIA)
        self.assertEqual(session.stats.candidate_sessions, 1)
        first = session.certify([Interval(-1.0, 1.0)])
        self.assertEqual(first.width[1], 0.0)
        self.assertEqual(session.stats.candidate_sessions, 3)
        executions = session.stats.engineering_model_executions
        second = session.certify([Interval(-1.0, 1.0)])
        self.assertEqual(second.width[1], 0.0)
        self.assertEqual(session.stats.engineering_model_executions, executions)


    def test_zero_knob_public_call_is_automatic(self):
        def model(p):
            return np.exp(3.0*p[0])

        result = autoia(model, [Interval(-1.0, 1.0)])
        self.assertIsInstance(result, AutoIAConfiguredResult)
        self.assertTrue(result.automatic_configuration)
        self.assertEqual(result.configuration_policy.mode, "auto")
        self.assertEqual(result.configuration_diagnostics["baseline_order"], 4)
        self.assertEqual(result.configuration_diagnostics["baseline_representation"], "polynomial")
        self.assertIn("configuration", repr(result).lower())
        self.assertIn("Attempt 1", result.explain_configuration())

    def test_zero_knob_branching_preserves_pathwise_metadata(self):
        def model(p):
            x = p[0]
            if x >= 0.0:
                return x*x + x
            return x*x - 2.0*x

        result = autoia(model, [Interval(-1.0, 1.0)])
        self.assertIsInstance(result, AutoIAConfiguredResult)
        self.assertEqual(len(result.branch_results), 2)
        self.assertEqual(len(result.branch_boxes), 2)
        self.assertEqual(len(result.branch_decisions), 2)
        self.assertIn("execution branch", repr(result))

    def test_explicit_order_is_fixed_expert_override(self):
        from autoia import AutoIAResult
        def model(p):
            return np.exp(3.0*p[0])

        result = autoia(model, [Interval(-1.0, 1.0)], order=4)
        self.assertIsInstance(result, AutoIAResult)
        self.assertNotIsInstance(result, AutoIAConfiguredResult)
        self.assertEqual(result.requested_representation, "polynomial")

    def test_explicit_representation_is_fixed_expert_override(self):
        from autoia import AutoIAResult
        def model(p):
            return p[0]*p[0]

        result = autoia(model, [Interval(-1.0, 1.0)], representation="generator")
        self.assertIsInstance(result, AutoIAResult)
        self.assertNotIsInstance(result, AutoIAConfiguredResult)
        self.assertEqual(result.requested_representation, "generator")

    def test_explicit_auto_policy_can_use_expert_baseline(self):
        def model(p):
            return np.exp(3.0*p[0])

        result = autoia(
            model, [Interval(-1.0, 1.0)], order=3,
            configuration_policy=ConfigurationPolicy(mode="auto"),
        )
        self.assertIsInstance(result, AutoIAConfiguredResult)
        self.assertEqual(result.configuration_diagnostics["baseline_order"], 3)

    def test_configuration_policy_constructor_defaults_to_auto(self):
        self.assertEqual(ConfigurationPolicy().mode, "auto")

    def test_prepare_zero_knob_is_automatic_but_explicit_order_is_fixed(self):
        from autoia import prepare, PreparedAutoIA, PreparedConfiguredAutoIA
        def model(p):
            return np.exp(p[0])

        automatic = prepare(model)
        fixed = prepare(model, order=4)
        self.assertIsInstance(automatic, PreparedConfiguredAutoIA)
        self.assertIsInstance(fixed, PreparedAutoIA)
        r = automatic.certify([Interval(-0.5, 0.5)])
        self.assertIsInstance(r, AutoIAConfiguredResult)

    def test_policy_validation_is_typed(self):
        with self.assertRaises(Exception) as ctx:
            ConfigurationPolicy(mode="mystery")
        self.assertEqual(type(ctx.exception).__name__, "InvalidInputError")


if __name__ == "__main__":
    unittest.main()
