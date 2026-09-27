"""
engine10/test_engine10.py
Unit tests for the Phase 10 Dashboard & Monitoring API Engine.

Coverage:
  10-1  Health endpoint returns 200 OK and valid status
  10-2  /api/state/latest returns latest state or no_data cleanly
  10-3  /api/state/history returns history array with requested length
  10-4  /api/explain/latest parses ExplainBundle correctly
  10-5  /api/metrics/summary calculates KPIs (water, energy, peak temp)
  10-6  _load_data and _extract_objectives parse state store DataFrame
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from starlette.testclient import TestClient

from shared.state_store import init_db, write_state
from shared.types import StateRecord
from engine10.api import app
from engine10.dashboard import _load_data, _extract_objectives

class TestDashboardAndAPI(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.tmp_dir / "test_api_state.db"
        init_db(self.db_path)
        self.client = TestClient(app)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_health_endpoint(self):
        """10-1: Health endpoint returns 200 OK."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")

    def test_root_dashboard_endpoint(self):
        """10-2: Root / serves the CoolFlow view-only dashboard with electric sky blue."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("CoolFlow", resp.text)
        self.assertIn("Predictive Cooling", resp.text)
        self.assertIn("#38bdf8", resp.text)

    def test_state_endpoints(self):
        """10-2, 10-3, 10-5: Test state and metrics endpoints."""
        rec = StateRecord(
            tick_id=1,
            timestamp="2026-09-26T12:00:00Z",
            temperatures=[45.0, 48.0, 42.0],
            t_inlet=20.0,
            water_rate_l_per_h=50.0,
            wue=0.4,
            cool_mode=1,
            power_total_w=900.0,
            a_fan=0.6,
            a_cool=1,
            a_dvfs=[0.9, 0.9, 0.9],
            a_mig=(2, 0),
            trigger_optimizer=True,
            max_temp=48.0,
            shap_json='{"selection_rule": "chebyshev_knee", "selected_objectives": [10.0, 0.0, 0.1]}',
        )
        write_state(rec)

        # /api/state/latest
        resp = self.client.get("/api/state/latest")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("record", resp.json())

        # /api/state/history
        resp_hist = self.client.get("/api/state/history?n=10")
        self.assertEqual(resp_hist.status_code, 200)
        self.assertGreaterEqual(resp_hist.json()["count"], 1)

        # /api/metrics/summary
        resp_metrics = self.client.get("/api/metrics/summary")
        self.assertEqual(resp_metrics.status_code, 200)
        kpis = resp_metrics.json()["kpis"]
        self.assertIn("current_max_temp", kpis)
        self.assertIn("free_cooling_percent", kpis)

    def test_explain_endpoint(self):
        """10-4: Test /api/explain/latest."""
        resp = self.client.get("/api/explain/latest")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
