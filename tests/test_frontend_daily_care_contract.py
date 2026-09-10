"""Focused regression checks for the frontend Daily Care data boundary."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "OmniCare-FE" / "src"


class FrontendDailyCareContractTest(unittest.TestCase):
    def test_elder_navigation_uses_backend_identity_not_mock_placeholder(self):
        source = (FRONTEND / "pages" / "ElderManagement.tsx").read_text(encoding="utf-8")
        self.assertIn("queryFn: elderService.getAll", source)
        self.assertIn("queryKey: ['customer-elders']", source)
        self.assertNotIn("placeholderData: mockElders", source)
        self.assertIn("encodeURIComponent(elder.id)", source)

        fixtures = "\n".join(
            (FRONTEND / path).read_text(encoding="utf-8")
            for path in ("services/mockData.ts", "pages/EldersPage.tsx", "pages/FamilyManagement.tsx")
        )
        self.assertNotIn("elderId: '1'", fixtures)
        self.assertNotIn("elderlyPersonId: '1'", fixtures)

        login = (FRONTEND / "pages" / "LoginPage.tsx").read_text(encoding="utf-8")
        self.assertIn("useState('admin@omnicare.local')", login)
        self.assertNotIn("useState('admin@omnicare.ai')", login)

    def test_daily_care_client_keeps_backend_endpoint_and_utc_range(self):
        service = (FRONTEND / "services" / "index.ts").read_text(encoding="utf-8")
        page = (FRONTEND / "pages" / "DailyCarePage.tsx").read_text(encoding="utf-8")
        self.assertIn("api.get<ElderDetailDto[]>('/customers/me/elderly')", service)
        self.assertNotIn("elderService.list({ pageSize: 50 })).data", service)
        self.assertIn("/customers/me/elderly/${encodeURIComponent(elderlyPersonId)}/daily-care", service)
        self.assertIn("/customers/me/elderly/${encodeURIComponent(elderlyPersonId)}/daily-care/summary", service)
        self.assertIn("{ params: { from, to } }", service)
        self.assertIn("from: from.toISOString(), to: to.toISOString()", page)
        self.assertIn("staleTime: 0", page)
        self.assertIn("gcTime: 0", page)
        self.assertIn("refetchOnMount: 'always'", page)
        self.assertIn("queryKey: ['daily-care-summary', elderId, range?.from, range?.to]", page)
        self.assertIn("summaryQuery.data.summary", page)
        self.assertIn("summaryQuery.data?.fallback", page)


if __name__ == "__main__":
    unittest.main()
