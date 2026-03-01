import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure we can import from the current directory
sys.path.append('.')

from config import settings
from services.draft_service import generate_notice_reply

class TestRAGGrounding(unittest.TestCase):
    def setUp(self):
        print(f"\n⚙️  CONFIG CHECK:")
        print(f"   Model:  Gemini 2.5 Flash")

    @patch('services.draft_service.retrieve_relevant_law')
    def test_grounded_drafting(self, mock_retrieve):
        # 1. Mock a legal passage [Section 73]
        mock_retrieve.return_value = (
            "[Section 73: Determination of tax]\n"
            "Where it appears to the proper officer that any tax has not been paid or short paid, "
            "he shall serve notice on the person chargeable with tax to show cause why he should "
            "not pay the amount specified in the notice."
        )

        print("\n🧪 Testing GROUNDED Generation (Mocked Context)...")
        query = "What does Section 73 say about tax determination?"
        response = generate_notice_reply(query)

        print(f"Draft Reply: {response.draft_reply[:150]}...")
        print(f"Citations: {response.citations}")
        print(f"Is Grounded: {response.is_grounded}")

        # Verification
        self.assertTrue(response.is_grounded)
        self.assertIn('Section 73', response.citations)
        self.assertNotIn('Insufficient information', response.draft_reply)

    @patch('services.draft_service.retrieve_relevant_law')
    def test_safe_failure(self, mock_retrieve):
        # 2. Mock empty context (Safe Failure)
        mock_retrieve.return_value = ""

        print("\n🧪 Testing SAFE FAILURE (Empty Context)...")
        query = "How to make a pizza?"
        response = generate_notice_reply(query)

        print(f"Draft Reply: {response.draft_reply}")
        print(f"Citations: {response.citations}")
        print(f"Is Grounded: {response.is_grounded}")

        # Verification
        self.assertEqual(response.draft_reply, "Insufficient information in current legal corpus.")
        self.assertFalse(response.is_grounded)

if __name__ == "__main__":
    print("🚀 Running Light-Touch Verification (Zero Vector Load)...")
    unittest.main()
