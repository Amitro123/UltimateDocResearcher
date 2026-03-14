import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock missing dependencies to allow import
mock_modules = [
    'aiohttp',
    'aiofiles',
    'fitz',
    'googleapiclient',
    'googleapiclient.discovery',
    'google.oauth2.service_account',
    'google.oauth2.credentials',
    'bs4'
]
for module_name in mock_modules:
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

import unittest
from collector.ultimate_collector import Document, PDFCollector, UltimateCollector

class TestCollectorLogic(unittest.TestCase):
    def test_document_cleaning(self):
        raw_text = "Hello\r\nWorld \t with  whitespace.\n\n\n\nToo many newlines."
        doc = Document(title='Test', text=raw_text, source='local')

        self.assertNotIn("\r\n", doc.text)
        self.assertNotIn("\n\n\n\n", doc.text)
        self.assertNotIn("  ", doc.text)
        self.assertNotIn("\t", doc.text)

    def test_document_id(self):
        doc1 = Document(title='A', text='Content', source='web', url='http://x.com')
        doc2 = Document(title='A', text='Content', source='web', url='http://x.com')
        doc3 = Document(title='B', text='Content', source='web', url='http://x.com')

        self.assertEqual(doc1.doc_id, doc2.doc_id)
        self.assertNotEqual(doc1.doc_id, doc3.doc_id)

    def test_drive_extractor_runtime_error_graceful_degradation(self):
        """Verify that UltimateCollector handles DriveExtractor init failure gracefully."""
        with patch('collector.ultimate_collector.HAS_GDRIVE', True), \
             patch('collector.ultimate_collector.DriveExtractor') as mock_drive:

            # Setup mock to raise RuntimeError on initialization
            mock_drive.side_effect = RuntimeError('Drive API error')

            # Initialize UltimateCollector with drive folder IDs
            collector = UltimateCollector(
                drive_folder_ids=['test_folder_id'],
                output_dir='test_data/'
            )

            # Assert that drive_extractor is None due to graceful degradation
            self.assertIsNone(collector.drive_extractor)

            # Ensure it didn't crash and other attributes are set
            self.assertEqual(collector.output_dir, Path('test_data/'))

if __name__ == '__main__':
    unittest.main()
