import sys
from pathlib import Path
import pytest

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collector.ultimate_collector import Document, PDFCollector

def test_document_cleaning():
    raw_text = "Hello\r\nWorld \t with  whitespace.\n\n\n\nToo many newlines."
    doc = Document(title="Test", text=raw_text, source="local")
    
    assert "\r\n" not in doc.text
    assert "\n\n\n\n" not in doc.text
    assert "  " not in doc.text
    assert "\t" not in doc.text

def test_document_id():
    doc1 = Document(title="A", text="Content", source="web", url="http://x.com")
    doc2 = Document(title="A", text="Content", source="web", url="http://x.com")
    doc3 = Document(title="B", text="Content", source="web", url="http://x.com")
    
    assert doc1.doc_id == doc2.doc_id
    assert doc1.doc_id != doc3.doc_id

def test_collector_filter():
    # Placeholder for collector filtering logic if we extract it to a method
    pass
