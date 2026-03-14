import os
import sys
import argparse
from pathlib import Path
from collector.ultimate_collector import Document, PDFCollector

def test_document_cleaning():
    """Verify that Document class cleans text correctly."""
    raw_text = "Hello\r\nWorld \t with  whitespace.\n\n\n\nToo many newlines."
    doc = Document(title="Test", text=raw_text, source="local")
    
    # Check that \r\n is normalized to \n
    assert "\r\n" not in doc.text
    # Check that multiple newlines are collapsed (max 3)
    assert "\n\n\n\n" not in doc.text
    # Check that multiple spaces/tabs are collapsed
    assert "  " not in doc.text
    assert "\t" not in doc.text
    
    print("✅ Document cleaning test passed")

def test_pdf_extraction(pdf_path: str):
    """Manually verify PDF extraction for a specific file."""
    path = Path(pdf_path)
    if not path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return

    collector = PDFCollector([path])
    docs = collector.collect()
    
    if not docs:
        print(f"❌ Error: No documents extracted from {pdf_path}")
        return
    
    doc = docs[0]
    print(f"\n--- Extraction Result for: {doc.title} ---")
    print(f"Source: {doc.source}")
    print(f"URL: {doc.url}")
    print(f"Character Count: {doc.chars}")
    print("\n--- Preview (first 500 chars) ---")
    print(doc.text[:500] + "..." if len(doc.text) > 500 else doc.text)
    print("\n✅ PDF extraction test complete")

if __name__ == "__main__":
    # If run via pytest, we don't want the CLI logic
    if "pytest" in sys.modules:
        pass
    else:
        parser = argparse.ArgumentParser(description="Collector Test Utility")
        parser.add_argument("--pdf", help="Path to a PDF file to test extraction")
        args = parser.parse_args()

        # Run core unit tests always
        test_document_cleaning()

        if args.pdf:
            test_pdf_extraction(args.pdf)
        else:
            print("\nTip: Run with --pdf <path> to test specific PDF extraction.")
