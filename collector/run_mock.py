import sys
import argparse
from pathlib import Path

# Add project root to sys.path to allow running this script directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collector.ultimate_collector import UltimateCollector

def run_mock_research(topic: str, output_dir: str = "data/"):
    """
    Run a mock research collection for a given topic.
    """
    print(f"🚀 Starting mock research for topic: '{topic}'")
    
    # Simple query generation based on topic
    queries = [
        f"{topic} best practices",
        f"{topic} architecture patterns",
        f"advanced {topic} techniques"
    ]
    
    print(f"🔍 Generated search queries: {queries}")
    
    # If the topic relates to 'resume', ensure we include the local PDF
    pdf_paths = []
    if "resume" in topic.lower():
        pdf_path = Path("C:/Users/Dana/Downloads/Amit-Rosen.pdf")
        if pdf_path.exists():
            print(f"📎 Including local PDF in research: {pdf_path.name}")
            pdf_paths.append(str(pdf_path))

    collector = UltimateCollector(
        pdf_paths=pdf_paths,
        google_queries=queries,
        output_dir=output_dir,
        min_chars=100  # Lower threshold for testing
    )

    docs = collector.run()
    
    print(f"\n✅ Mock run complete!")
    print(f"📊 Total documents collected: {len(docs)}")
    print(f"📂 Results saved to: {Path(output_dir).resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UltimateDocResearcher Mock Run")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    args = parser.parse_args()

    run_mock_research(args.topic, args.output_dir)
