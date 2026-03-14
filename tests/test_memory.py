"""
tests/test_memory.py
--------------------
Tests for memory/memory.py (RunMemory, topic_similarity) and
memory/cache.py (PromptCache).

Covers:
  - topic_similarity: identical, similar, unrelated topics
  - RunMemory: start_run, finish_run, log_iteration, find_similar
  - RunMemory: recent_runs ordering, get_run, get_metrics, stats
  - RunMemory: find_similar threshold and max_age filtering
  - PromptCache: exact get/set, miss on unknown prompt
  - PromptCache: fuzzy get with threshold
  - PromptCache: invalidate, clear, stats
  - PromptCache: model filtering
  - Demo seeder: runs without error and produces expected rows
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.memory import RunMemory, topic_similarity
from memory.cache import PromptCache


# ── topic_similarity ──────────────────────────────────────────────────────────

class TestTopicSimilarity(unittest.TestCase):

    def test_identical_topics(self):
        s = topic_similarity("Claude tool use", "Claude tool use")
        self.assertAlmostEqual(s, 1.0, places=2)

    def test_clearly_similar(self):
        s = topic_similarity(
            "Building SKILL.md files for Claude",
            "How to write SKILL.md files for Claude Code",
        )
        self.assertGreater(s, 0.5)

    def test_unrelated_topics(self):
        s = topic_similarity("LoRA fine-tuning", "French cooking recipes")
        self.assertLess(s, 0.3)

    def test_symmetric(self):
        a = "Claude skills optimization"
        b = "Optimization of Claude skill files"
        self.assertAlmostEqual(topic_similarity(a, b), topic_similarity(b, a), places=5)

    def test_empty_string(self):
        s = topic_similarity("", "Claude tool use")
        self.assertEqual(s, 0.0)

    def test_both_empty(self):
        s = topic_similarity("", "")
        self.assertEqual(s, 0.0)

    def test_returns_float_between_0_and_1(self):
        s = topic_similarity("anything", "something")
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


# ── RunMemory ─────────────────────────────────────────────────────────────────

class TestRunMemory(unittest.TestCase):

    def _mem(self) -> RunMemory:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        return RunMemory(tmp.name)

    def test_start_run_returns_int(self):
        with self._mem() as mem:
            run_id = mem.start_run("Claude tool use")
            self.assertIsInstance(run_id, int)
            self.assertGreater(run_id, 0)

    def test_start_run_sets_status_running(self):
        with self._mem() as mem:
            run_id = mem.start_run("topic")
            run = mem.get_run(run_id)
            self.assertEqual(run["status"], "running")

    def test_finish_run_updates_fields(self):
        with self._mem() as mem:
            run_id = mem.start_run("Claude skills")
            mem.finish_run(
                run_id,
                status="completed",
                iterations=3,
                avg_score=0.82,
                pass_rate=0.75,
                weighted_eval=4.1,
                corpus_chars=50000,
                n_suggestions=5,
            )
            run = mem.get_run(run_id)
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["iterations"], 3)
            self.assertAlmostEqual(run["avg_score"], 0.82)

    def test_log_iteration_stores_metrics(self):
        with self._mem() as mem:
            run_id = mem.start_run("topic")
            mem.log_iteration(run_id, 1, val_score=0.72, judge_avg_score=3.5)
            mem.log_iteration(run_id, 2, val_score=0.78, judge_avg_score=3.8)
            metrics = mem.get_metrics(run_id)
            self.assertEqual(len(metrics), 2)
            self.assertEqual(metrics[0]["iteration"], 1)
            self.assertAlmostEqual(metrics[1]["val_score"], 0.78)

    def test_recent_runs_ordered_newest_first(self):
        with self._mem() as mem:
            id1 = mem.start_run("first topic")
            id2 = mem.start_run("second topic")
            runs = mem.recent_runs()
            self.assertEqual(runs[0]["id"], id2)
            self.assertEqual(runs[1]["id"], id1)

    def test_recent_runs_respects_limit(self):
        with self._mem() as mem:
            for i in range(10):
                mem.start_run(f"topic {i}")
            runs = mem.recent_runs(limit=3)
            self.assertEqual(len(runs), 3)

    def test_get_run_returns_none_for_missing(self):
        with self._mem() as mem:
            self.assertIsNone(mem.get_run(9999))

    def test_stats_counts_correctly(self):
        with self._mem() as mem:
            id1 = mem.start_run("topic a")
            id2 = mem.start_run("topic b")
            mem.finish_run(id1, status="completed", avg_score=0.8)
            mem.finish_run(id2, status="failed")
            stats = mem.stats()
            self.assertEqual(stats["total_runs"], 2)
            self.assertEqual(stats["completed"], 1)
            self.assertEqual(stats["failed"], 1)

    def test_find_similar_returns_similar_runs(self):
        with self._mem() as mem:
            id1 = mem.start_run("Claude SKILL.md file creation")
            mem.finish_run(id1, status="completed", avg_score=0.85)

            similar = mem.find_similar("Building SKILL.md files for Claude", threshold=0.3)
            self.assertGreater(len(similar), 0)
            self.assertIn("similarity", similar[0])

    def test_find_similar_respects_threshold(self):
        with self._mem() as mem:
            id1 = mem.start_run("Claude tool use")
            mem.finish_run(id1, status="completed")

            # Totally unrelated topic — should not match at high threshold
            similar = mem.find_similar("French cooking", threshold=0.8)
            self.assertEqual(len(similar), 0)

    def test_find_similar_only_returns_completed(self):
        with self._mem() as mem:
            # running run — should not appear
            mem.start_run("Claude SKILL.md")

            similar = mem.find_similar("Claude SKILL.md", threshold=0.5, status="completed")
            self.assertEqual(len(similar), 0)

    def test_find_similar_sorted_by_similarity_desc(self):
        with self._mem() as mem:
            id1 = mem.start_run("Claude tool use patterns")
            id2 = mem.start_run("Claude tool registration and use")
            mem.finish_run(id1, status="completed")
            mem.finish_run(id2, status="completed")

            similar = mem.find_similar("Claude tool use", threshold=0.1)
            if len(similar) >= 2:
                self.assertGreaterEqual(similar[0]["similarity"], similar[1]["similarity"])

    def test_all_topics_returns_distinct(self):
        with self._mem() as mem:
            id1 = mem.start_run("topic A")
            id2 = mem.start_run("topic A")  # duplicate
            id3 = mem.start_run("topic B")
            mem.finish_run(id1, status="completed")
            mem.finish_run(id2, status="completed")
            mem.finish_run(id3, status="completed")
            topics = mem.all_topics()
            self.assertEqual(len(topics), 2)


# ── PromptCache ───────────────────────────────────────────────────────────────

class TestPromptCache(unittest.TestCase):

    def _cache(self) -> PromptCache:
        tmp = tempfile.mkdtemp()
        return PromptCache(tmp)

    def test_set_and_get_exact(self):
        cache = self._cache()
        cache.set("What is LoRA?", "LoRA is low-rank adaptation.", model="ollama:llama3.2")
        hit = cache.get("What is LoRA?", model="ollama:llama3.2")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["response"], "LoRA is low-rank adaptation.")

    def test_miss_returns_none(self):
        cache = self._cache()
        self.assertIsNone(cache.get("This prompt was never stored"))

    def test_set_updates_existing(self):
        cache = self._cache()
        cache.set("What is Claude?", "An AI.", model="m")
        cache.set("What is Claude?", "An AI assistant by Anthropic.", model="m")
        hit = cache.get("What is Claude?", model="m")
        self.assertEqual(hit["response"], "An AI assistant by Anthropic.")

    def test_hit_increments_counter(self):
        cache = self._cache()
        cache.set("prompt", "response")
        cache.get("prompt")
        cache.get("prompt")
        hit = cache.get("prompt")
        self.assertGreaterEqual(hit["hits"], 2)

    def test_model_filter_on_get(self):
        cache = self._cache()
        cache.set("prompt", "resp-a", model="model-a")
        cache.set("prompt", "resp-b", model="model-b")
        hit_a = cache.get("prompt", model="model-a")
        hit_b = cache.get("prompt", model="model-b")
        self.assertEqual(hit_a["response"], "resp-a")
        self.assertEqual(hit_b["response"], "resp-b")

    def test_get_fuzzy_finds_similar(self):
        cache = self._cache()
        cache.set("What is LoRA fine-tuning?", "LoRA is low-rank adaptation.")
        hit = cache.get_fuzzy("Explain LoRA fine-tuning for LLMs", threshold=0.3)
        self.assertIsNotNone(hit)
        self.assertIn("_fuzzy_similarity", hit)

    def test_get_fuzzy_misses_on_unrelated(self):
        cache = self._cache()
        cache.set("What is LoRA?", "LoRA is low-rank adaptation.")
        hit = cache.get_fuzzy("French cooking recipes", threshold=0.8)
        self.assertIsNone(hit)

    def test_invalidate_removes_entry(self):
        cache = self._cache()
        cache.set("remove me", "value")
        removed = cache.invalidate("remove me")
        self.assertTrue(removed)
        self.assertIsNone(cache.get("remove me"))

    def test_invalidate_returns_false_for_missing(self):
        cache = self._cache()
        self.assertFalse(cache.invalidate("never stored"))

    def test_clear_removes_all(self):
        cache = self._cache()
        cache.set("p1", "r1")
        cache.set("p2", "r2")
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_stats_returns_correct_counts(self):
        cache = self._cache()
        cache.set("p1", "r1", model="m1")
        cache.set("p2", "r2", model="m2")
        stats = cache.stats()
        self.assertEqual(stats["total_entries"], 2)
        self.assertIn("m1", stats["models"])

    def test_len_reflects_entries(self):
        cache = self._cache()
        self.assertEqual(len(cache), 0)
        cache.set("a", "b")
        self.assertEqual(len(cache), 1)

    def test_persists_across_instances(self):
        tmp = tempfile.mkdtemp()
        c1 = PromptCache(tmp)
        c1.set("persistent prompt", "stored value")

        c2 = PromptCache(tmp)
        hit = c2.get("persistent prompt")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["response"], "stored value")


# ── Demo seeder ───────────────────────────────────────────────────────────────

class TestDemoSeeder(unittest.TestCase):

    def test_seeder_runs_and_produces_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        from dashboard.seed_demo import seed
        seed(db_path)

        mem = RunMemory(db_path)
        runs = mem.recent_runs(limit=100)
        self.assertGreaterEqual(len(runs), 5)

        completed = [r for r in runs if r["status"] == "completed"]
        self.assertGreater(len(completed), 0)

        # Every completed run should have at least one iteration metric
        for r in completed:
            if r.get("iterations") and r["iterations"] > 0:
                metrics = mem.get_metrics(r["id"])
                self.assertGreater(len(metrics), 0)

        mem.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
