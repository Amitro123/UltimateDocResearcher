import unittest
import json
import os
import sys
from pathlib import Path
import importlib

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import trigger_kaggle using importlib due to hyphen in directory name
trigger_kaggle = importlib.import_module("api-triggers.trigger_kaggle")

class TestSecurityFix(unittest.TestCase):
    def test_generate_kernel_notebook_escaping(self):
        topic = 'some topic" \nimport os; os.system("echo VULNERABLE_TOPIC"); #'
        github_repo = 'some/repo" \nimport os; os.system("echo VULNERABLE_REPO"); #'
        n_iterations = "10; import os; os.system('echo VULNERABLE_ITERATIONS')"
        output_path = "test_notebook_security.ipynb"

        # This should not raise an error for n_iterations if it's correctly cast to int
        try:
            n_iterations_int = int(n_iterations.split(';')[0])
        except ValueError:
            n_iterations_int = 10

        trigger_kaggle.generate_kernel_notebook(topic, n_iterations_int, github_repo, output_path)

        try:
            with open(output_path, 'r') as f:
                notebook = json.load(f)

            # Check cell 0 (header comment)
            header_cell = notebook['cells'][0]['source']
            self.assertIn('# UltimateDocResearcher', header_cell)
            self.assertNotIn('\n', header_cell)
            self.assertIn('some topic"', header_cell)

            # Check cell 1 (repo assignment)
            repo_cell = notebook['cells'][1]['source']
            expected_repo_line = 'repo = "some/repo\\" \\nimport os; os.system(\\"echo VULNERABLE_REPO\\"); #"'
            self.assertIn(expected_repo_line, repo_cell)

            # Check cell 3 (TOPIC assignment)
            topic_cell = notebook['cells'][3]['source']
            expected_topic_line = 'TOPIC = "some topic\\" \\nimport os; os.system(\\"echo VULNERABLE_TOPIC\\"); #"'
            self.assertIn(expected_topic_line, topic_cell)
            self.assertIn('N_ITERATIONS = 10', topic_cell)

        finally:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                pass  # cleanup failure is non-fatal in test env

if __name__ == '__main__':
    unittest.main()
