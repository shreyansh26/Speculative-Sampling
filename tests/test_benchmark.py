import io
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from benchmarks.benchmark import benchmark_experiment, benchmark_inputs, parse_args, profile_stages, summarize
from speculative_sampling import speculative_sampling
from tests.test_sampling import tiny_qwen


class BenchmarkTests(unittest.TestCase):
    def test_lookahead_argument(self):
        with patch('sys.argv', ['benchmark.py']):
            args = parse_args()
            self.assertEqual(args.lookahead, 3)
            self.assertFalse(args.cuda_graphs)
        with patch('sys.argv', ['benchmark.py', '--cuda-graphs']):
            self.assertTrue(parse_args().cuda_graphs)
        with patch('sys.argv', ['benchmark.py', '--lookahead', '4']):
            self.assertEqual(parse_args().lookahead, 4)
        for invalid in ('0', '-1'):
            with patch('sys.argv', ['benchmark.py', '--lookahead', invalid]), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_summary_uses_total_tokens_and_keeps_profile_separate(self):
        profile = {
            'wall_seconds': 12, 'calls': {'verify': 3},
            'accepted_draft_tokens_per_round': 2.0, 'committed_tokens_per_round': 2.5,
            'total_ms': dict.fromkeys(('prefill', 'draft', 'verify', 'append'), 10),
            'mean_ms_per_call': dict.fromkeys(('prefill', 'draft', 'verify', 'append'), 2),
            'outside_stages_ms': 11960,
        }
        row = summarize({'ar': 10, 'spec': 8}, 2, 8, 3, profile, 4, 0)
        self.assertEqual(row['speedup'], 1.25)
        self.assertEqual(row['ar_tokens_per_s'], 4.8)
        self.assertEqual(row['spec_tokens_per_s'], 6)
        self.assertEqual(row['spec_mean_e2e_ms'], 8000 / 6)
        self.assertEqual(row['profile_total_e2e_s'], 12)
        self.assertEqual(row['profile_avg_accepted_draft_tokens'], 2)
        self.assertEqual(row['profile_avg_committed_tokens'], 2.5)
        self.assertEqual(row['greedy_exact_matches'], 4)
        self.assertEqual(summarize({'ar': 10, 'spec': 8}, 2, 8, 3, profile, 0, 0.5)['greedy_exact_matches'], '')

    def test_long_prompt_arguments_and_construction(self):
        with patch('sys.argv', ['benchmark.py', '--prompt-lengths', '1024', '8192', '32768',
                                '--cuda-graphs', '--check-graphs']):
            args = parse_args()
        self.assertEqual(args.prompt_lengths, [1024, 8192, 32768])
        self.assertTrue(args.check_graphs)
        for flags in (['--prompt-lengths', '0'], ['--check-graphs'],
                      ['--cuda-graphs', '--check-graphs', '--max-new-tokens', '1']):
            with patch('sys.argv', ['benchmark.py', *flags]), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_args()
        tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))
        for length in (1, 3, 8, 32768):
            ids, = benchmark_inputs(tokenizer, length)
            torch.testing.assert_close(ids[0], torch.arange(length) % 3 + 1)
        self.assertEqual(len(benchmark_inputs(tokenizer)), 6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA stage timing requires a GPU')
    def test_profile_acceptance_excludes_bonus_and_preserves_final_block(self):
        model = tiny_qwen().to('cuda')
        inputs = [torch.tensor([[1, 5, 4]], device='cuda')]
        sampler = partial(speculative_sampling, model, model, tokenizer=None, debug=False)
        for new_tokens, rounds, accepted, committed in ((1, 0, 0, 0), (2, 1, 3, 1), (8, 2, 3, 3.5)):
            with self.subTest(new_tokens=new_tokens), redirect_stdout(io.StringIO()):
                profile = profile_stages(sampler, inputs, new_tokens, 0)
            self.assertEqual(profile['calls']['prefill'], 1)
            self.assertEqual(profile['calls']['verify'], rounds)
            self.assertEqual(profile['accepted_draft_tokens_per_round'], accepted)
            self.assertEqual(profile['committed_tokens_per_round'], committed)
            self.assertGreater(profile['wall_seconds'], 0)
        with redirect_stdout(io.StringIO()):
            row = benchmark_experiment(model, model, inputs, 0.5, 2, 2)
        self.assertEqual(row['greedy_exact_matches'], '')
        self.assertEqual(row['profile_requests'], 1)
        self.assertGreater(row['ar_total_e2e_s'], 0)
        self.assertGreater(row['spec_total_e2e_s'], 0)
        with redirect_stdout(io.StringIO()):
            row = benchmark_experiment(model, model, inputs, 0, 8, 1)
        self.assertEqual(row['profile_avg_accepted_draft_tokens'], 3)
        self.assertEqual(row['profile_rounds'], 2)
        self.assertEqual(row['spec_graph_setup_s'], 0)
        self.assertEqual(row['ar_graph_setup_s'], 0)
        with redirect_stdout(io.StringIO()):
            graphed = benchmark_experiment(model, model, inputs, 0, 8, 1, cuda_graphs=True)
        self.assertEqual(graphed['profile_avg_accepted_draft_tokens'], 3)
        self.assertEqual(graphed['profile_rounds'], 2)
        self.assertGreater(graphed['spec_graph_setup_s'], 0)
        self.assertGreater(graphed['ar_graph_setup_s'], 0)
        with redirect_stdout(io.StringIO()):
            reused = benchmark_experiment(model, model, inputs + [inputs[0][:, :1]], 0, 8, 1, cuda_graphs=True)
        self.assertEqual(reused['ar_graph_count'], 1)
        self.assertEqual(reused['spec_graph_count'], 3)
        self.assertEqual(reused['max_seq_len'], 11)
        self.assertEqual(reused['greedy_exact_matches'], 2)


if __name__ == '__main__':
    unittest.main()
