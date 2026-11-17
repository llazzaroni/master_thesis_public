import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from causal_chamber.results import load_result_document
from causal_chamber.run_experiment import main
from causal_chamber import run_experiment as runner
from causal_chamber.plot_results import combine_experiments, plot_experiment
from causal_chamber.splits import create_manifest


class ProgressiveResultsTests(unittest.TestCase):
    def test_resume_preserves_completed_ranked_fits_and_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=root/'data'
            data.mkdir()
            for offset,name in enumerate(('a','b','c')):
                with (data/f'{name}.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(['x','z','ir_1'])
                    writer.writerows((i,(i*7)%13,2*i+((i*7)%13)*.4+(i%3+offset)*.1)
                                     for i in range(40))
            manifest=root/'splits.json'
            manifest.write_text(json.dumps(create_manifest(data,seed=0,strategy='random',fractions=[.6,.2,.2])))
            argv=['run','--data-dir',str(data),'--manifest',str(manifest),'--output-root',str(root),
                  '--experiment-name','partial','--target','a','--budgets','5','10',
                  '--subset-seeds','0','1','--methods','pseudo_lasso','pseudo_lasso_centered',
                  '--ranked-feature-counts','2','5','--lasso-lambda-ratios','1','.1','--cv-folds','3']

            def launch(arguments):
                with patch.object(sys,'argv',arguments), contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    main()

            original_append=runner.ProgressiveWriter.append
            count=0
            def interrupt(writer,document,row):
                nonlocal count
                original_append(writer,document,row)
                count+=1
                if count==5:
                    raise KeyboardInterrupt()
            with patch.object(runner.ProgressiveWriter,'append',interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    launch(argv)
            folder=root/'partial'
            before=combine_experiments([folder])['results']
            self.assertEqual(len(before),5)
            journals=list(folder.glob('*/results.jsonl'))
            with journals[0].open('ab') as stream:
                stream.write(b'{"unfinished":')
            csv_path=journals[0].with_name('results.csv')
            with csv_path.open('a') as stream:
                stream.write('unfinished csv row')
            saved_bytes={path:path.read_bytes() for path in folder.rglob('*') if path.is_file()}
            with self.assertRaises(SystemExit):
                launch(argv+['--resume','--lasso-tol','.01'])
            self.assertEqual(saved_bytes,{path:path.read_bytes() for path in folder.rglob('*') if path.is_file()})

            with patch.object(runner,'fit_pseudo_lasso',wraps=runner.fit_pseudo_lasso) as original, \
                 patch.object(runner,'fit_centered_pseudo_lasso',wraps=runner.fit_centered_pseudo_lasso) as centered:
                launch(argv+['--resume'])
                self.assertEqual(original.call_count+centered.call_count,11)
            resumed=combine_experiments([folder])['results']
            self.assertEqual(len(resumed),16)
            resumed_by_key={runner.result_key(row):row for row in resumed}
            self.assertEqual(len(resumed_by_key),16)
            for row in before:
                self.assertEqual(row,resumed_by_key[runner.result_key(row)])
            full_argv=list(argv)
            full_argv[full_argv.index('partial')]='full'
            launch(full_argv)
            self.assertEqual(resumed_by_key,{runner.result_key(row):row
                             for row in combine_experiments([root/'full'])['results']})
            for path in folder.glob('*/results.json'):
                document=load_result_document(path)
                self.assertEqual(document['status'],'complete')
                with path.with_name('results.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))),4)
            with patch.object(runner,'fit_pseudo_lasso',side_effect=AssertionError('already completed')), \
                 patch.object(runner,'fit_centered_pseudo_lasso',side_effect=AssertionError('already completed')):
                launch(argv+['--resume'])
            self.assertEqual(len(combine_experiments([folder])['results']),16)

    def test_failure_keeps_completed_models_readable_and_skips_incomplete_journal_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=root/'data'
            data.mkdir()
            for name in ('a','b'):
                with (data/f'{name}.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(['x','ir_1'])
                    writer.writerows((i,2*i+(i%3)*.1) for i in range(20))
            manifest=root/'splits.json'
            manifest.write_text(json.dumps(create_manifest(data,seed=0,strategy='random',fractions=[.6,.2,.2])))
            argv=['run','--data-dir',str(data),'--manifest',str(manifest),'--output-root',str(root),
                  '--experiment-name','partial','--target','a','--budgets','5','--subset-seeds','0',
                  '--methods','source_only_ols','target_only_ols','pseudo_lasso','--threshold-screen']
            with patch.object(sys,'argv',argv), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 patch('causal_chamber.run_experiment.fit_pseudo_lasso',side_effect=ValueError('simulated failure')):
                with self.assertRaises(SystemExit):
                    main()
            folder=root/'partial'
            result=combine_experiments([folder])
            self.assertEqual(len(result['results']),2)
            self.assertEqual(result['status'],'partial')
            for method in ('source_only_ols','target_only_ols','pseudo_lasso'):
                self.assertEqual(load_result_document(folder/method/'results.json')['status'],'failed')
            pending=load_result_document(folder/'pseudo_lasso'/'results.json')
            self.assertEqual(pending['results'],[])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(plot_experiment(folder/'pseudo_lasso'),[])
            journal=folder/'source_only_ols'/'results.jsonl'
            with journal.open('ab') as stream:
                stream.write(b'{"partially_written":')
            self.assertEqual(len(load_result_document(folder/'source_only_ols'/'results.json')['results']),1)
            with journal.open('ab') as stream:
                stream.write(b'\n')
            with self.assertRaises(json.JSONDecodeError):
                load_result_document(folder/'source_only_ols'/'results.json')

    def test_final_snapshot_does_not_double_count_retained_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            row={'method':'target_only_ols','target':'a','target_budget':5,'subset_seed':0,'target_test_mse':1}
            (folder/'results.json').write_text(json.dumps({'storage':'snapshot','status':'complete','results':[row]}))
            (folder/'results.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(load_result_document(folder/'results.json')['results'],[row])
