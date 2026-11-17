import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from causal_chamber.estimators import SourceScreen, estimate_source_coefficients, ranked_source_screen
from causal_chamber.methods import expand_ranked_methods
from causal_chamber.plot_results import combine_experiments, plot_experiment, method_label
from causal_chamber.run_experiment import main, ProgressiveWriter
from causal_chamber.splits import create_manifest
from causal_chamber.summarize_support import load_results, summarize_support


class RankedExperimentsTests(unittest.TestCase):
    def test_cli_default_grid_and_explicit_overrides(self):
        cases = [(['--methods','pseudo_lasso','pseudo_lasso_centered'], [2,5,10]),
                 (['--methods','pseudo_lasso','--ranked-feature-counts','2','5'], [2,5]),
                 (['--methods','pseudo_lasso','--threshold-screen'], None),
                 (['--methods','target_only_ols'], None)]
        with tempfile.TemporaryDirectory() as tmp:
            for flags, counts in cases:
                with patch.object(sys,'argv',['run','--output-root',tmp,'--experiment-name','new']+flags), \
                     patch('causal_chamber.run_experiment.run',side_effect=ValueError('stop before data loading')) as run, \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main()
                    args=run.call_args.args[0]
                    self.assertEqual(args.ranked_feature_counts,counts)
                    methods=flags[flags.index('--methods')+1:]
                    methods=methods[:next((i for i,v in enumerate(methods) if v.startswith('--')),len(methods))]
                    self.assertEqual(args.methods,expand_ranked_methods(methods,counts))

    def test_ranked_support_uses_scores_not_sign_gates_and_caps_explicitly(self):
        details = [{'feature':i,'ranking_score':score,'identifiable':identified,'sign_consistent':False}
                   for i,score,identified in [(0,2,True),(1,9,False),(2,3,True),(3,3,True)]]
        screen = SourceScreen(np.arange(4.),0,[],details)
        two = ranked_source_screen(screen,2)
        three = ranked_source_screen(screen,3)
        capped = ranked_source_screen(screen,20)
        self.assertEqual(two.invariant_support,[2,3])
        self.assertEqual(three.invariant_support,[0,2,3])
        self.assertEqual(capped.invariant_support,three.invariant_support)
        self.assertIsNone(capped.details[1]['rank'])
        forced = ranked_source_screen(screen,2,include_unidentified=True)
        self.assertEqual(forced.invariant_support,[1,2])
        with self.assertRaises(ValueError):
            expand_ranked_methods(['pseudo_lasso'],[2,2])

    def test_ten_models_share_rankings_and_subsets_and_are_plottable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=root/'data'
            data.mkdir()
            rng=np.random.default_rng(18)
            for name,slope in [('a',1),('b',-1),('c',2),('uniform_reference',100)]:
                x=rng.normal(size=(80,6))
                y=4+x @ np.array([2,slope,.5,1,-2,.3])+rng.normal(size=80)*.5
                with (data/f'{name}.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow([f'x{i}' for i in range(6)]+['constant','ir_1'])
                    writer.writerows(np.column_stack((x,np.ones(80),y)))
            manifest=create_manifest(data,seed=0,strategy='random',fractions=[.6,.2,.2])
            manifest_path=root/'splits.json'
            manifest_path.write_text(json.dumps(manifest))
            argv=['run_experiment','--data-dir',str(data),'--manifest',str(manifest_path),
                  '--output-root',str(root),'--experiment-name','ranked',
                  '--methods','pseudo_lasso','pseudo_lasso_centered',
                  '--ranked-feature-counts','2','5','10','15','20',
                  '--budgets','10','30','--subset-seeds','0','--lasso-lambda-ratios','1','.1','--cv-folds','3']
            live_checked=[]
            original_append=ProgressiveWriter.append
            def observe_append(writer,document,row):
                original_append(writer,document,row)
                if not live_checked:
                    live=combine_experiments([root/'ranked'])
                    self.assertEqual(len(live['results']),1)
                    self.assertEqual(live['status'],'partial')
                    saved=load_results(root/'ranked',row['method'])
                    self.assertEqual(saved['status'],'running')
                    self.assertEqual(saved['results'],[row])
                    methods=expand_ranked_methods(['pseudo_lasso','pseudo_lasso_centered'],[2,5,10,15,20])
                    files=plot_experiment(root/'ranked',result=live,methods=methods,output_root=root/'live_plots')
                    self.assertEqual(len(files),1) # pending models excluded; no unfair comparison summary
                    live_checked.append(True)
            with patch.object(sys,'argv',argv), contextlib.redirect_stdout(io.StringIO()), \
                 patch('causal_chamber.run_experiment.estimate_source_coefficients',wraps=estimate_source_coefficients) as fits, \
                 patch.object(ProgressiveWriter,'append',autospec=True,side_effect=observe_append):
                main()
            self.assertEqual(live_checked,[True])
            self.assertEqual(fits.call_count,3) # reference excluded; every remaining source fit only once
            folder=root/'ranked'
            methods=expand_ranked_methods(['pseudo_lasso','pseudo_lasso_centered'],[2,5,10,15,20])
            self.assertEqual({p.name for p in folder.iterdir() if p.is_dir()},set(methods))
            documents={method:load_results(folder,method) for method in methods}
            self.assertTrue(all(document['status']=='complete' for document in documents.values()))
            for target in manifest['environments']:
                previous=set()
                for k in (2,5,10,15,20):
                    original=documents[f'pseudo_lasso_ranked_k{k}']['source_screens'][target]
                    centered=documents[f'pseudo_lasso_centered_ranked_k{k}']['source_screens'][target]
                    self.assertEqual(original,centered)
                    self.assertEqual(original['requested_features'],k)
                    self.assertEqual(original['actual_features'],min(k,6))
                    support=set(original['invariant_support'])
                    self.assertTrue(previous <= support)
                    previous=support
                    rows=[row for row in documents[f'pseudo_lasso_ranked_k{k}']['results'] if row['target']==target]
                    for row in rows:
                        self.assertEqual(row['requested_invariant_features'],k)
                        self.assertEqual(row['num_invariant_features'],min(k,6))
                        self.assertEqual(row['lasso_cv_folds'],3)
                        self.assertEqual(len(row['lasso_cv_rows']),2)
                        for j in support:
                            self.assertEqual(row['coefficients'][j],original['mean_coefficients'][j])
                pools=[document['target_budget_pools'][target] for document in documents.values()]
                self.assertTrue(all(pool==pools[0] for pool in pools))
            combined=combine_experiments([folder])
            self.assertEqual(len(combined['results']),80)
            self.assertEqual(summarize_support(documents['pseudo_lasso_ranked_k20'])['mean_size'],6)
            from matplotlib.axes import Axes
            original_plot=Axes.plot
            with patch('matplotlib.axes.Axes.plot',autospec=True,side_effect=original_plot) as lines, \
                 contextlib.redirect_stdout(io.StringIO()):
                files=plot_experiment(folder,targets=['a'],result=combined)
            labels={call.kwargs.get('label') for call in lines.call_args_list}
            self.assertTrue({method_label(method) for method in methods} <= labels)
            self.assertEqual(len(files),2)
            self.assertTrue(all(path.is_file() for path in files))
