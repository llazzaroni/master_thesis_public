import csv
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from causal_chamber.estimators import SourceScreen, SourceMeanFit, fit_ols, average_source_fits, fit_mixed_estimator, fit_lasso_coordinate_descent, fit_target_lasso, fit_pseudo_lasso, fit_centered_pseudo_lasso, minimum_norm_least_squares, screen_sources
from causal_chamber.splits import create_manifest
from causal_chamber.plot_results import combine_experiments


class LassoTests(unittest.TestCase):
    def test_underdetermined_ols_uses_minimum_norm_pseudoinverse(self):
        design = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        response = np.array([2.0, -1.0])
        beta, rank = minimum_norm_least_squares(design, response)
        expected = np.linalg.pinv(design) @ response
        self.assertEqual(rank, 2)
        np.testing.assert_allclose(beta, expected)
        np.testing.assert_allclose(design @ beta, response)

    def test_target_lasso_matches_empty_support_with_identical_cv(self):
        rng=np.random.default_rng(81)
        x=rng.normal(size=(40,3))*[1,100,.01]
        y=6+x @ np.array([2,.03,0])+rng.normal(size=40)*.1
        baseline=fit_target_lasso(x,y,seed=2)
        empty=fit_pseudo_lasso(x,y,SourceScreen(np.zeros(3),0,[],[]),seed=2)
        np.testing.assert_array_equal(baseline.coefficients,empty.coefficients)
        self.assertEqual(baseline.intercept,empty.intercept)
        self.assertEqual(baseline.cv_rows,empty.cv_rows)
        self.assertEqual(baseline.lambda_reg,empty.lambda_reg)
        self.assertEqual(baseline.cv_folds,5)
        self.assertEqual(len(baseline.cv_rows),7)
        single=fit_target_lasso(x[:1],y[:1])
        self.assertEqual(single.cv_folds,0)
        self.assertEqual(single.tuning,'insufficient_samples_strongest_penalty')

    def test_centered_lasso_corrects_screened_features_and_handles_empty_support(self):
        x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        y = 7 + x @ np.array([5., -2.])
        screen = SourceScreen(np.array([2., 99.]), 3., [0], [])
        original = fit_pseudo_lasso(x, y, screen, lambda_grid=[.5])
        centered = fit_centered_pseudo_lasso(x, y, screen, lambda_grid=[.5])
        np.testing.assert_allclose(original.coefficients, [2., -1.5])
        np.testing.assert_allclose(centered.coefficients, [4.5, -1.5])
        self.assertAlmostEqual(centered.intercept, 7)
        self.assertLess(np.mean((centered.predict(x)-y)**2), np.mean((original.predict(x)-y)**2))
        strong = fit_centered_pseudo_lasso(x, y, screen, lambda_grid=[100])
        np.testing.assert_allclose(strong.coefficients, [2., 0.])
        for support in ([], [0, 1]):
            candidate = SourceScreen(np.array([2., -1.]), 3., support, [])
            fitted = fit_centered_pseudo_lasso(x, y, candidate, lambda_grid=[.5])
            if not support:
                lasso = fit_lasso_coordinate_descent(x, y, lambda_reg=.5)
                np.testing.assert_allclose(fitted.coefficients, lasso.coefficients)
                self.assertAlmostEqual(fitted.intercept, lasso.intercept)
            else:
                np.testing.assert_allclose(fitted.coefficients, [4.5, -1.5])
                self.assertNotEqual(fitted.tuning, 'no_complement')
        # The source center stays fixed; each target fold fits its own scaling.
        rng = np.random.default_rng(18)
        x = rng.normal(size=(50,2))
        y = 7+x @ np.array([5., -2.])
        fitted = fit_centered_pseudo_lasso(x,y,screen,seed=1)
        repeated = fit_centered_pseudo_lasso(x,y,screen,seed=1)
        self.assertEqual(fitted.cv_folds,5)
        self.assertEqual(len(fitted.cv_rows),7)
        self.assertEqual(fitted.cv_rows,repeated.cv_rows)
        np.testing.assert_array_equal(fitted.coefficients,repeated.coefficients)
        no_intercept = fit_centered_pseudo_lasso(x,y-7,screen,lambda_grid=[.001],fit_intercept=False)
        self.assertEqual(no_intercept.intercept,0)

    def test_orthogonal_solution_and_intercept(self):
        x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        y = 7 + x @ np.array([3., -.2])
        fit = fit_lasso_coordinate_descent(x, y, lambda_reg=.5)
        np.testing.assert_allclose(fit.coefficients, [2.5, 0])
        self.assertAlmostEqual(fit.intercept, 7)
        self.assertTrue(fit.converged)
        scaled = fit_lasso_coordinate_descent(x * [100, .01], y, lambda_reg=.5)
        np.testing.assert_allclose(scaled.predict(x * [100, .01]), fit.predict(x))
        no_intercept = fit_lasso_coordinate_descent(x, y - 7, lambda_reg=.5, fit_intercept=False)
        self.assertEqual(no_intercept.intercept, 0)
        np.testing.assert_allclose(no_intercept.coefficients, [2.5, 0])

    def test_source_screen_and_target_correction(self):
        rng = np.random.default_rng(11)
        x = np.column_stack((rng.normal(size=(200, 2)), np.ones(200)))
        sources = [(x, 3 + x[:, 0]*2 + x[:, 1]*slope) for slope in (-3, 3)]
        screen = screen_sources(sources)
        self.assertEqual(screen.invariant_support, [0])
        self.assertFalse(screen.details[2]['identifiable'])
        y = 10 + x[:, 0]*2 + x[:, 1]*5
        fit = fit_pseudo_lasso(x, y, screen, lambda_grid=[.001], seed=0)
        self.assertAlmostEqual(fit.coefficients[0], screen.mean_coefficients[0])
        self.assertAlmostEqual(fit.intercept, 10, places=2)
        self.assertLess(np.mean((fit.predict(x)-y)**2), 1e-4)
        cv_fit = fit_pseudo_lasso(x[:40], y[:40], screen, seed=2)
        self.assertEqual(cv_fit.cv_folds, 5)
        self.assertEqual(len(cv_fit.cv_rows), 7)
        self.assertGreater(cv_fit.lambda_reg, 0)
        repeat = fit_pseudo_lasso(x[:40], y[:40], screen, seed=2)
        np.testing.assert_array_equal(repeat.coefficients, cv_fit.coefficients)
        self.assertEqual(repeat.cv_rows, cv_fit.cv_rows)
        empty_screen = SourceScreen(np.zeros(3), 0., [], [])
        fit = fit_pseudo_lasso(x[:1], y[:1], empty_screen)
        self.assertEqual(fit.tuning, 'insufficient_samples_strongest_penalty')
        self.assertTrue(np.isfinite(fit.predict(x)).all())
        all_screen = SourceScreen(np.array([2.,5.,0.]), 3., [0,1,2], [])
        fit = fit_pseudo_lasso(x, y, all_screen)
        self.assertEqual(fit.tuning, 'no_complement')
        np.testing.assert_allclose(fit.predict(x), y)
        with self.assertRaises(ValueError):
            fit_pseudo_lasso(x, y, screen, lambda_grid=[-1])

    def test_permissive_screening_defaults_and_rejection_diagnostics(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(200,2))
        design = np.column_stack((np.ones(200),x))
        noise = rng.normal(size=200)
        noise -= design @ np.linalg.lstsq(design,noise,rcond=None)[0]
        noise *= 10/np.std(noise)
        sources = [(x,2+x @ np.array([.05,slope])+noise) for slope in (1.,2.,3.)]
        strict = screen_sources(sources,z_threshold=2,stability_threshold=.35)
        loose = screen_sources(sources)
        self.assertEqual(strict.invariant_support,[])
        self.assertEqual(loose.invariant_support,[0,1])
        self.assertIn('detectability',strict.details[0]['rejection_reasons'])
        self.assertIn('stability',strict.details[1]['rejection_reasons'])
        capped = screen_sources(sources,max_invariant_features=1)
        self.assertEqual(capped.invariant_support,[0])
        self.assertIn('feature_cap',capped.details[1]['rejection_reasons'])

    def test_runner_outputs_and_test_label_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            data.mkdir()
            rng = np.random.default_rng(3)
            matrices = {}
            for name, slope in [('a', 5), ('b', -3), ('c', 3), ('uniform_reference', 100)]:
                x = rng.normal(size=(40,2))
                matrices[name] = np.column_stack((x, 7+2*x[:,0]+slope*x[:,1]+rng.normal(size=40)*.1))
            def write_data():
                for name, values in matrices.items():
                    with (data/f'{name}.csv').open('w',newline='') as stream:
                        writer = csv.writer(stream)
                        writer.writerow(['x1','x2','ir_1'])
                        writer.writerows(values)
            def run(name, extra=()):
                write_data()
                manifest = create_manifest(data,seed=0,strategy='random',fractions=[.6,.2,.2])
                path = root/'splits.json'
                path.write_text(json.dumps(manifest))
                command = [sys.executable,'-m','causal_chamber.run_experiment','--data-dir',str(data),
                           '--manifest',str(path),'--output-root',str(root/'out'),'--experiment-name',name,
                           '--target','a','--budgets','5','10','--threshold-screen']
                proc = subprocess.run(command+list(extra),capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stderr)
                directory = root/'out'/name
                self.assertFalse((directory/'results.json').exists())
                self.assertTrue((directory/'splits.json').exists())
                for method_dir in directory.iterdir():
                    if method_dir.is_dir():
                        saved = json.loads((method_dir/'results.json').read_text())
                        self.assertEqual(saved['methods'],[method_dir.name])
                        self.assertTrue(all(row['method'] == method_dir.name for row in saved['results']))
                return combine_experiments([directory]), manifest
            result, manifest = run('first')
            self.assertEqual(len(result['results']),24)
            self.assertEqual(result['source_environments_by_target']['a.csv'],['b.csv','c.csv'])
            self.assertIn('source_screens',result)
            self.assertIn('mixed_source_means',result)
            self.assertEqual({r['method'] for r in result['results']},
                             {'source_only_ols','target_only_ols','pseudo_lasso','mixed_estimator'})
            csv_rows, summary_rows = 0, 0
            for method in result['methods']:
                folder = root/'out'/'first'/method
                with (folder/'results.csv').open() as stream:
                    csv_rows += len(list(csv.DictReader(stream)))
                with (folder/'summary.csv').open() as stream:
                    summary_rows += len(list(csv.DictReader(stream)))
            self.assertEqual(csv_rows,24)
            self.assertEqual(summary_rows,8)
            screen = result['source_screens']['a.csv']
            self.assertEqual(screen['settings']['z_threshold'],0)
            self.assertEqual(screen['settings']['stability_threshold'],1)
            self.assertIn('rejection_reasons',screen['details'][0])
            matrices['a'][manifest['environments']['a.csv']['splits']['test'], -1] += 1000
            changed, _ = run('changed')
            for before, after in zip(result['results'], changed['results']):
                self.assertEqual(before['coefficients'],after['coefficients'])
                self.assertEqual(before['intercept'],after['intercept'])
                self.assertEqual(before['lambda_lasso'],after['lambda_lasso'])
                self.assertEqual(before['mixed_lambda'],after['mixed_lambda'])
                self.assertEqual(before['mixed_cv_rows'],after['mixed_cv_rows'])
                self.assertNotEqual(before['target_test_mse'],after['target_test_mse'])
            ols, _ = run('ols',['--methods','source_only_ols','target_only_ols'])
            self.assertEqual(len(ols['results']),12)
            self.assertEqual(ols['source_screens'],{})
            mixed, _ = run('mixed',['--methods','mixed_estimator','--mixed-lambda-grid','0','.5','1','--cv-folds','3'])
            self.assertEqual(len(mixed['results']),6)
            self.assertEqual(mixed['source_screens'],{})
            self.assertTrue(all(row['mixed_cv_folds'] == 3 for row in mixed['results']))
            self.assertTrue(all(row['mixed_lambda'] in {0,.5,1} for row in mixed['results']))
            first, _ = run('incremental',['--methods','source_only_ols'])
            directory = root/'out'/'incremental'
            preserved = {path: path.read_bytes() for path in (
                directory/'config.json',directory/'splits.json',directory/'source_only_ols'/'results.json')}
            added, _ = run('incremental',['--methods','target_only_ols'])
            self.assertEqual(set(added['methods']),{'source_only_ols','target_only_ols'})
            self.assertEqual(len(added['results']),12)
            for path, content in preserved.items():
                self.assertEqual(path.read_bytes(),content)
            command = [sys.executable,'-m','causal_chamber.run_experiment','--data-dir',str(data),
                       '--manifest',str(root/'splits.json'),'--output-root',str(root/'out'),
                       '--experiment-name','incremental','--target','a','--budgets','5','10','--threshold-screen']
            duplicate = subprocess.run(command+['--methods','target_only_ols'],capture_output=True,text=True)
            self.assertNotEqual(duplicate.returncode,0)
            self.assertIn('Method folder already exists',duplicate.stderr)
            mismatch = subprocess.run(command+['--methods','mixed_estimator','--subset-seeds','99'],capture_output=True,text=True)
            self.assertNotEqual(mismatch.returncode,0)
            self.assertIn('subset_seeds differs',mismatch.stderr)
            self.assertFalse((directory/'mixed_estimator').exists())
            centered, _ = run('incremental',['--methods','pseudo_lasso_centered'])
            self.assertIn('pseudo_lasso_centered',centered['methods'])
            self.assertEqual(len(centered['results']),18)
            centered_file = json.loads((directory/'pseudo_lasso_centered'/'results.json').read_text())
            self.assertIn('a.csv',centered_file['source_screens'])
            self.assertTrue(all(row['lasso_cv_rows'] is not None for row in centered_file['results']))
            target_lasso, _ = run('incremental',['--methods','target_only_lasso'])
            self.assertEqual(len(target_lasso['results']),24)
            baseline_file=json.loads((directory/'target_only_lasso'/'results.json').read_text())
            self.assertEqual(baseline_file['source_screens'],{})
            self.assertEqual(baseline_file['mixed_source_means'],{})
            self.assertEqual(baseline_file['status'],'complete')
            self.assertTrue(all(row['lasso_cv_folds']==5 for row in baseline_file['results']))
            self.assertTrue(all(row['lambda_lasso'] is not None for row in baseline_file['results']))
            self.assertTrue(all(row['num_invariant_features'] is None for row in baseline_file['results']))
            baseline_before, _ = run('lasso_before',['--methods','target_only_lasso'])
            centered_changed, _ = run('centered_before',['--methods','pseudo_lasso_centered'])
            matrices['a'][manifest['environments']['a.csv']['splits']['test'], -1] += 200
            centered_after, _ = run('centered_after',['--methods','pseudo_lasso_centered'])
            baseline_after, _ = run('lasso_after',['--methods','target_only_lasso'])
            for before, after in zip(baseline_before['results'],baseline_after['results']):
                self.assertEqual(before['coefficients'],after['coefficients'])
                self.assertEqual(before['lasso_cv_rows'],after['lasso_cv_rows'])
                self.assertNotEqual(before['target_test_mse'],after['target_test_mse'])
            for before, after in zip(centered_changed['results'],centered_after['results']):
                self.assertEqual(before['coefficients'],after['coefficients'])
                self.assertEqual(before['lasso_cv_rows'],after['lasso_cv_rows'])
                self.assertNotEqual(before['target_test_mse'],after['target_test_mse'])


class MixedEstimatorTests(unittest.TestCase):
    def test_equal_environment_average_and_endpoints(self):
        small = np.arange(3.).reshape(-1,1)
        large = np.arange(50.).reshape(-1,1)
        source_mean = average_source_fits([fit_ols(small, 2*small[:,0]+1),
                                          fit_ols(large, 6*large[:,0]+9)])
        np.testing.assert_allclose(source_mean.coefficients, [4])
        self.assertAlmostEqual(source_mean.intercept, 5)
        x = np.arange(12.).reshape(-1,1)
        y = 8*x[:,0]-3
        source_endpoint = fit_mixed_estimator(x,y,source_mean,lambda_grid=[0])
        target_endpoint = fit_mixed_estimator(x,y,source_mean,lambda_grid=[1])
        blend = fit_mixed_estimator(x,y,source_mean,lambda_grid=[.25])
        np.testing.assert_allclose(source_endpoint.predict(x),source_mean.predict(x))
        np.testing.assert_allclose(target_endpoint.predict(x),y)
        np.testing.assert_allclose(blend.predict(x),.25*y+.75*source_mean.predict(x))
        self.assertEqual(blend.tuning,'fixed_weight')

    def test_target_cv_and_small_budget_fallback(self):
        x = np.arange(15.).reshape(-1,1)
        y = 3*x[:,0]+2
        source = SourceMeanFit(np.array([-10.]),100.)
        with patch("causal_chamber.estimators.fit_ols", wraps=fit_ols) as fits:
            fit = fit_mixed_estimator(x,y,source,seed=4)
        self.assertEqual([len(call.args[1]) for call in fits.call_args_list], [12]*5+[15])
        validation_folds = np.array_split(np.random.default_rng(4).permutation(15),5)
        for call, validation in zip(fits.call_args_list[:5],validation_folds):
            train = np.setdiff1d(np.arange(15),validation)
            np.testing.assert_array_equal(call.args[0],x[train])
            np.testing.assert_array_equal(call.args[1],y[train])
        self.assertEqual(fit.lambda_mix,1)
        self.assertEqual(fit.cv_folds,5)
        np.testing.assert_allclose(fit.predict(x),y,atol=1e-12)
        repeat = fit_mixed_estimator(x,y,source,seed=4)
        self.assertEqual(fit.cv_rows,repeat.cv_rows)
        # Balanced responses and constant predictors: fold training means
        # overfit; the zero source prediction gives the best validation score.
        x = np.zeros((8,1))
        y = np.array([1.,-1.]*4)
        fit = fit_mixed_estimator(x,y,SourceMeanFit(np.zeros(1),0),seed=0)
        self.assertEqual(fit.lambda_mix,0)
        one = fit_mixed_estimator(x[:1],y[:1],source)
        self.assertEqual(one.cv_folds,0)
        self.assertEqual(one.lambda_mix,0)
        self.assertEqual(one.tuning,'insufficient_samples_smallest_target_weight')
        with self.assertRaises(ValueError):
            fit_mixed_estimator(x,y,source,lambda_grid=[1.1])
        x = np.arange(12.).reshape(-1,1)
        y = 3*x[:,0]
        no_intercept = fit_mixed_estimator(x,y,SourceMeanFit(np.zeros(1),0),fit_intercept=False)
        self.assertEqual(no_intercept.intercept,0)
        self.assertEqual(no_intercept.lambda_mix,1)


if __name__ == '__main__':
    unittest.main()
