import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from causal_chamber.plot_results import aggregate_results, combine_experiments, mean_environment_curves, worst_source_targets, improvement_distribution, plot_experiment


def example_result(response="ir_1"):
    rows = []
    for target in ("uniform_blue_mid.csv", "uniform_green_mid.csv"):
        for method in ("source_only_ols", "target_only_ols", "pseudo_lasso", "mixed_estimator"):
            for budget in (10, 50, 100):
                for seed in (0, 1, 2):
                    error = 20.0 if method == "source_only_ols" else (1000 / budget) * (1 + seed * .1)
                    rows.append({"target": target, "method": method, "target_budget": budget,
                                 "subset_seed": seed, "target_test_mse": error})
    return {"response": response, "results": rows}


class PlotResultsTests(unittest.TestCase):
    def test_pooled_tasks_separate_pair_identity_and_seven_comparisons(self):
        from causal_chamber.plot_results import plot_pooled_distributions
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folders=[]
            methods=['source_only_ols','target_only_ols','target_only_lasso',
                     'mixed_loss_estimator','pseudo_lasso_centered_ranked_k5']
            for response,scale in [('ir_1',1),('ir_2',100)]:
                folder=root/response; folder.mkdir(); folders.append(folder)
                rows=[{'target':target,'method':method,'target_budget':10,'subset_seed':seed,
                       'target_test_mse':scale*(1+seed)*(1 if method.startswith('pseudo') else 2)}
                      for target in ['a.csv','b.csv'] for method in methods for seed in [0,1,2]
                      if not(response=='ir_2' and target=='b.csv' and method.startswith('pseudo'))]
                (folder/'results.json').write_text(json.dumps({'response':response,'results':rows}))
            with patch('causal_chamber.plot_results.draw_improvement_distributions',return_value=[]) as draw:
                plot_pooled_distributions(folders,root/'plots')
            self.assertEqual(draw.call_count,2)
            pseudo_call,mixed_call=draw.call_args_list
            self.assertEqual(len(pseudo_call.args[0]),4)
            self.assertEqual(pseudo_call.args[2],['pseudo_lasso_centered_ranked_k5'])
            self.assertEqual(len(pseudo_call.kwargs['baselines']),5)
            self.assertEqual(mixed_call.args[2],['mixed_loss_estimator'])
            curves=pseudo_call.args[0]
            rows=improvement_distribution(curves,sorted(curves),'pseudo_lasso_centered_ranked_k5','target_only_lasso')
            self.assertEqual(len(rows[0]['targets']),3)
            self.assertEqual(rows[0]['mean_percent'],50.)
            for target in curves:
                self.assertEqual(curves[target]['target_only_lasso'][0]['num_seeds'],3)

    def test_mixed_loss_vs_centered_k5_plot_uses_available_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            centered='pseudo_lasso_centered_ranked_k5'
            rows=[{'target':target,'method':method,'target_budget':budget,
                   'subset_seed':seed,'target_test_mse':error}
                  for target in ('a.csv','b.csv') for budget in (10,30) for seed in (0,1,2)
                  for method,error in ((centered,2.),('mixed_loss_estimator',1.))
                  if not (target=='b.csv' and budget==30 and method==centered)]
            (folder/'results.json').write_text(json.dumps({'response':'ir_1','results':rows,'status':'running'}))
            outputs=plot_experiment(folder)
            direct=[path for path in outputs if path.name==f'improvement_distribution_vs_{centered}.png']
            self.assertEqual(len(direct),1)
            self.assertTrue(direct[0].is_file())
            curves=aggregate_results({"results":rows})
            comparisons=improvement_distribution(curves,['a.csv','b.csv'],'mixed_loss_estimator',centered)
            self.assertEqual([len(row['targets']) for row in comparisons],[2,1])
            self.assertEqual([row['mean_percent'] for row in comparisons],[50.,50.])

    def test_coverage_report_identifies_partial_input_and_does_not_write_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            result=example_result()
            result['results']=[{**row,'method':'target_only_lasso'} for row in result['results']
                               if row['method']=='target_only_ols' and row['target']=='uniform_blue_mid.csv']
            result.update(methods=['target_only_lasso'],status='failed')
            path=folder/'results.json'
            path.write_text(json.dumps(result))
            command=[sys.executable,'-m','causal_chamber.plot_results','--experiment-dir',str(folder),'--coverage-only']
            process=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertIn(str(path),process.stdout)
            self.assertIn('status=failed',process.stdout)
            self.assertIn('target_only_lasso, budget=10: 1 environments, seeds per environment=3–3',process.stdout)
            self.assertFalse((folder/'plots').exists())

    def test_partial_distributions_use_each_pair_and_budget_instead_of_all_methods(self):
        result={'response':'ir_1','status':'partial','results':[]}
        for target in ('a','b','c'):
            for budget in (10,30):
                result['results'].append({'target':target,'method':'target_only_lasso','target_budget':budget,
                                          'subset_seed':0,'target_test_mse':10})
            budgets=(10,30) if target!='c' else (10,)
            for budget in budgets:
                result['results'].append({'target':target,'method':'pseudo_lasso_ranked_k2','target_budget':budget,
                                          'subset_seed':0,'target_test_mse':5})
        # Another variant hasn't reached any shared budget yet.
        result['results'].append({'target':'a','method':'pseudo_lasso_centered_ranked_k20','target_budget':50,
                                  'subset_seed':0,'target_test_mse':5})
        curves=aggregate_results(result)
        rows=improvement_distribution(curves,list(curves),'pseudo_lasso_ranked_k2','target_only_lasso')
        self.assertEqual([(row['budget'],row['wins'],len(row['targets'])) for row in rows],[(10,3,3),(30,2,2)])
        self.assertEqual(rows[1]['pending_targets'],['c'])
        self.assertEqual(improvement_distribution(curves,list(curves),'pseudo_lasso_centered_ranked_k20','target_only_lasso'),[])
        from matplotlib.figure import Figure
        original_save=Figure.savefig
        labels=[]
        def save(fig,path,*args,**kwargs):
            if Path(path).name=='improvement_distribution_vs_target_only_lasso.png':
                labels.extend(tick.get_text() for ax in fig.axes for tick in ax.get_xticklabels())
            return original_save(fig,path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=save):
                files=plot_experiment(Path(tmp),result=result)
            self.assertTrue(any(path.name=='improvement_distribution_vs_target_only_lasso.png' for path in files))
            self.assertIn('10\n3/3 wins',labels)
            self.assertIn('30\n2/2 wins',labels)

    def test_target_lasso_curve_and_adaptation_comparison(self):
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
        original_plot=Axes.plot
        original_save=Figure.savefig
        result=example_result()
        result['results'].extend([{**row,'method':'target_only_lasso','target_test_mse':row['target_test_mse']*.8}
                                  for row in list(result['results']) if row['method']=='target_only_ols'])
        for method in ('pseudo_lasso_ranked_k2','pseudo_lasso_centered_ranked_k2'):
            result['results'].extend([{**row,'method':method} for row in list(result['results']) if row['method']=='pseudo_lasso'])
        distributions={}
        def save(fig,path,*args,**kwargs):
            if Path(path).name.startswith('improvement_distribution'):
                distributions[Path(path).name]=[(ax.get_title(),[tick.get_text() for tick in ax.get_xticklabels()]) for ax in fig.axes]
            return original_save(fig,path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            legacy=Path(tmp)/'plots'/'ir_1'/'improvement_distribution_pseudo_lasso_ranked_k2_vs_target_only_lasso.png'
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b'old separate plot')
            with patch('matplotlib.axes.Axes.plot',autospec=True,side_effect=original_plot) as lines, \
                 patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=save):
                files=plot_experiment(Path(tmp),result=result)
            labels={call.kwargs.get('label') for call in lines.call_args_list}
            self.assertIn('Target-only lasso',labels)
            self.assertEqual(len(files),8)
            self.assertEqual(set(distributions),{f'improvement_distribution_vs_{baseline}.png'
                             for baseline in ('source_only_ols','target_only_ols','target_only_lasso','oracle_source_or_target')})
            panels=distributions['improvement_distribution_vs_target_only_lasso.png']
            self.assertEqual(len(panels),4) # one distribution per estimator; no extra concentration axes
            self.assertFalse(legacy.exists())
            self.assertTrue(any('Centered pseudo-lasso, K=2' in title for title,ticks in panels))
            self.assertTrue(any('Pseudo-lasso, K=2' in title for title,ticks in panels))
            self.assertTrue(all(any('wins' in tick for tick in ticks) for title,ticks in panels))

    def test_centered_method_appears_in_curves_and_improvement_distributions(self):
        from matplotlib.axes import Axes
        original_plot = Axes.plot
        result = example_result()
        result['results'].extend([{**row,'method':'pseudo_lasso_centered','target_test_mse':row['target_test_mse']*.8}
                                  for row in list(result['results']) if row['method']=='pseudo_lasso'])
        with tempfile.TemporaryDirectory() as tmp:
            with patch('matplotlib.axes.Axes.plot', autospec=True, side_effect=original_plot) as plot:
                files = plot_experiment(Path(tmp), result=result)
            labels = {call.kwargs.get('label') for call in plot.call_args_list}
            self.assertIn('Centered pseudo-lasso',labels)
            self.assertEqual(len(files),7)
            self.assertTrue(all(path.is_file() for path in files))

    def test_combine_separate_estimators_and_reject_incompatible_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = example_result()
            full.update(manifest_sha256="same-data", features=["x"], fit_intercept=True)
            directories = [root / "ols", root / "pseudo"]
            for directory in directories:
                directory.mkdir()
                rows = [row for row in full["results"] if (row["method"] == "pseudo_lasso") == (directory.name == "pseudo")]
                (directory / "results.json").write_text(json.dumps({**full, "results": rows}))
            combined = combine_experiments(directories)
            self.assertEqual(len(combined["results"]), len(full["results"]))
            self.assertEqual(set(aggregate_results(combined)["uniform_blue_mid.csv"]), {"source_only_ols", "target_only_ols", "pseudo_lasso", "mixed_estimator"})
            command = [sys.executable, "-m", "causal_chamber.plot_results", "--experiment-dir",
                       *map(str, directories), "--combine", "--methods", "source_only_ols", "target_only_ols", "pseudo_lasso"]
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            output = directories[0] / "plots_combined" / "ir_1"
            self.assertEqual({p.name for p in output.iterdir()}, {"uniform_blue_mid.png", "uniform_green_mid.png", "summary_all_environments.png", "summary_worst_10pct_source.png", "improvement_distribution_vs_source_only_ols.png", "improvement_distribution_vs_target_only_ols.png", "improvement_distribution_vs_oracle_source_or_target.png"})
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                combine_experiments([directories[0], directories[0]])
            path = directories[1] / "results.json"
            changed = json.loads(path.read_text())
            changed["manifest_sha256"] = "different-data"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                combine_experiments(directories)

    def test_improvement_distribution_distinguishes_typical_and_aggregate_gains(self):
        curves = {}
        for target, base, candidate in [("large",1000,500),("small1",10,11),("small2",10,11)]:
            curves[target] = {method: [{"budget": budget, "mean": error, "num_seeds":3} for budget in (10,30)]
                              for method,error in [("source_only_ols",base),("pseudo_lasso",candidate)]}
        rows = improvement_distribution(curves,list(curves),"pseudo_lasso","source_only_ols")
        self.assertEqual(rows[0]["wins"],1)
        self.assertAlmostEqual(rows[0]["mean_percent"],10)
        self.assertAlmostEqual(rows[0]["median_percent"],-10)
        self.assertAlmostEqual(rows[0]["aggregate_reduction_percent"],100*498/1020)
        curves["zero"] = {method: [{"budget": b, "mean": 0, "num_seeds":3} for b in (10,30)]
                          for method in ("source_only_ols","pseudo_lasso")}
        rows = improvement_distribution(curves,list(curves),"pseudo_lasso","source_only_ols")
        self.assertEqual(rows[0]["excluded_targets"],["zero"])
        self.assertEqual(len(rows[0]["targets"]),3)
        for target in curves:
            for point in curves[target]["pseudo_lasso"]:
                point["mean"] = 2*curves[target]["source_only_ols"][0]["mean"]
        rows = improvement_distribution(curves,list(curves),"pseudo_lasso","source_only_ols")
        self.assertEqual(rows[0]["wins"],0)
        self.assertTrue(all(value < 0 for value in rows[0]["percent"]))

    def test_parent_experiment_loads_method_folders_and_root_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root/'new_layout'
            full = example_result()
            full.update(manifest_sha256='same-data',features=['x'],fit_intercept=True)
            for method in {row['method'] for row in full['results']}:
                folder = experiment/method
                folder.mkdir(parents=True)
                saved = {**full,'methods':[method], 'results':[row for row in full['results'] if row['method']==method]}
                (folder/'results.json').write_text(json.dumps(saved))
            combined = combine_experiments([experiment])
            self.assertEqual(len(combined['results']),len(full['results']))
            command = [sys.executable,'-m','causal_chamber.plot_results','--results-root',str(root)]
            process = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertTrue((experiment/'plots'/'ir_1'/'summary_all_environments.png').exists())
            self.assertEqual(set(combined['methods']),{'source_only_ols','target_only_ols','pseudo_lasso','mixed_estimator'})

    def test_environment_means_use_equal_weights_and_fixed_worst_subset(self):
        result = {"results": []}
        for i in range(11):
            for method in ("source_only_ols", "target_only_ols"):
                for budget in (10, 50):
                    for seed in range(1 if i == 0 else 3):
                        result["results"].append({"target": f"env_{i}.csv", "method": method,
                            "target_budget": budget, "subset_seed": seed,
                            "target_test_mse": float(i + 1) if method == "source_only_ols" else float(i + budget)})
        curves = aggregate_results(result)
        targets = sorted(curves)
        worst = worst_source_targets(curves, targets)
        self.assertEqual(worst, ["env_10.csv", "env_9.csv"])
        methods = ["source_only_ols", "target_only_ols"]
        all_mean = mean_environment_curves(curves, targets, methods)
        self.assertEqual(all_mean["source_only_ols"][0]["mean"], 6)
        self.assertEqual(all_mean["target_only_ols"][0]["mean"], 15)
        worst_mean = mean_environment_curves(curves, worst, methods)
        self.assertEqual([p["mean"] for p in worst_mean["source_only_ols"]], [10.5, 10.5])
        self.assertEqual([p["mean"] for p in worst_mean["target_only_ols"]], [19.5, 59.5])
        curves["env_0.csv"]["target_only_ols"].pop()
        common = mean_environment_curves(curves, targets, methods)
        self.assertEqual([p["budget"] for p in common["target_only_ols"]], [10])

    def test_seed_aggregation_and_invalid_records(self):
        result = example_result()
        curves = aggregate_results(result)
        target = curves["uniform_blue_mid.csv"]
        self.assertEqual([p["budget"] for p in target["target_only_ols"]], [10, 50, 100])
        self.assertAlmostEqual(target["target_only_ols"][0]["mean"], 110)
        self.assertNotIn("std", target["target_only_ols"][0])
        for method in target.values():
            self.assertTrue(all(p["num_seeds"] == 3 for p in method))
        result["results"].append(dict(result["results"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_results(result)
        result = example_result()
        result["results"][0]["target_test_mse"] = float("nan")
        with self.assertRaises(ValueError):
            aggregate_results(result)

    def test_plot_files_and_multiple_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task in ("ir_1", "vis_1"):
                experiment = root / task
                experiment.mkdir()
                original = json.dumps(example_result(task))
                (experiment / "results.json").write_text(original)
                import matplotlib.pyplot as plt
                subplots = plt.subplots
                axes = []
                def track_axes(*args, **kwargs):
                    fig, ax = subplots(*args, **kwargs)
                    if hasattr(ax, "lines"):
                        axes.append(ax)
                    return fig, ax
                with patch.object(plt, "subplots", side_effect=track_axes):
                    files = plot_experiment(experiment)
                curve_axes=[ax for ax in axes if ax.get_ylabel().startswith('Mean target test MSE')]
                self.assertEqual(len(curve_axes),4)
                self.assertTrue(all(len(ax.collections) == 0 for ax in curve_axes))
                self.assertTrue(all(ax.get_yscale() == "log" for ax in curve_axes))
                self.assertTrue(all({line.get_label() for line in ax.lines} ==
                                    {"Source-only OLS", "Target-only OLS", "Pseudo-lasso", "Mixed estimator"} for ax in curve_axes))
                self.assertEqual(len(files), 7)
                self.assertTrue(all(p.stat().st_size > 1000 for p in files))
                self.assertEqual((experiment / "results.json").read_text(), original)
                files = plot_experiment(experiment, targets=["uniform_blue_mid"], formats=["png"])
                self.assertEqual(len(files), 6)
                self.assertTrue(files[0].exists())
                self.assertEqual({p.suffix for p in files}, {".png"})
                self.assertEqual({p.name for p in (experiment / "plots" / task).iterdir()},
                                 {"uniform_blue_mid.png", "uniform_green_mid.png", "summary_all_environments.png", "summary_worst_10pct_source.png"} |
                                 {f"improvement_distribution_vs_{baseline}.png" for baseline in ('source_only_ols','target_only_ols','oracle_source_or_target')})
            process = subprocess.run([sys.executable, "-m", "causal_chamber.plot_results",
                                      "--results-root", str(root), "--formats", "png"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("response ir_1", process.stdout)
            self.assertIn("response vis_1", process.stdout)

    def test_zero_error_with_linear_axis_and_single_seed(self):
        result = {"response": "ir_1", "results": [{"target": "example.csv", "method": "source_only_ols",
                  "target_budget": 10, "target_test_mse": 0.0}]}
        curves = aggregate_results(result)
        self.assertEqual(curves["example.csv"]["source_only_ols"][0]["mean"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "results.json").write_text(json.dumps(result))
            self.assertEqual(len(plot_experiment(directory, formats=["png"], y_scale="linear")), 3)
            with self.assertRaisesRegex(ValueError, "Zero mean MSE"):
                plot_experiment(directory, y_scale="log")


if __name__ == "__main__":
    unittest.main()
