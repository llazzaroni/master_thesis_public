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

from causal_chamber.estimators import fit_mixed_loss_estimator, prepare_source_loss, fit_ols
from causal_chamber.run_experiment import main
from causal_chamber.results import load_result_document
from causal_chamber.splits import create_manifest


class MixedLossTests(unittest.TestCase):
    def test_weighted_solution_endpoints_and_replication_invariance(self):
        rng=np.random.default_rng(7)
        xs=rng.normal(size=(41,3)); ys=xs @ np.array([2.,-1.,3.])+4+rng.normal(size=41)
        xt=rng.normal(size=(9,3)); yt=xt @ np.array([-2.,1.,2.])-3+rng.normal(size=9)
        for intercept in (True,False):
            source=prepare_source_loss(xs,ys,fit_intercept=intercept)
            for weight in (0,.2,.7,1):
                fit=fit_mixed_loss_estimator(xt,yt,source,lambda_grid=[weight],fit_intercept=intercept)
                ds=np.column_stack((np.ones(len(xs)),xs)) if intercept else xs
                dt=np.column_stack((np.ones(len(xt)),xt)) if intercept else xt
                if weight==0:
                    a,b=dt,yt
                elif weight==1:
                    a,b=ds,ys
                else:
                    a=np.vstack((ds*np.sqrt(weight/len(xs)),dt*np.sqrt((1-weight)/len(xt))))
                    b=np.concatenate((ys*np.sqrt(weight/len(xs)),yt*np.sqrt((1-weight)/len(xt))))
                expected=np.linalg.lstsq(a,b,rcond=None)[0]
                actual=np.r_[fit.intercept,fit.coefficients] if intercept else fit.coefficients
                np.testing.assert_allclose(actual,expected,atol=1e-11)
                repeated=prepare_source_loss(np.tile(xs,(3,1)),np.tile(ys,3),fit_intercept=intercept)
                replicated=fit_mixed_loss_estimator(xt,yt,repeated,lambda_grid=[weight],fit_intercept=intercept)
                np.testing.assert_allclose(fit.predict(xt),replicated.predict(xt),atol=1e-11)
            errors={v:0. for v in (0,.1,.2,.4,.6,.8,1)}
            design=np.column_stack((np.ones(len(xt)),xt)) if intercept else xt
            folds=np.array_split(np.random.default_rng(2).permutation(len(yt)),3)
            for val in folds:
                train=np.setdiff1d(np.arange(len(yt)),val)
                for weight in errors:
                    fold=fit_mixed_loss_estimator(xt[train],yt[train],source,lambda_grid=[weight],fit_intercept=intercept)
                    errors[weight]+=np.sum((fold.predict(xt[val])-yt[val])**2)/len(yt)
            fitted=fit_mixed_loss_estimator(xt,yt,source,seed=2,cv_folds=3,fit_intercept=intercept)
            for row in fitted.cv_rows:
                self.assertAlmostEqual(row['mse'],errors[row['candidate']])
            self.assertEqual(fitted.lambda_mix,min(errors,key=lambda v:(errors[v],-v)))

    def test_runner_output_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; data.mkdir()
            for offset,name in enumerate(('a','b','uniform_reference')):
                with (data/f'{name}.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream); writer.writerow(['x','ir_1'])
                    writer.writerows((i,2*i+offset+(i%3)*.1) for i in range(30))
            manifest=root/'splits.json'
            manifest.write_text(json.dumps(create_manifest(data,seed=0,strategy='random',fractions=[.6,.2,.2])))
            argv=['run','--data-dir',str(data),'--manifest',str(manifest),'--output-root',str(root),
                  '--experiment-name','mixed','--methods','mixed_loss_estimator','--budgets','5','10',
                  '--subset-seeds','0','1','--cv-folds','3']
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                main()
            path=root/'mixed'/'mixed_loss_estimator'/'results.json'
            document=load_result_document(path)
            self.assertEqual(len(document['results']),12)
            self.assertNotIn('uniform_reference.csv',document['source_environments_by_target']['a.csv'])
            self.assertTrue(all(row['mixed_cv_folds']==3 and row['mixed_cv_rows'] for row in document['results']))
            before=document['results']
            with patch.object(sys,'argv',argv+['--resume']),contextlib.redirect_stdout(io.StringIO()), \
                 patch('causal_chamber.run_experiment.fit_mixed_loss_estimator',side_effect=AssertionError('completed')):
                main()
            self.assertEqual(load_result_document(path)['results'],before)
