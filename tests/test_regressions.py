"""Behavior regressions. Run on the server, without historical training."""
import importlib.util
import json
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from common import data as D, model as M, train as T
from common.evaluate import evaluate_run


class DataContract(unittest.TestCase):
    def test_package_fingerprint_changes_with_helper_code(self):
        from common.provenance import script_fingerprint
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'helper.py').write_text('value=1\n')
            (root/'note.txt').write_text('not code')
            before=script_fingerprint(root)
            self.assertEqual(set(before['files']),{'helper.py'})
            (root/'helper.py').write_text('value=2\n')
            self.assertNotEqual(before['package_sha256'],script_fingerprint(root)['package_sha256'])

    def test_stale_author_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'input.csv'
            p.write_text('f0,class\n99,0\n99,1\n')
            with self.assertRaises(ValueError):
                D._author_csv_once(np.array([[0.],[1.]],np.float32),np.array([0,1]),p)

    def test_standardize_preserves_historical_tuple_contract(self):
        out, ref = D.standardize(np.arange(12).reshape(4, 3))
        self.assertEqual(out.shape, (4, 3))
        self.assertIsNone(ref)
        np.testing.assert_allclose(out.mean(0), 0, atol=1e-6)

    def test_arbitrary_data_path_is_recorded_and_verified(self):
        from common.provenance import resolve_data_dir
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / 'cache').mkdir()
            (root / 'cache/x.npz').write_bytes(b'first')
            import hashlib
            cfg = {'data_dir': str(root), 'dataset': 'x',
                   'data_sha256': hashlib.sha256(b'first').hexdigest()}
            self.assertEqual(resolve_data_dir(cfg, None), root)
            (root / 'cache/x.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                resolve_data_dir(cfg, None)


class SourceIsolation(unittest.TestCase):
    def test_direct_train_help_works(self):
        result=subprocess.run([sys.executable,str(SCRIPTS/'common/train.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--schedule',result.stdout)

    def test_author_model_loaded_from_explicit_source(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for name, marker in [('a', 1), ('b', 2)]:
                p = root / name
                p.mkdir()
                (p / 'model.py').write_text('import torch\nclass VaDE(torch.nn.Module):\n'
                    ' def __init__(self,args):\n  super().__init__()\n  self.register_buffer("rho_",torch.ones(1))\n  self.marker=%d\n' % marker)
            state={'rho_':torch.ones(1)}
            self.assertEqual(M.build_vade(state, 1, .05, 2, source=root/'a').marker, 1)
            self.assertEqual(M.build_vade(state, 1, .05, 2, source=root/'b').marker, 2)


class EvaluationPopulation(unittest.TestCase):
    def test_final_checkpoint_fallback_never_mislabels_round(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/'config.json').write_text(json.dumps({'clients':1,'batch_size':2,
                'alpha':.05,'dataset':'x','partition_mode':'clusterlet'}))
            torch.save({'round':300,'server':{}},root/'last_checkpoint.pt')
            X=torch.tensor([[0.],[1.]]);y=torch.tensor([0,1])
            ld=DataLoader(TensorDataset(X,y),batch_size=2)
            class Predictor:
                def predict(self,x): return x[:,0].numpy().astype(int)
            with patch('common.evaluate.D.make_clients',return_value=(2,[ld],(X,y),[2],{})), \
                 patch('common.evaluate.M.build_vade',return_value=Predictor()):
                result=evaluate_run(root,'x','clusterlet',1,rounds=(100,300),data_dir=root,
                    output=root/'evaluation.json')
            self.assertEqual(set(result['rounds']),{'300'})

    def test_unique_evaluation_uses_full_data_once(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root/'config.json').write_text(json.dumps({'clients':1,'batch_size':2,
                'alpha':.05,'dataset':'x','partition_mode':'clusterlet'}))
            (root/'epoch_300_server.pt').touch()
            X, y = torch.tensor([[0.], [1.], [2.], [3.]]), torch.tensor([0,0,1,1])
            loader = DataLoader(TensorDataset(X[:2], y[:2]), batch_size=2)
            class Predictor:
                def predict(self, x): return (x[:,0]>=2).numpy().astype(int)
            with patch('common.evaluate.D.make_clients', return_value=(2,[loader],(X,y),[2],{})), \
                 patch('common.evaluate.M.load_server_state', return_value=({}, {}, {})), \
                 patch('common.evaluate.M.build_vade', return_value=Predictor()):
                result = evaluate_run(root,'x','clusterlet',1,rounds=(300,),
                    data_dir=root,population='unique_full',output=root/'new.json')
            self.assertEqual(result['n_evaluation'],4)
            self.assertEqual(result['rounds']['300']['ari']['mean'],1.)


class SchedulerAndSampling(unittest.TestCase):
    def test_epoch_scheduler_changes_effective_patch(self):
        source = '                    c[\'lr_s\'].step()\n\n                clients_vae_param.append(c[\'vade\'].state_dict())\n'
        changed = T.patch_schedule(source, 'epoch')
        self.assertIn("                c['lr_s'].step()", changed)
        self.assertNotIn("                    c['lr_s'].step()", changed)
        self.assertEqual(T.patch_schedule(source, 'author'), source)

    def test_sample_floor_rejects_impossible_budget(self):
        spec = importlib.util.spec_from_file_location('sample_patch', SCRIPTS/'lines/server_reweight/main.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'model.py'
            p.write_text('def sample(K,nums,global_pi):\n    y_hat = np.random.choice(K, size=nums, p=global_pi)\n    return y_hat\n')
            module.apply(p, 1, 0, 4)
            ns={'np':np}
            exec(p.read_text(),ns)
            with self.assertRaises(ValueError): ns['sample'](4,3,np.ones(4)/4)
            draw=ns['sample'](3,10,np.ones(3)/3)
            self.assertEqual(len(draw),10)
            self.assertTrue((np.bincount(draw,minlength=3)>=1).all())


class CoverageControls(unittest.TestCase):
    def test_missing_class_positive_control(self):
        spec=importlib.util.spec_from_file_location('coverage',SCRIPTS/'lines/coverage_metric/main.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        result=module.metric_controls()
        self.assertTrue(result['identity_pass'])
        self.assertTrue(result['missing_class_pass'])


if __name__ == '__main__': unittest.main()
