"""Explicit artifact paths and content identifiers, without historical inference."""
import hashlib
import importlib.util
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dataset_file(data_dir, dataset):
    root = Path(data_dir).resolve()
    for path in (root/'cache'/(dataset+'.npz'), root/'dataset'/(dataset+'.csv'), root/(dataset+'.csv')):
        if path.is_file():
            return path
    raise FileNotFoundError('dataset not found under explicit directory: %s / %s' % (root, dataset))


def resolve_data_dir(cfg, explicit):
    value = explicit if explicit is not None else cfg.get('data_dir')
    if value is None:
        raise ValueError('historical config has no data_dir; pass --data-dir explicitly')
    root = Path(value).resolve()
    # Some synthetic/legacy configs have no fingerprint. Do not manufacture one.
    if cfg.get('data_sha256'):
        actual = sha256(dataset_file(root, cfg['dataset']))
        if actual != cfg['data_sha256']:
            raise ValueError('data SHA-256 mismatch; refusing a different same-named dataset')
    return root


def author_module(source, name):
    if source is None:
        raise ValueError('explicit author source directory is required')
    path = Path(source).resolve()/(name+'.py')
    key = '_fedvade_%s_%s' % (name, sha256(path)[:16])
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hashes(source):
    return {p.name: sha256(p) for p in sorted(Path(source).glob('*.py'))}


def script_fingerprint(root):
    root = Path(root).resolve()
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*.py'))
             if '__pycache__' not in p.parts and not p.name.startswith('._')}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'package_sha256': digest, 'files': files}


def runtime_versions():
    result = {}
    for name in ('torch', 'numpy', 'scikit-learn', 'scipy', 'pandas', 'munkres', 'threadpoolctl'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def array_sha256(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
