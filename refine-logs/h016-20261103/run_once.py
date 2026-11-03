"""One admitted paired-scope attempt; exact package/Git binding before weights."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def main():
    helper = ROOT / 'outputs/h015-prefix-trie-20261103/run_engineering_probe.py'
    import hashlib
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(helper) == 'fd241fa9e9d9a9f3d36e01f041e0dd3ee9c8ccc0d051d8692dca7732e6c4a14e'
    module_spec = importlib.util.spec_from_file_location('_reviewed_package_binding', helper)
    admission = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(admission)
    spec_path = Path(sys.argv[1]).absolute()
    initial_spec_hash = sha(spec_path)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    assert sha(__file__) == spec['runner_sha256']
    receipt = json.loads(Path(spec['package_receipt']).read_text(encoding='utf-8'))
    assert sha(spec['package_receipt']) == spec['package_receipt_sha256'] and receipt['status'] == 'passed'
    target, files = admission.package_snapshot(receipt)
    admission.git_package_binding(spec['core_commit'], files)
    assert Path(sys.executable).absolute() == Path(spec['python']).absolute()
    assert os.environ['PYTHONPATH'] == str(target)
    assert os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '0')
    from experiments import data, paired_scope_qualification as collector
    from experiments.prefix_completion import HFPrefixCompletion
    assert Path(collector.__file__).resolve().is_relative_to(target)
    preparation = Path(spec['preparation'])
    assert data.file_hash(preparation / 'seal.json') == spec['preparation_seal_sha256']
    manifest = data.read_json(preparation / 'manifest.json')
    assert manifest['budget'] == spec['budget']
    assert manifest['model_identity']['execution']['core_commit'] == spec['core_commit']
    assert data.file_hash(HERE / 'evaluation-protocol-v1.json') == spec['evaluation_protocol_sha256']
    # CPU evaluation protocol was fixed by the installed reader during preparation.
    # Do not import CPU-only sklearn dependencies into the isolated GPU environment.
    output = Path(spec['output']); assert not output.exists()
    started = time.monotonic()
    try:
        import torch
        hardware = manifest['model_identity']['hardware']
        assert torch.cuda.get_device_name(0) == hardware['device_name']
        assert list(torch.cuda.get_device_capability(0)) == hardware['capability']
        assert torch.version.cuda == hardware['cuda_runtime']
        result = collector.qualify(preparation, output,
            HFPrefixCompletion(manifest['model_path'], manifest['model_identity'], admit_resources=True), manifest['model_identity'])
        # Package identities and exact Git bytes remain fixed through the attempt.
        admission.package_snapshot(receipt)
        admission.git_package_binding(spec['core_commit'], files)
        assert sha(spec_path) == initial_spec_hash
        assert result['costs']['actual_forward_passes'] == spec['budget']['new_forward_passes']
        with (HERE / 'observation-finished.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(status='complete_not_evaluated', seconds_wall=time.monotonic() - started,
                initial_seal_sha256=data.file_hash(output / 'seal.json'), costs=result['costs'],
                all_four_arms_frozen=True, evaluation_material_accessed=False), stream, indent=2)
        print(json.dumps(result['costs']))
    except BaseException:
        # A collector failure remains partial. Post-collector admission failure also
        # revokes its root success marker without rewriting any captured vectors.
        if output.exists():
            (output / 'complete.json').unlink(missing_ok=True)
            (output / 'seal.json').unlink(missing_ok=True)
        with (HERE / 'runner-failure.txt').open('x', encoding='utf-8') as stream:
            stream.write(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
