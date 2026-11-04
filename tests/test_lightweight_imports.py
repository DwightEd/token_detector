"""Pure cached-data and alignment stages must not import a model runtime."""
import os
from pathlib import Path
import subprocess
import sys


def test_alignment_and_experiment_cli_do_not_import_torch(tmp_path):
    script = '''
import importlib.abc
import sys
class RefuseModelRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers'}:
            raise AssertionError('Pure CPU imports requested '+fullname)
sys.meta_path.insert(0, RefuseModelRuntime())
import myproject
from myproject import Observations, WitnessDetector
from myproject.inputs import text_units
from myproject.measure import answer_axis
from experiments.provenance import research_identity
from experiments.run import parser
from experiments.head_carrier_frontend import prepare_frontend
from experiments.head_carrier_observation import run
assert 'HallucinationDetector' in dir(myproject)
assert callable(text_units) and callable(answer_axis)
assert research_identity()['source_sha256']
assert parser().prog
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
assert myproject.Observations is Observations
assert myproject.WitnessDetector is WitnessDetector
try:
    myproject.no_such_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export accepted')
print('lightweight canonical imports passed')
'''
    # Use the actually imported package location, including when the suite is
    # executed against an installed wheel outside the checkout.
    import myproject
    package_root = str(Path(myproject.__file__).resolve().parent.parent)
    environment=dict(os.environ,PYTHONPATH=package_root,OMP_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,env=environment,
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'lightweight canonical imports passed' in result.stdout
