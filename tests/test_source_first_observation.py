"""Resource/order/partial-failure tests; no model, tokenizer or truth labels."""
import inspect
import io
import json
from types import SimpleNamespace

import pytest

from experiments import data
from experiments import source_first_observation as obs


GATES = dict(pre_import_available_ram=2684354560,
             post_import_tokenizer_and_preload_and_prerequest_available_ram=2147483648,
             post_import_tokenizer_and_preload_and_prerequest_free_cuda=3221225472)
IDENTITY = dict(source_sha256={'source.py': 'a' * 64}, python='test')


class FakeInstrument:
    def __init__(self, *, lengths=None, free_ram=4 * 1024**3, fail=False, bad_stop=False):
        self.lengths = lengths or [2] * 6
        self.free_ram, self.fail, self.bad_stop = free_ram, fail, bad_stop
        self.tokenizer_calls = 0
        self.resources, self.events = [], []

    def snapshot(self, stage, *, cuda=False):
        row = dict(stage=stage, available_ram_bytes=self.free_ram, free_cuda_bytes=6 * 1024**3)
        self.resources.append(row)
        return row

    def input_ids(self, messages):
        self.events.append('tokenize')
        length = self.lengths[self.tokenizer_calls]
        self.tokenizer_calls += 1
        return [10] * length

    def load(self):
        self.events.append('load')
        return {'effective': 'fake'}

    def generate(self, messages, ids, root, started):
        self.events.append('generate')
        forward = dict(forward_entry_attempts=2 if self.fail else 1,
                       successful_python_returns=1, processed_input_positions=len(ids),
                       rows=[dict(input_token_ids=ids)])
        data.write_json(root / 'forward-accounting.json', forward)
        if self.fail:
            raise RuntimeError('simulated partial decoder failure')
        return dict(text='invalid JSON intentionally retained', raw_text='invalid JSON',
                    input_tokens=len(ids), output_tokens=1,
                    output_token_ids=[999 if self.bad_stop else 151645], finish_reason='eos')


@pytest.fixture
def stages(tmp_path, monkeypatch):
    cases = [dict(world_key='world-' + str(i), prompt_sha256=str(i) * 64,
                  anchors=[], passages=[], transform={'segments': []}) for i in range(6)]
    plan = dict(protocol={'resource_gates_bytes': GATES, 'output_root': str(tmp_path)}, inventory={'cases': cases},
                messages=[[{'role': 'user', 'content': 'source data ' + str(i)}] for i in range(6)])
    args = SimpleNamespace(output=tmp_path / 'new-run', plan=tmp_path / 'M0', preflight=None,
                           model=tmp_path / 'model', model_identity=tmp_path / 'identity.json',
                           protocol=tmp_path / 'protocol.json', admit_resources=True)
    args.plan.mkdir()
    (args.plan / 'seal.json').write_text('{}', encoding='utf-8')
    args.protocol.write_text('{}', encoding='utf-8')
    data.write_json(args.model_identity, dict(model_id='fake', official_revision='test',
                                             model_files_sha256={'f': 'a' * 64}, runtime={}))
    tokens = [dict(index=i, world_key=cases[i]['world_key'], input_token_ids=[10, 10],
                   input_tokens=2, eligible=True, status='eligible') for i in range(6)]
    monkeypatch.setattr(obs, '_load_plan', lambda _: plan)
    monkeypatch.setattr(obs, 'research_identity', lambda: IDENTITY)
    monkeypatch.setattr(obs, 'completed_identity', lambda value: value)
    return args, plan, tokens


def test_resource_refusal_precedes_tokenization_and_model_load(stages):
    args, _, _ = stages
    backend = FakeInstrument(free_ram=2684354559)
    with pytest.raises(ValueError, match='host RAM'):
        obs.preflight(args, instrument=backend)
    assert backend.events == []
    failure = data.read_json(args.output / 'failure.json')
    assert failure['accounting']['model_requests'] == 0
    data.verify_seal(args.output, required=('failure.json',))


def test_all_six_inputs_are_counted_without_loading_and_no_truncation(stages):
    args, _, _ = stages
    backend = FakeInstrument(lengths=[4097, 2, 2, 2, 2, 2])
    result = obs.preflight(args, instrument=backend)
    assert backend.events == ['tokenize'] * 6
    assert result['tokens'][0]['input_tokens'] == len(result['tokens'][0]['input_token_ids']) == 4097
    assert result['tokens'][0]['eligible'] is False
    assert all(row['eligible'] for row in result['tokens'][1:])
    assert result['actual_model_loads'] == result['model_requests'] == 0


def test_fixed_requests_continue_after_format_unknown_and_raw_is_preserved(stages, monkeypatch):
    args, _, tokens = stages
    monkeypatch.setattr(obs, '_collect_prerequisites', lambda *unused: {'tokens': tokens})
    backend = FakeInstrument()
    result = obs.collect(args, instrument=backend)
    assert backend.events == ['tokenize'] * 6 + ['load'] + ['generate'] * 6
    assert result['accounting']['backend_returns'] == 6
    assert result['accounting']['model_successful_python_returns'] == 6
    assert result['risk_outputs'] == result['annotation_reads'] == 0 and result['auroc'] is None
    for index in range(6):
        child = args.output / ('request-' + str(index).zfill(2))
        assert data.read_json(child / 'backend_raw.json')['text'].startswith('invalid JSON')
        assert data.read_json(child / 'proposal_validation.json')['mechanical_gate_passed'] is False
        data.verify_seal(child, required=('request.json', 'backend_raw.json', 'proposal_validation.json'))
    data.verify_seal(args.output, required=('result.json',))


def test_partial_forward_failure_keeps_real_counts_and_does_not_retry(stages, monkeypatch):
    args, _, tokens = stages
    monkeypatch.setattr(obs, '_collect_prerequisites', lambda *unused: {'tokens': tokens})
    backend = FakeInstrument(fail=True)
    with pytest.raises(RuntimeError, match='partial decoder'):
        obs.collect(args, instrument=backend)
    assert backend.events == ['tokenize'] * 6 + ['load', 'generate']
    failure = data.read_json(args.output / 'failure.json')
    counts = failure['accounting']
    assert counts['backend_attempts'] == 1 and counts['backend_returns'] == 0
    assert counts['model_forward_entry_attempts'] == 2 and counts['model_successful_python_returns'] == 1
    assert failure['partial_generated_tokens'] == 'unknown_without_backend_return'
    assert failure['automatic_retry'] is False
    assert not (args.output / 'request-01').exists()
    data.verify_seal(args.output, required=('failure.json',))


def test_bad_backend_stop_metadata_is_saved_then_stops(stages, monkeypatch):
    args, _, tokens = stages
    monkeypatch.setattr(obs, '_collect_prerequisites', lambda *unused: {'tokens': tokens})
    with pytest.raises(ValueError, match='stop reason'):
        obs.collect(args, instrument=FakeInstrument(bad_stop=True))
    assert data.read_json(args.output / 'request-00/backend_raw.json')['output_token_ids'] == [999]
    assert not (args.output / 'request-01').exists()


def test_resource_refusal_never_overwrites_an_existing_run(stages):
    args, _, _ = stages
    args.output.mkdir()
    marker = args.output / 'old.json'
    marker.write_text('{"retained":true}', encoding='utf-8')
    backend = FakeInstrument()
    with pytest.raises(FileExistsError):
        obs.preflight(args, instrument=backend)
    assert backend.events == [] and json.loads(marker.read_text()) == {'retained': True}


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def tolist(self):
        return [self.values]


class Handle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class FakeModule:
    def __init__(self):
        self.handles = []

    def forward(self, input_ids, logits_to_keep=None):
        return object()

    def register_forward_pre_hook(self, callback, *, with_kwargs):
        assert with_kwargs is True
        handle = Handle()
        self.handles.append(handle)
        return handle

    def register_forward_hook(self, callback, *, with_kwargs, always_call):
        assert with_kwargs is always_call is True
        handle = Handle()
        self.handles.append(handle)
        return handle


def test_hooks_preserve_forward_signature_and_count_prefill_decode():
    import time
    model = FakeModule()
    signature = inspect.signature(model.forward)
    forward = model.forward.__func__
    journal = io.StringIO()
    counter = obs.ForwardAccounting(model, journal, [11, 12], time.monotonic())
    with counter:
        for ids in ([11, 12], [99]):
            kwargs = dict(input_ids=FakeTensor(ids), logits_to_keep=1)
            counter.before(model, (), kwargs)
            counter.after(model, (), kwargs, object())
    assert model.forward.__func__ is forward and inspect.signature(model.forward) == signature
    assert all(handle.removed for handle in model.handles)
    report = counter.report()
    assert report['forward_entry_attempts'] == report['successful_python_returns'] == 2
    assert report['processed_input_positions'] == 3
    assert [row['phase'] for row in report['rows']] == ['prefill', 'decode']
    assert len(journal.getvalue().splitlines()) == 6


def test_prefill_identity_refusal_keeps_attempt_and_removes_hooks():
    import time
    model = FakeModule()
    counter = obs.ForwardAccounting(model, io.StringIO(), [11, 12], time.monotonic())
    with pytest.raises(ValueError, match='prefill token IDs'):
        with counter:
            counter.before(model, (), dict(input_ids=FakeTensor([11, 999]), logits_to_keep=1))
    assert counter.report()['forward_entry_attempts'] == 1
    assert counter.report()['successful_python_returns'] == 0
    assert all(handle.removed for handle in model.handles)


def test_ledger_header_cannot_hide_a_missing_source_world(stages):
    args, plan, _ = stages
    args.preflight = args.output.parent / 'preflight'
    args.preflight.mkdir()
    args.ledger = args.output.parent / 'ledger.json'
    args.model_admission = args.output.parent / 'admission.json'
    data.write_json(args.preflight / 'result.json', dict(completed=True,
        stage='M1_tokenizer_resource_preflight', source_runtime=IDENTITY,
        source_plan_seal_sha256=data.file_hash(args.plan / 'seal.json')))
    obs._seal_tree(args.preflight)
    data.write_json(args.ledger, dict(schema='h020-source-only-obligation-ledger-v1',
        frozen_before_model_outputs=True, model_inputs_contain_R=False,
        source_plan_seal_sha256=data.file_hash(args.plan / 'seal.json'),
        cases=plan['inventory']['cases'][:-1]))
    data.write_json(args.model_admission, {})
    with pytest.raises(ValueError, match='exact six'):
        obs._collect_prerequisites(args, plan, IDENTITY)


def test_output_cannot_escape_the_frozen_unified_data_folder(stages):
    args, plan, _ = stages
    args.output = args.output.parent.parent / 'different-data-root'
    backend = FakeInstrument()
    with pytest.raises(ValueError, match='unified data root'):
        obs.preflight(args, instrument=backend)
    assert backend.events == [] and not args.output.exists()


@pytest.mark.parametrize('field,value', [('stop_strings', ['unexpected']),
    ('cache_implementation', 'static'), ('bad_words_ids', [[123]]), ('future_processor', True)])
def test_unbound_generation_rules_are_rejected(field, value):
    expected = {'repetition_penalty': 1.05, 'stop_strings': None, 'cache_implementation': None,
                'bad_words_ids': None, '_from_model_config': False}
    observed = dict(expected, **{field: value})
    with pytest.raises(ValueError, match='Unbound generation'):
        obs._check_generation_configuration(observed, expected)


def test_only_nonbehavioral_generation_metadata_are_ignored():
    obs._check_generation_configuration({'repetition_penalty': 1.05, '_commit_hash': 'a'},
                                        {'repetition_penalty': 1.05, 'transformers_version': 'version'})
    with pytest.raises(ValueError, match='Unbound generation'):
        obs._check_generation_configuration({'_from_model_config': True}, {'_from_model_config': False})


@pytest.mark.parametrize('stage', ['prepare', 'preflight', 'collect'])
def test_real_cli_dispatch_preserves_top_level_stage(stage, monkeypatch, capsys):
    from experiments import run as cli
    argv = ['transfer', 'observe-source-first-records', '--stage', stage]
    for option in ('source', 'protocol', 'system-prompt', 'schema', 'admission',
                   'review-root', 'model-identity', 'model', 'output'):
        argv += ['--' + option, 'unused-' + option]
    parsed = cli.parser().parse_args(argv)
    assert parsed.stage == 'transfer' and parsed.source_stage == stage
    called = []
    def fake(args):
        called.append(args.source_stage)
        return {'stage': 'fake-source-' + args.source_stage}
    monkeypatch.setattr(obs, 'run', fake)
    import sys
    monkeypatch.setattr(sys, 'argv', ['token-experiment', *argv])
    cli.main()
    assert called == [stage]
    assert 'Source-only proposal stage: fake-source-' + stage in capsys.readouterr().out
