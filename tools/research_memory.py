"""Compact research handoff and content-addressed experiment deduplication.

This tool does not implement methods, open annotations or launch jobs. Central
experiment commands remain responsible for scoring, sealing and evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import uuid


IDENTITY_FIELDS = ('stage', 'method', 'code_sha256',
                   'data_sha256', 'observer', 'population', 'evaluation', 'seed')
OUTCOMES = {'supported_scope', 'refuted', 'inconclusive', 'engineering_failure'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(spec):
    """Ignore prose/output paths; bind only explicitly declared run semantics."""
    if not isinstance(spec, dict) or any(k not in spec for k in (*IDENTITY_FIELDS, 'hypothesis_id', 'intervention')):
        raise ValueError('Incomplete experiment identity')
    if any(not isinstance(spec[k], str) or not spec[k].strip()
           for k in ['hypothesis_id', 'intervention', 'stage']):
        raise ValueError('Stable hypothesis and intervention IDs required')
    for key in ['code_sha256', 'data_sha256']:
        bindings = spec[key]
        if (not isinstance(bindings, dict) or not bindings
                or any(not isinstance(k, str) or not k or not isinstance(v, str)
                       or re.fullmatch('[0-9a-f]{64}', v) is None
                       for k, v in bindings.items())):
            raise ValueError('Nonempty explicit code/data SHA256 bindings required')
    for key in ['method', 'observer', 'population', 'evaluation']:
        if not isinstance(spec[key], dict) or not spec[key]:
            raise ValueError('Explicit method, observer, population and evaluation required')
    method = spec['method']
    if (not isinstance(method.get('entrypoint'), str) or not method['entrypoint']
            or not isinstance(method.get('config'), dict)):
        raise ValueError('Method requires its executable entrypoint and complete frozen config')
    if spec['seed'] is not None and (type(spec['seed']) is not int or spec['seed'] < 0):
        raise ValueError('Seed must be null or a nonnegative integer')
    bound = {k: spec[k] for k in IDENTITY_FIELDS}
    bound['method'] = {k: method[k] for k in ['entrypoint', 'config']}
    encoded = json.dumps(bound, sort_keys=True,
                         separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _path_key(path):
    """Compare resolved Windows aliases without changing any I/O path."""
    if isinstance(path, PureWindowsPath):
        text = str(path)
        if text[:8].casefold() == '\\\\?\\unc\\':
            ordinary = PureWindowsPath('\\\\' + text[8:])
            if ordinary.is_absolute():
                return ordinary
        elif re.match(r'^\\\\\?\\[A-Za-z]:\\', text):
            return PureWindowsPath(text[4:])
    return path


def _record_path(project, run_id):
    if re.fullmatch('[0-9a-f]{64}', run_id) is None:
        raise ValueError('Run ID must be its full experiment SHA256')
    project = Path(project).resolve()
    path = (project/'research-state'/'runs'/run_id).resolve()
    if not _path_key(path).is_relative_to(_path_key(project)):
        raise ValueError('Experiment bookkeeping escapes the project')
    return path


def _read_claim(directory, run_id):
    value = read(directory/'claim.json')
    if (not isinstance(value, dict) or value.get('schema') != 1
            or value.get('run_id') != run_id
            or fingerprint(value.get('spec')) != run_id):
        raise ValueError('Stored experiment claim identity changed; recover the record')
    return value


def _artifact_path(project, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or PurePosixPath(relative).is_absolute() or Path(relative).drive
            or any(part in {'', '.', '..'} for part in relative.split('/'))):
        raise ValueError('Evidence paths must be contained, canonical project-relative paths')
    project = Path(project).resolve()
    path = (project/relative).resolve()
    if not _path_key(path).is_relative_to(_path_key(project)):
        raise ValueError('Evidence escapes the project')
    if _path_key(path).relative_to(_path_key(project)).as_posix().startswith('research-state/runs/'):
        raise ValueError('Run bookkeeping cannot certify itself as scientific evidence')
    return path


def inspect_run(project, spec):
    run_id = fingerprint(spec)
    directory = _record_path(project, run_id)
    if not directory.exists():
        return {'action': 'new', 'run_id': run_id}
    if not (directory/'claim.json').exists():
        return {'action': 'reserved', 'run_id': run_id}
    _read_claim(directory, run_id)
    if not (directory/'result.json').exists():
        return {'action': 'already_claimed', 'run_id': run_id,
                'note': 'Reservation exists; process liveness is not inferred'}
    result = read(directory/'result.json')
    if (not isinstance(result, dict) or result.get('schema') != 1
            or result.get('run_id') != run_id or not isinstance(result.get('outcome'), str)
            or result['outcome'] not in OUTCOMES
            or not isinstance(result.get('conclusion'), str) or not result['conclusion'].strip()
            or not isinstance(result.get('artifacts'), list) or not result['artifacts']):
        raise ValueError('Stored experiment result identity/schema changed; recover the record')
    paths = []
    for item in result['artifacts']:
        if (not isinstance(item, dict) or not isinstance(item.get('sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', item['sha256']) is None):
            raise ValueError('Invalid stored evidence digest')
        paths.append((_artifact_path(project, item.get('path')), item))
    missing = []
    changed = []
    for path, item in paths:
        if not path.is_file():
            missing.append(item['path'])
        elif file_hash(path) != item['sha256']:
            changed.append(item['path'])
    if missing or changed:
        return {'action': 'artifact_recovery_required', 'run_id': run_id,
                'missing': missing, 'changed': changed,
                'rule': 'Recover evidence; do not silently rerun or reinterpret the result'}
    return {'action': 'reuse', 'run_id': run_id, 'outcome': result['outcome'],
            'conclusion': result['conclusion'], 'artifacts': result['artifacts']}


def claim(project, spec):
    """Atomic directory reservation blocks concurrent duplicate launchers."""
    status = inspect_run(project, spec)
    if status['action'] != 'new':
        return status
    directory = _record_path(project, status['run_id'])
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError:
        return inspect_run(project, spec)
    write_new(directory/'claim.json', {'schema': 1, 'created_utc': now(), 'spec': spec,
              'run_id': status['run_id'], 'phase': 'reserved_not_launched'})
    return {'action': 'claimed', 'run_id': status['run_id']}


def finish(project, run_id, outcome, conclusion, artifacts):
    project = Path(project).resolve()
    directory = _record_path(project, run_id)
    if not (directory/'claim.json').is_file():
        raise ValueError('An experiment must be claimed before completion')
    _read_claim(directory, run_id)
    if (outcome not in OUTCOMES or not isinstance(conclusion, str)
            or not conclusion.strip() or not artifacts):
        raise ValueError('Outcome, nonempty conclusion and evidence artifacts required')
    records = []
    for artifact in artifacts:
        path = Path(artifact).resolve()
        if not _path_key(path).is_relative_to(_path_key(project)) or not path.is_file():
            raise ValueError('Evidence must be a real file inside the project')
        relative = _path_key(path).relative_to(_path_key(project)).as_posix()
        _artifact_path(project, relative)
        records.append({'path': relative, 'sha256': file_hash(path)})
    value = {'schema': 1, 'run_id': run_id, 'completed_utc': now(), 'outcome': outcome,
             'conclusion': conclusion, 'artifacts': records}
    write_new(directory/'result.json', value)
    return value


def compact_context(project):
    """Return the current question and authoritative indices, not old transcripts."""
    root = Path(project).resolve()/'research-state'
    state = read(root/'state.json')
    return {'objective': state['objective'], 'phase': state['phase'],
            'next_question': state['next_question'],
            'active_hypothesis': state.get('active_hypothesis'),
            'known_fact_ids': state['known_fact_ids'],
            'do_not_repeat': state['do_not_repeat'],
            'unresolved_assumptions': state['unresolved_assumptions'],
            'read_first': ['research-state/RESUME.md', 'research-state/facts.json',
                           'research-state/prior_experiments.json'],
            'rule': 'A known phenomenon needs a new discriminating intervention, not another description'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['check', 'claim']:
        command = commands.add_parser(name)
        command.add_argument('--spec', type=Path, required=True)
    commands.add_parser('context')
    done = commands.add_parser('finish')
    done.add_argument('--run-id', required=True)
    done.add_argument('--outcome', choices=sorted(OUTCOMES), required=True)
    done.add_argument('--conclusion', required=True)
    done.add_argument('--artifacts', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    if args.command in {'check', 'claim'}:
        result = (inspect_run if args.command == 'check' else claim)(args.project, read(args.spec))
    elif args.command == 'finish':
        result = finish(args.project, args.run_id, args.outcome, args.conclusion, args.artifacts)
    else:
        result = compact_context(args.project)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
