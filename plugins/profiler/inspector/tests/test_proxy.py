#!/usr/bin/env python3
"""Run Proxy timing and lifecycle regressions against the production code."""
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from inspector_json_reader import DumpContext


def complete(records, n, nested=True, steps=True):
    assert len({r['record_sn'] for r in records}) == len(records)
    assert [r['record_sn'] for r in records] == sorted(r['record_sn'] for r in records)
    final = records[-1]
    assert final['record_type'] == 'proxy_op'
    assert final['n_steps_completed'] == n and final['n_steps_dropped'] == 0
    assert final['trans_size_bytes'] == n * 100 and final['total_us'] > 0
    assert final['op_bw_gbs'] >= 0
    children = []
    if nested:
        assert records[0]['record_type'] == 'proxy_op_start'
        assert records[0]['is_final'] is False and final['is_final'] is True
        assert 'total_us' not in records[0] and 'op_bw_gbs' not in records[0]
        for r in records[1:-1]:
            assert r['record_type'] == 'proxy_op_segment' and r['is_final'] is False
            assert 'total_us' not in r and 'op_bw_gbs' not in r
        if steps:
            for r in records[1:]:
                children += r['steps']
            assert len(children) == n
        else:
            assert len(records) == 2 and 'steps' not in final
        for s in children:
            assert not ({'rank','peer','channel_id','parent_sn','parent_type','direction','proxy_op_sn','record_sn'} & set(s))
    else:
        children = records[:-1]
        assert all(r['record_type'] == 'proxy_step' and 'rank' not in r for r in children)
        assert not any('is_final' in r for r in records)
        assert len(children) == n
    assert len({s['proxy_step_sn'] for s in children}) == len(children)
    assert [s['step'] for s in children] == list(range(len(children)))
    for s in children:
        assert s['trans_size_bytes'] == 100 and s['wire_us'] >= 0
    return final



FIELDS = {"total_us", "gpu_produce_us", "peer_credit_us", "wire_us",
          "flush_us", "gpu_consume_us", "op_bw_gbs"}


def verify_timing(name, record, trace_sn):
    assert record["timing_source"] == "proxy_cpu", name
    is_op = record["record_type"] == "proxy_op"
    assert ("rank" in record) == is_op, name
    send = record["direction"] == "send"
    if is_op:
        states = ["proxy_op_start", "proxy_op_in_progress", "proxy_op_stop"]
        times = [100, 150, 300]
        mask = int(name.rsplit("_", 1)[1]) if "_mask_" in name else 7
        pairs = {"total_us": (0, 2)}
        if name == "op_zero_duration": times[-1] = 100
        if name == "op_reversed": times[-1] = 99
        if name == "op_zero_start": times[0] = 0
    else:
        phases = ["send_gpu_wait", "send_peer_wait", "send_wait"] if send else [
            "recv_wait", "recv_flush_wait", "recv_gpu_wait"]
        states = ["proxy_step_start"] + phases + ["proxy_step_stop"]
        times = [100, 110, 130, 160, 200]
        mask = int(name.rsplit("_", 1)[1]) if "_mask_" in name else 31
        pairs = {"total_us": (0, 4)}
        keys = ["gpu_produce_us", "peer_credit_us", "wire_us"] if send else [
            "wire_us", "flush_us", "gpu_consume_us"]
        pairs.update({key: (i + 1, i + 2) for i, key in enumerate(keys)})
        if name.endswith("zero_time"): times = [0] * 5
        elif name.endswith("phase_reversed"): times[2] = 105
        elif name.endswith("reversed"): times[-1] = 90
    expected_ts = {state + "_ts": times[i] for i, state in enumerate(states) if mask & (1 << i)}
    assert record["event_trace_ts"] == expected_ts, (name, record)
    assert ("event_trace_sn" in record) == trace_sn, name
    if trace_sn:
        for i, state in enumerate(states):
            assert record["event_trace_sn"][state + "_sn"] == (i + 1 if mask & (1 << i) else 0), name
    expected = {key: times[b] - times[a] for key, (a, b) in pairs.items()
                if mask & (1 << a) and mask & (1 << b) and times[b] >= times[a]}
    if is_op and expected.get("total_us", 0) > 0:
        expected["op_bw_gbs"] = record["trans_size_bytes"] / (expected["total_us"] * 1000)
    assert set(record) & FIELDS == set(expected), (name, record, expected)
    for key, value in expected.items():
        if key == "op_bw_gbs":
            assert math.isfinite(record[key]), name
            assert math.isclose(record[key], value, rel_tol=1e-6, abs_tol=0.00000051), (name, record)
        else:
            assert record[key] == value, (name, record, expected)


def verify_segments(cases, stream):
    for n in (0,1,2,3,4,5,7,8,9,15,16,17,32,33,512):
        complete(cases[f'nested_{n}'],n)
        assert len(cases[f'nested_{n}']) == 2  # same-dump segments merged
    assert len(cases['start_visible']) == 1 and cases['start_visible'][0]['record_type']=='proxy_op_start'
    assert len(cases['stalled_tail']) == 1 and len(cases['stalled_tail'][0]['steps'])==16
    assert len(cases['stalled_final']) == 1 and len(cases['stalled_final'][0]['steps'])==1
    complete(cases['start_visible']+cases['stalled_tail']+cases['stalled_final'],17)
    complete(cases['stop_before_child'],16)
    assert len(cases['stop_before_child'])==2
    complete(cases['op_only'],33,steps=False)
    complete(cases['flat'],33,nested=False)
    assert len(cases['overwrite'])==1 and cases['overwrite'][0]['steps']==[]
    allocation_loss = cases['allocation_loss'][-1]
    assert allocation_loss['n_steps_completed']==3 and allocation_loss['n_steps_dropped']==1
    assert allocation_loss['trans_size_bytes']==300
    assert [s['step'] for s in allocation_loss['steps']]==[1,2]
    context=DumpContext()
    restored=[context.restore(json.loads(line)) for line in stream.read_text().splitlines()]
    context.finish()
    markers=[r['dump_stats'] for r in restored if 'dump_stats' in r]
    assert [m['proxy_records'] for m in markers]==[2,2,1,4]
    payloads=[r['proxy_trace'] for r in restored if 'proxy_trace' in r]
    assert [len(r.get('steps',[])) for r in payloads[:5]]==[0,33,0,16,1]
    assert [r['is_final'] for r in payloads[:5]]==[False,True,False,False,True]
    complete(payloads[5:],3,nested=False)


def main():
    with tempfile.TemporaryDirectory(prefix='nccl-proxy-') as tmp:
        binary = Path(tmp)/'test'
        output = Path(tmp)/'records.jsonl'
        stream = Path(tmp)/'stream.jsonl'
        cmd = shlex.split(os.environ.get('CXX','g++')) + [
            '-std=c++14','-O2','-Wall','-Wextra','-ffunction-sections','-fdata-sections',
            '-I'+str(ROOT),'-I'+str(ROOT/'nccl'),str(ROOT/'tests/proxy_test.cc'),
            str(ROOT/'json.cc'),str(ROOT/'inspector_ring.cc'),str(ROOT/'inspector_event_pool.cc'),
            '-Wl,--gc-sections','-pthread','-o',str(binary)]
        cmd += shlex.split(os.environ.get('TEST_CXXFLAGS',''))
        subprocess.run(cmd,check=True)
        outputs = []
        for trace_sn in (False, True):
            subprocess.run([str(binary), str(output), str(stream), str(int(trace_sn))],
                           check=True, timeout=30)
            cases = {x['case']: x['records'] for x in map(json.loads, output.read_text().splitlines())}
            timing = {name[7:]: rows[0] for name, rows in cases.items() if name.startswith('timing_')}
            assert len(timing) == 82
            for name, record in timing.items(): verify_timing(name, record, trace_sn)
            verify_segments(cases, stream)
            outputs.append(timing)
        for name, enabled in outputs[1].items():
            del enabled['event_trace_sn']
            assert outputs[0][name] == enabled, name
        print('PASS: 164 timing fixtures; lifecycle with SN off/on; lazy growth/failure/recycling; '
              'flat/op-only; FIFO parent retention; grouped dump counts')


if __name__ == '__main__': main()
