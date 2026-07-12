"""Weight-version boundary tests: abort retry and replica barrier."""

import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from opd_v2.data_plane.clients import RolloutClient
from opd_v2.orchestrator import Orchestrator


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return self.payload


class _Session:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.posts = 0

    def post(self, *args, **kwargs):
        del args, kwargs
        self.posts += 1
        return _Response(self.payloads.pop(0))


class _Trainer:
    async def save(self):
        return {"path": "/weights/v1", "weight_version": 1}


class _Replica:
    def __init__(self, events, name, success=True):
        self.events = events
        self.name = name
        self.success = success

    async def pause_generation(self, mode):
        self.events.append((self.name, "pause", mode))
        await asyncio.sleep(0)

    async def update_weights_from_disk(self, *args, **kwargs):
        del args, kwargs
        self.events.append((self.name, "reload"))
        await asyncio.sleep(0)
        return {"success": self.success, "message": "injected rejection"}

    async def continue_generation(self):
        self.events.append((self.name, "resume"))


async def _run():
    session = _Session(
        [
            {
                "output_ids": [11],
                "meta_info": {"finish_reason": {"type": "abort"}},
            },
            {
                "output_ids": [22],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "weight_version": "1",
                },
            },
        ]
    )
    client = RolloutClient(session, "http://rollout")
    output, version, reason = await client.generate_one(
        [1, 2],
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=4,
        ignore_eos=False,
    )
    assert session.posts == 2
    assert (output, version, reason) == ([22], 1, "stop")

    events = []
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.trainer = _Trainer()
    orchestrator.rollout_clients = [
        _Replica(events, "a"),
        _Replica(events, "b"),
    ]
    orchestrator.weight_version = 0
    orchestrator._last_sync_s = 0
    orchestrator._last_sync_ok = 0
    assert await orchestrator.weight_sync() == 1
    first_reload = min(i for i, event in enumerate(events) if event[1] == "reload")
    assert sum(event[1] == "pause" for event in events[:first_reload]) == 2
    assert all(event[2] == "abort" for event in events if event[1] == "pause")

    events = []
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.trainer = _Trainer()
    orchestrator.rollout_clients = [
        _Replica(events, "a"),
        _Replica(events, "b", success=False),
    ]
    orchestrator.weight_version = 0
    orchestrator._last_sync_s = 0
    orchestrator._last_sync_ok = 0
    try:
        await orchestrator.weight_sync()
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial replica reload was accepted")
    assert orchestrator.weight_version == 0
    assert not any(event[1] == "resume" for event in events)


if __name__ == "__main__":
    asyncio.run(_run())
    print("weight sync barrier tests passed")
