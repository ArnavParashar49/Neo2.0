"""``python -m neo2 fetch`` (neo2/core/fetch.py): what NEO downloads, only when asked — the plan, then
each item's progress as JSON lines (and nothing else on stdout), what's here already never asked for
again, failures in fixed words with a code, a stop that stops; each item the core's own loader at
its pins, measured where it lands; the core's own downloads one group at a time, in a child process.

Hermetic: every download, cache and program here is the test's own (fake loaders, a fake Hugging
Face cache in tmp_path, a fake `playwright install`); nothing reaches the network."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path

import pytest

from neo2.core import config, fetch, models
from neo2.core.fetch import Item, Job
from neo2.core.fetch import chromium_here as REAL_CHROMIUM_HERE  # (conftest fakes the module's…)
from neo2.core.voice.pipeline import local_ready as REAL_LOCAL_READY  # (…and this one)


def _item(id_="a", group="essential", size=1000, *, here=False, get=None, skip="", where=None) -> Item:
    """An item that is here once its `get` returned (the way a real download leaves it)."""
    there = [here]

    def default(job: Job) -> None:
        for n in (250, 500, 1000):
            job.at(n)
            time.sleep(0.02)

    def fetched(job: Job) -> None:
        (get or default)(job)
        there[0] = True

    return Item(id_, group, f"Item {id_}", "the test", size, lambda: there[0], fetched, skip=skip, where=where)


def _events(items, groups=("essential",), **kw):
    out: list[dict] = []
    code = fetch.run(groups, out.append, known=items, poll_s=0.01, **kw)
    return code, out


# ---- the plan, the progress, the end ------------------------------------------------------------------


def test_the_plan_then_each_item_then_done(monkeypatch):
    monkeypatch.setattr(fetch, "_EMIT_EVERY_S", 0.0)
    got: list[str] = []

    def get(job: Job) -> None:
        job.sized(800)  # the server said the exact size
        for n in (200, 400, 800):
            job.at(n)
            time.sleep(0.03)
        got.append("fetched")

    items = [_item("here", here=True), _item("new", get=get), _item("off", skip="NEO_REFLEX is off"),
             _item("voice", group="local_voice")]  # fmt: skip
    code, ev = _events(items)
    assert code == 0 and got == ["fetched"]
    plan = ev[0]
    assert plan["event"] == "plan" and [i["id"] for i in plan["items"]] == ["here", "new", "off"]  # (its group only)
    assert plan["size"] == 1000  # what will be downloaded: only what isn't here or skipped
    assert {i["id"]: (i["here"], i["skip"]) for i in plan["items"]} == {
        "here": (True, None), "new": (False, None), "off": (False, "NEO_REFLEX is off")}  # fmt: skip
    kinds = [(e["event"], e.get("id"), e.get("state")) for e in ev[1:]]
    assert kinds[0] == ("item", "here", "here") and kinds[1] == ("start", "new", None)
    assert ("item", "new", "done") in kinds and ("item", "off", "skipped") in kinds
    progress = [e for e in ev if e["event"] == "progress"]
    assert progress and progress[-1]["done"] == progress[-1]["total"] == 800  # exact once said
    assert progress[-1]["all_done"] == progress[-1]["all_total"] == 800
    assert all(p["done"] <= p["total"] for p in progress)
    assert ev[-1]["event"] == "done" and ev[-1]["ok"] is True and ev[-1]["failed"] == []


def test_status_says_what_is_here_and_downloads_nothing():
    touched: list[str] = []
    items = [_item("a", here=True), _item("b", get=lambda job: touched.append("b"))]
    code, ev = _events(items, status=True)
    assert code == 0 and touched == []
    assert [e["event"] for e in ev] == ["plan", "done"] and ev[-1]["ok"] is False  # (b isn't here)


def test_a_failure_is_said_with_a_code_and_the_rest_still_comes():
    def offline(job: Job) -> None:
        import httpx

        raise httpx.ConnectError("[Errno 8] nodename nor servname provided")

    def full(job: Job) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    def damaged(job: Job) -> None:
        raise RuntimeError("the wake-word model download didn't match its checksum — not using it")

    items = [_item("a", get=offline), _item("b", get=full), _item("c", get=damaged), _item("d")]
    code, ev = _events(items)
    assert code == 1
    failed = {e["id"]: e["code"] for e in ev if e["event"] == "item" and e["state"] == "failed"}
    assert failed == {"a": "offline", "b": "disk_full", "c": "checksum"}
    assert ("item", "d", "done") in [(e["event"], e.get("id"), e.get("state")) for e in ev]
    assert ev[-1] == {**ev[-1], "event": "done", "ok": False, "failed": ["a", "b", "c"]}
    words = {e["id"]: e["message"] for e in ev if e["event"] == "item" and e["state"] == "failed"}
    assert words["a"] == "Can't reach the server — check the internet connection." and "nodename" not in words["a"]


def test_an_item_that_isnt_complete_after_its_download_failed():
    item = Item("a", "essential", "Item a", "the test", 10, lambda: False, lambda job: None)
    code, ev = _events([item])  # it "downloaded", yet it still isn't here
    assert code == 1 and ev[-2]["state"] == "failed" and ev[-2]["code"] == "failed"


def test_too_little_disk_is_said_before_anything_downloads(tmp_path, monkeypatch):
    class Usage:
        free = 10 << 20

    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda p: Usage())
    touched: list[str] = []
    code, ev = _events([_item("big", size=5 << 30, get=lambda j: touched.append("x"), where=tmp_path / "not" / "yet")])
    assert code == 1 and touched == []
    assert ev[1]["code"] == "disk_full" and "GB free space is needed" in ev[1]["message"]


def test_a_stop_stops_now():
    stop = threading.Event()
    started = threading.Event()

    def slow(job: Job) -> None:
        started.set()
        time.sleep(5)

    def stopper():
        started.wait(2)
        stop.set()

    threading.Thread(target=stopper, daemon=True).start()
    t0 = time.monotonic()
    code, ev = _events([_item("a", get=slow), _item("b")], stop=stop)
    assert code == 130 and time.monotonic() - t0 < 3
    assert ev[-1]["event"] == "done" and ev[-1]["cancelled"] is True and ev[-1]["failed"] == ["a"]


def test_an_estimate_that_was_low_grows_with_the_download():
    job = Job(_item(size=100))
    job.at(150)
    assert job.done() == 150 and job.total == 150
    job.sized(400)
    job.at(200)
    assert job.done() == 200 and job.total == 400 and job.exact


def test_failures_in_fixed_words():
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    assert fetch.failure(OSError(errno.ENOSPC, "x"))[0] == "disk_full"
    assert fetch.failure(OfflineModeIsEnabled("off"))[0] == "offline"
    assert fetch.failure(LocalEntryNotFoundError("not cached and no connection"))[0] == "offline"
    assert fetch.failure(RuntimeError("404 Client Error: Not Found for url"))[0] == "not_found"
    try:
        try:
            raise TimeoutError("read timed out")
        except TimeoutError as inner:
            raise RuntimeError("download failed") from inner
    except RuntimeError as e:
        assert fetch.failure(e)[0] == "offline"  # (the cause counts)
    code, words = fetch.failure(ValueError("something odd\nsecond line"))
    assert code == "failed" and words == "something odd"


# ---- the command line -----------------------------------------------------------------------------------


def test_json_lines_are_all_stdout_says(monkeypatch, capfd):
    def noisy(job: Job) -> None:
        print("a library's chatter")  # (lands on stderr)
        os.write(1, b"a C library writing to fd 1\n")
        job.at(10)

    monkeypatch.setattr(fetch, "items", lambda s=None: [_item("a", size=10, get=noisy), _item("b", here=True)])
    assert fetch.main(["--json"]) == 0
    out, err = capfd.readouterr()
    lines = [json.loads(line) for line in out.splitlines()]  # every line of stdout is an event
    assert [e["event"] for e in lines][0] == "plan" and lines[-1]["event"] == "done" and lines[-1]["ok"]
    assert "chatter" in err and "C library" in err and "chatter" not in out
    print("stdout is stdout again")
    assert "stdout is stdout again" in capfd.readouterr().out


def test_the_groups_and_bad_arguments(monkeypatch, capfd):
    asked: list = []
    monkeypatch.setattr(fetch, "run", lambda groups, emit, **kw: asked.append((list(groups), kw["status"])) or 0)
    assert fetch.main([]) == 0 and fetch.main(["--local-voice", "--browser", "--status"]) == 0
    assert asked == [(["essential"], False), (["local_voice", "browser"], True)]
    assert fetch.main(["--everything"]) == 2
    capfd.readouterr()


def test_python_m_neo2_fetch_is_the_cli(monkeypatch):
    import sys

    import neo2.core
    from neo2.core import logredact
    from neo2.core import main as main_mod

    seen: list = []
    monkeypatch.setattr(fetch, "main", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(logredact, "install", lambda: False)  # (process-wide: not in a test)
    monkeypatch.setattr(neo2.core, "log_to_stdout", lambda: None)
    monkeypatch.setattr(sys, "argv", ["neo2", "fetch", "--vision", "--json"])
    with pytest.raises(SystemExit) as e:
        main_mod.main()
    assert e.value.code == 0 and seen == [["--vision", "--json"]]


# ---- the items: the core's own loaders at their pins ---------------------------------------------------


def test_every_item_by_group_as_the_settings_load_it(monkeypatch):
    monkeypatch.setattr(fetch, "_playwright_dir", lambda: Path("/nowhere"))
    got = {it.id: it for it in fetch.items()}
    assert {g: [i for i, it in got.items() if it.group == g] for g in fetch.GROUPS} == {
        "essential": ["laya", "decide", "embedder", "wake"],
        "local_voice": ["stt", "tts", "voice_pack"],
        "browser": ["chromium"],
        "vision": ["dfine"],
    }
    assert got["laya"].size == fetch.LAYA_BYTES["multilingual"] and "multilingual" in got["laya"].label
    assert sum(got[i].size for i in ("stt", "tts", "voice_pack")) == fetch.LOCAL_VOICE_BYTES
    # NEO_REFLEX off (conftest): no Laya to fetch; the rest of the essentials still come.
    assert got["laya"].skip == got["decide"].skip == "NEO_REFLEX is off"
    assert not got["embedder"].skip and not got["wake"].skip
    monkeypatch.setenv("NEO_REFLEX", "laya")
    monkeypatch.setenv("NEO_REFLEX_ENCODER", "english")
    monkeypatch.setenv("NEO_EMBEDDINGS", "off")
    monkeypatch.setattr(config, "_settings", None)
    got = {it.id: it for it in fetch.items()}
    assert not got["laya"].skip and got["laya"].size == fetch.LAYA_BYTES["english"]
    assert got["embedder"].skip == "NEO_EMBEDDINGS is off"
    monkeypatch.setattr(fetch, "_playwright_dir", lambda: None)
    assert {it.id: it for it in fetch.items()}["chromium"].skip == "the playwright package isn't installed"


def _blob(blobs: Path, name: str, size: int) -> None:
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / name).write_bytes(b"x" * size)


def test_a_hugging_face_item_is_measured_in_its_cache_folder(tmp_path, monkeypatch):
    """The loader downloads; the bytes are read where the hub writes them (finished blobs and the
    parts still downloading), against the size a dry run of the same call lists."""
    import huggingface_hub

    cache = tmp_path / "hub"
    blobs = cache / "models--o--r" / "blobs"
    _blob(blobs, "old", 50)  # (already here before: not this download's)

    class Info:
        def __init__(self, size, will):
            self.file_size, self.will_download = size, will

    dry: list = []

    def snapshot_download(repo, **kw):
        assert kw.get("dry_run") is True and kw["allow_patterns"] == ["w/*"] and kw["revision"] == "r1"
        dry.append(repo)
        return [Info(300, True), Info(50, False)]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    seen: list[int] = []
    job = Job(_item(size=999))

    def load():
        _blob(blobs, "a.1234.incomplete", 120)
        seen.append(job.done())
        os.rename(blobs / "a.1234.incomplete", blobs / "a")
        _blob(blobs, "b.99.incomplete", 30)
        seen.append(job.done())

    fetch._hf("o/r", load, revision="r1", patterns=["w/*"], cache_dir=cache)(job)
    assert dry == ["o/r"] and job.total == 300 and job.exact
    assert seen == [120, 150]


def test_parts_left_by_a_download_that_died_long_ago_are_swept(tmp_path):
    blobs = tmp_path / "blobs"
    _blob(blobs, "stale.1.incomplete", 10)
    _blob(blobs, "live.2.incomplete", 10)
    _blob(blobs, "done", 10)
    old = time.time() - 4000
    os.utime(blobs / "stale.1.incomplete", (old, old))
    assert fetch._sweep(blobs) == 1
    assert sorted(p.name for p in blobs.iterdir()) == ["done", "live.2.incomplete"]


def test_the_wake_word_reports_its_bytes(tmp_path, monkeypatch):
    import httpx

    from neo2.core.voice import wake

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{wake._MODEL}/am/final.mdl", "m")
        z.writestr(f"{wake._MODEL}/conf/model.conf", "c")
    data = buf.getvalue()

    class Resp:
        headers = {"content-length": str(len(data))}

        def raise_for_status(self):
            pass

        def iter_bytes(self, n):
            yield data[:10]
            yield data[10:]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(wake, "_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: Resp())
    assert not wake.model_here()
    job = Job(_item(size=1))
    fetch._wake(job)
    assert wake.model_here() and job.total == len(data) and job.exact and job.done() == len(data)


def _fake_hub(root: Path, repo: str, rev: str, files: dict[str, bytes]) -> Path:
    snap = root / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    for name, data in files.items():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_bytes(data)
    return snap


def test_a_model_is_here_only_with_its_config_and_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    repo = "mlx-community/parakeet-tdt-0.6b-v3"
    rev = models.revision(repo)
    assert not models.here(repo, cache_dir=str(tmp_path))
    snap = _fake_hub(tmp_path, repo, rev, {"config.json": b"{}"})
    assert not models.here(repo, cache_dir=str(tmp_path))  # no weights yet
    (snap / "model.safetensors").write_bytes(b"w")
    assert models.here(repo, cache_dir=str(tmp_path))
    os.symlink(tmp_path / "gone", snap / "vocab.txt")  # a link to a blob that isn't there
    assert not models.here(repo, cache_dir=str(tmp_path))
    folder = tmp_path / "mine"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "kokoro.safetensors").write_text("w")
    assert models.here(str(folder))  # a model the user keeps in a folder of their own


def test_laya_is_here_only_as_a_whole_checkpoint(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    files = {"multilingual/rl_agent_config.json": b"{}", "multilingual/model.safetensors": b"w",
             "multilingual/encoder/config.json": b"{}"}  # fmt: skip
    _fake_hub(tmp_path, models.LAYA, models.LAYA_REVISION, files)
    assert not models.laya_here("multilingual")  # no tokenizer
    _fake_hub(tmp_path, models.LAYA, models.LAYA_REVISION, {"multilingual/tokenizer/tokenizer.json": b"{}"})
    assert models.laya_here("multilingual") and not models.laya_here("english")
    assert models.laya_patterns("multilingual")[0] == "multilingual/rl_agent_config.json"


def test_the_on_device_voice_is_ready_only_when_both_models_are(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(models, "here", lambda m, **kw: calls.append(m) or m.endswith("bf16"))
    assert REAL_LOCAL_READY() is False
    assert calls == ["mlx-community/parakeet-tdt-0.6b-v3"]  # (the first one missing: no need to look on)
    monkeypatch.setattr(models, "here", lambda m, **kw: True)
    assert REAL_LOCAL_READY() is True

    def broken(m, **kw):
        raise OSError("the cache can't be read")

    monkeypatch.setattr(models, "here", broken)
    assert REAL_LOCAL_READY() is False


# ---- Playwright's Chromium -------------------------------------------------------------------------------


def _playwright(tmp_path: Path, revs: dict[str, str]) -> Path:
    pkg = tmp_path / "playwright"
    (pkg / "driver" / "package").mkdir(parents=True)
    (pkg / "driver" / "package" / "browsers.json").write_text(
        json.dumps({"browsers": [{"name": n, "revision": r} for n, r in revs.items()]})
    )
    return pkg


def test_chromium_is_here_by_playwrights_own_marker(tmp_path, monkeypatch):
    pkg = _playwright(tmp_path, {"chromium": "1243", "chromium-headless-shell": "1243"})
    monkeypatch.setattr(fetch, "_playwright_dir", lambda: pkg)
    browsers = tmp_path / "ms-playwright"
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(browsers))
    assert fetch.browsers_dir() == browsers
    assert not REAL_CHROMIUM_HERE() and not REAL_CHROMIUM_HERE(shell_ok=True)
    (browsers / "chromium_headless_shell-1243").mkdir(parents=True)
    (browsers / "chromium_headless_shell-1243" / "INSTALLATION_COMPLETE").touch()
    assert not REAL_CHROMIUM_HERE() and REAL_CHROMIUM_HERE(shell_ok=True)  # enough for a headless page only
    (browsers / "chromium-1243").mkdir()
    assert not REAL_CHROMIUM_HERE()  # half-unpacked: not yet
    (browsers / "chromium-1243" / "INSTALLATION_COMPLETE").touch()
    assert REAL_CHROMIUM_HERE()
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH")
    assert fetch.browsers_dir() == Path.home() / "Library" / "Caches" / "ms-playwright"


def test_chromium_comes_from_playwrights_installer_without_the_shell(tmp_path, monkeypatch):
    """`playwright install chromium --no-shell`, measured in its own temporary folder; its size from
    its own progress lines; the zip's unpacking is a stage."""
    ran: list = []

    class Proc:
        returncode = 0

        def __init__(self, cmd, **kw):
            ran.append((cmd, kw["env"]["TMPDIR"]))
            tmp = Path(kw["env"]["TMPDIR"]) / "playwright-download-x"
            tmp.mkdir()
            (tmp / "chrome.zip").write_bytes(b"z" * 1000)
            self.stdout = iter([
                "Downloading Chrome for Testing 153.0 (playwright build v1243) from https://cdn.playwright.dev/x\n",
                "|                                                                                |   0% of 150 MiB\n",
                "|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■| 100% of 150 MiB\n",
                "Chrome for Testing 153.0 (playwright build v1243) downloaded to /x/chromium-1243\n",
                "Downloading FFmpeg (playwright build v1011) from https://cdn.playwright.dev/y\n",
                "|                                                                                |   0% of 1 MiB\n",
            ])  # fmt: skip

        def wait(self):
            return 0

    monkeypatch.setattr(fetch.subprocess, "Popen", Proc)
    job = Job(_item(size=1))
    fetch._chromium(job)
    (cmd, tmpdir), = ran
    assert cmd[-5:] == ["-m", "playwright", "install", "chromium", "--no-shell"]
    assert job.total == 151 * (1 << 20) and job.stage == "" and not Path(tmpdir).exists()

    class Fails(Proc):
        def __init__(self, cmd, **kw):
            self.stdout = iter(["Error: Download failure, code=1\n"])

        def wait(self):
            return 1

    monkeypatch.setattr(fetch.subprocess, "Popen", Fails)
    with pytest.raises(RuntimeError, match="Download failure"):
        fetch._chromium(Job(_item(size=1)))


# ---- D-FINE: the user's yes, kept ----------------------------------------------------------------------


def test_boxes_on_the_camera_are_d_fine_through_its_own_loader_and_the_yes_is_kept(tmp_path, monkeypatch):
    from neo2.core.vision import detect

    env = tmp_path / "home.env"
    env.write_text("GEMINI_API_KEY=keep-me\nNEO_VISION_DOWNLOAD_OK=false\n")
    loads: list = []
    monkeypatch.setattr(detect, "download", lambda: loads.append("dfine") or detect.REPO)
    monkeypatch.setattr(fetch, "_hf", lambda repo, load, **kw: (lambda job: load()))
    monkeypatch.setattr(config, "USER_ENV", env)
    monkeypatch.setattr(config, "_env_lines", lambda env_path, what: (env, env.read_text().splitlines()))
    fetch._dfine(Job(_item()), tmp_path / "models")
    assert loads == ["dfine"] and config.settings().vision_download_ok is True
    assert env.read_text() == "GEMINI_API_KEY=keep-me\nNEO_VISION_DOWNLOAD_OK=true\n"


def test_d_fines_weights_are_here_only_with_config_and_weights(tmp_path, monkeypatch):
    from neo2.core.vision import detect

    monkeypatch.setenv("NEO_MODELS_PATH", str(tmp_path))
    monkeypatch.setattr(config, "_settings", None)
    assert not detect.weights_here()
    snap = tmp_path / "hf" / "models--ustc-community--dfine-small-coco" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert not detect.weights_here()
    (snap / "model.safetensors").write_text("w")
    assert detect.weights_here()


def test_the_yes_to_d_fine_keeps_every_other_line(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\nNEO_VISION_DOWNLOAD_OK=false\nB=2\nNEO_VISION_DOWNLOAD_OK=false\n")
    config.set_vision_download_ok(True, env)
    assert env.read_text() == "A=1\nNEO_VISION_DOWNLOAD_OK=true\nB=2\n"
    with pytest.raises(ValueError):
        config.set_vision_download_ok("yes", env)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="effects are off"):
        config.set_vision_download_ok(True)  # (the user's own ~/.neo/.env: never from a test)


# ---- the core's own downloads ----------------------------------------------------------------------------


def _scripted(*events: dict, hold: asyncio.Event | None = None):
    async def spawn(group):
        for ev in events:
            if hold is not None and ev.get("event") == "done":
                await hold.wait()
            yield ev

    return spawn


def test_the_cores_downloads_run_one_of_each_at_a_time_and_say_how_far(monkeypatch):
    async def go():
        hold = asyncio.Event()
        d = fetch.Downloads(_scripted(
            {"event": "plan", "size": 300}, {"event": "start", "id": "stt"},
            {"event": "progress", "id": "stt", "all_done": 100, "all_total": 300, "eta_s": 20},
            {"event": "done", "ok": True, "failed": []}, hold=hold,
        ))  # fmt: skip
        heard: list = []
        d.listen(lambda g, st: heard.append((g, st.state, st.done, st.total, st.item)))
        assert d.start("local_voice") and not d.start("local_voice")  # one at a time
        await asyncio.sleep(0.05)
        assert d.running("local_voice") and d.state("local_voice").done == 100
        hold.set()
        st = await d.wait("local_voice")
        return st, heard

    st, heard = asyncio.run(go())
    assert st.state == "done" and not st.item
    assert heard[0] == ("local_voice", "running", 0, 0, "")
    assert ("local_voice", "running", 100, 300, "stt") in heard and heard[-1][1] == "done"


def test_a_failed_or_stopped_download_says_so():
    async def go(*events, cancel=False):
        hold = asyncio.Event()
        d = fetch.Downloads(_scripted(*events, hold=hold if cancel else None))
        d.start("browser")
        await asyncio.sleep(0.02)
        if cancel:
            assert d.cancel("browser") and not d.cancel("local_voice")
        return await d.wait("browser")

    st = asyncio.run(go({"event": "item", "id": "chromium", "state": "failed", "code": "offline", "message": "x"},
                        {"event": "done", "ok": False, "failed": ["chromium"]}))  # fmt: skip
    assert (st.state, st.code) == ("failed", "offline")
    st = asyncio.run(go({"event": "start", "id": "chromium"}))  # (it ended without its "done")
    assert (st.state, st.code) == ("failed", "failed")
    st = asyncio.run(go({"event": "start", "id": "chromium"}, {"event": "done", "ok": True}, cancel=True))
    assert st.state == "cancelled"
    with pytest.raises(ValueError):
        fetch.Downloads().start("everything")


def test_the_child_is_this_interpreter_running_fetch_with_json(monkeypatch):
    """The core's download is `python -m neo2 fetch --<group> --json` as this interpreter runs (-I in
    a release engine), each JSON line an event; anything else it prints is skipped."""
    seen: dict = {}

    class Out:
        def __init__(self, lines):
            self.lines = lines

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for line in self.lines:
                yield line

    class Proc:
        returncode = None

        def __init__(self):
            self.stdout = Out([b'{"event": "plan", "size": 5}\n', b"not json\n", b'{"event": "done", "ok": true}\n'])

        async def wait(self):
            self.returncode = 0
            return 0

    async def spawn(*cmd, **kw):
        seen["cmd"] = list(cmd)
        return Proc()

    monkeypatch.setattr(fetch.asyncio, "create_subprocess_exec", spawn)

    async def go():
        d = fetch.Downloads()
        d.start("local_voice")
        return await d.wait("local_voice")

    st = asyncio.run(go())
    assert st.state == "done" and st.total == 5
    assert seen["cmd"][0] == fetch.sys.executable and seen["cmd"][-4:] == ["neo2", "fetch", "--local-voice", "--json"]


def test_the_on_device_voice_as_every_window_shows_it(monkeypatch):
    from neo2.core.voice import pipeline

    assert fetch.local_voice() == {"size": fetch.LOCAL_VOICE_BYTES, "state": "missing"}  # (conftest: not here)

    async def go():
        hold = asyncio.Event()
        fetch.downloads._spawn = _scripted(
            {"event": "progress", "all_done": 7, "all_total": 70, "eta_s": 3.5}, {"event": "done", "ok": True}, hold=hold
        )
        assert fetch.start_local_voice() == ""
        await asyncio.sleep(0.02)
        during = fetch.local_voice()
        assert fetch.start_local_voice() == "busy"
        monkeypatch.setattr(pipeline, "local_ready", lambda: True)  # (the download put it there)
        hold.set()
        await fetch.downloads.wait("local_voice")
        return during, fetch.local_voice()

    during, after = asyncio.run(go())
    assert during == {"size": fetch.LOCAL_VOICE_BYTES, "state": "downloading", "done": 7, "total": 70, "eta_s": 3.5}
    assert after == {"size": fetch.LOCAL_VOICE_BYTES, "state": "ready"}
    assert fetch.start_local_voice() == "ready"  # here already: nothing to download
