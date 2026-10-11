"""An offline-first start: the pinned models are read from the Hugging Face cache before the Hub is
asked, and a brain that isn't set up says so once, not on every step.

Asking the Hub first on every start printed "Fetching 5 files" and the Hub's unauthenticated-request
warning each time (dozens of times in one log) and stalled an offline start on its timeouts; a brain
with no key logged "unavailable" on every chain build (270 lines). A cache that misses, or lacks a
file, still falls back once to the call exactly as it was (under HF_HUB_OFFLINE the hub's own, which
never goes online). No model, no network: snapshot_download is a fake that records how it was called."""

from __future__ import annotations

import logging

import pytest

from neo2.core import models

_LAYA = ("rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json")


def _laya_files(root, sub=""):
    for f in _LAYA:
        p = root / sub / f if sub else root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")


class _Hub:
    """snapshot_download as a cache that hands back `root` when `cached` (else misses), and a call
    without local_files_only that brings down `download` first (a callable) and hands back the same
    folder — except under HF_HUB_OFFLINE (`offline`), where the hub's own call is the cache alone and
    never goes online. `downloads`: the times it went online."""

    def __init__(self, monkeypatch, root, *, cached: bool, download=None, offline: bool = False) -> None:
        import huggingface_hub

        self.root, self.cached, self.download, self.offline = root, cached, download, offline
        self.calls: list[tuple[str, dict]] = []
        self.downloads = 0
        monkeypatch.setattr(huggingface_hub, "snapshot_download", self)

    def __call__(self, repo, **kw):
        self.calls.append((repo, kw))
        if kw.get("local_files_only") or self.offline:
            if not self.cached:
                from huggingface_hub.errors import LocalEntryNotFoundError

                raise LocalEntryNotFoundError("not in the cache")
            return str(self.root)
        self.downloads += 1
        if self.download:
            self.download()
        return str(self.root)


# ---- laya_dir ---------------------------------------------------------------------------------
def test_a_cached_checkpoint_is_one_call_to_the_cache_and_none_to_the_hub(monkeypatch, tmp_path):
    _laya_files(tmp_path)
    hub = _Hub(monkeypatch, tmp_path, cached=True)
    assert models.laya_dir() == str(tmp_path)
    assert len(hub.calls) == 1
    repo, kw = hub.calls[0]
    assert repo == models.LAYA and kw["revision"] == models.LAYA_REVISION and kw["local_files_only"] is True
    assert "model.safetensors" in kw["allow_patterns"]  # still only this checkpoint's files


def test_a_cache_miss_downloads_once_exactly_as_before(monkeypatch, tmp_path):
    hub = _Hub(monkeypatch, tmp_path, cached=False, download=lambda: _laya_files(tmp_path, "multilingual"))
    assert models.laya_dir("multilingual") == str(tmp_path / "multilingual")
    assert [kw.get("local_files_only") for _, kw in hub.calls] == [True, None]
    repo, kw = hub.calls[1]  # today's call, word for word: no local_files_only, the same patterns
    assert repo == models.LAYA and kw == {
        "revision": models.LAYA_REVISION,
        "allow_patterns": [f"multilingual/{f}" for f in models.LAYA_FILES],
    }


def test_a_cached_revision_without_this_checkpoints_files_downloads_them(monkeypatch, tmp_path):
    _laya_files(tmp_path)  # only the English checkpoint ever came down
    hub = _Hub(monkeypatch, tmp_path, cached=True, download=lambda: _laya_files(tmp_path, "multilingual"))
    assert models.laya_dir("multilingual") == str(tmp_path / "multilingual")
    assert [kw.get("local_files_only") for _, kw in hub.calls] == [True, None]


def test_files_still_missing_after_the_download_are_the_same_clear_error(monkeypatch, tmp_path):
    hub = _Hub(monkeypatch, tmp_path, cached=True)  # an empty folder, and the Hub brings nothing
    with pytest.raises(FileNotFoundError, match="english checkpoint isn't on this Mac yet"):
        models.laya_dir()
    assert len(hub.calls) == 2


def test_offline_never_goes_online_and_errors_as_before(monkeypatch, tmp_path):
    hub = _Hub(monkeypatch, tmp_path, cached=True, offline=True)  # the folder's there, the files aren't
    with pytest.raises(FileNotFoundError, match="english checkpoint isn't on this Mac yet"):
        models.laya_dir()  # (the second call is the hub's own offline one, as it always was)
    hub.cached = False  # not even the folder: the hub's own error, a FileNotFoundError as before
    with pytest.raises(FileNotFoundError):
        models.laya_dir("multilingual")
    _laya_files(tmp_path)
    hub.cached, hub.calls = True, []
    assert models.laya_dir() == str(tmp_path) and len(hub.calls) == 1  # here: the cache, and nothing else
    assert hub.downloads == 0


def test_a_cache_that_errors_any_other_way_still_falls_back_to_the_download(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []

    def snapshot_download(repo, **kw):
        calls.append(kw)
        if kw.get("local_files_only"):
            raise OSError("the cache folder can't be read")
        _laya_files(tmp_path)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert models.laya_dir() == str(tmp_path) and len(calls) == 2


# ---- local_path -------------------------------------------------------------------------------
_STT = "mlx-community/parakeet-tdt-0.6b-v3"


def _voice_files(root):
    (root / "config.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_text("weights")


def test_an_unpinned_model_is_handed_on_as_named_without_asking_anyone(monkeypatch, tmp_path):
    hub = _Hub(monkeypatch, tmp_path, cached=True)
    assert models.local_path("someone/their-own-model") == "someone/their-own-model" and hub.calls == []


def test_a_cached_voice_model_is_one_call_to_the_cache(monkeypatch, tmp_path):
    _voice_files(tmp_path)
    hub = _Hub(monkeypatch, tmp_path, cached=True)
    assert models.local_path(_STT) == str(tmp_path)
    assert hub.calls == [(_STT, {"revision": models.REVISIONS[_STT], "local_files_only": True})]


def test_a_voice_model_not_in_the_cache_downloads_once_exactly_as_before(monkeypatch, tmp_path):
    hub = _Hub(monkeypatch, tmp_path, cached=False, download=lambda: _voice_files(tmp_path))
    assert models.local_path(_STT) == str(tmp_path)
    assert hub.calls[1] == (_STT, {"revision": models.REVISIONS[_STT]})  # today's call, word for word
    assert len(hub.calls) == 2


def test_an_empty_or_broken_cached_folder_is_downloaded_again(monkeypatch, tmp_path):
    hub = _Hub(monkeypatch, tmp_path, cached=True, download=lambda: _voice_files(tmp_path))
    assert models.local_path(_STT) == str(tmp_path)  # an empty folder: nothing was ever fetched
    assert [kw.get("local_files_only") for _, kw in hub.calls] == [True, None]
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "af_heart.safetensors").symlink_to(tmp_path / "blobs" / "gone")
    hub.calls.clear()
    assert models.local_path(_STT) == str(tmp_path)  # a link to a blob that was cleared away
    assert [kw.get("local_files_only") for _, kw in hub.calls] == [True, None]


def test_a_voice_model_offline_never_goes_online(monkeypatch, tmp_path):
    hub = _Hub(monkeypatch, tmp_path, cached=False, offline=True)
    with pytest.raises(FileNotFoundError):
        models.local_path(_STT)
    _voice_files(tmp_path)
    hub.cached, hub.calls = True, []
    assert models.local_path(_STT) == str(tmp_path) and len(hub.calls) == 1
    assert hub.downloads == 0


# ---- no Hub to be had: a cached folder the hub calls incomplete ------------------------------------
class _Partial:
    """snapshot_download as the hub behaves with a folder that lacks some of the repo's files (a
    loader fetched only what it reads, never the README): the cache raises IncompleteSnapshotError
    naming the folder; the network is down (`online=False`) or brings the rest down."""

    def __init__(self, monkeypatch, root, *, online: bool) -> None:
        import huggingface_hub

        self.root, self.online = root, online
        self.calls: list[dict] = []
        monkeypatch.setattr(huggingface_hub, "snapshot_download", self)

    def __call__(self, repo, **kw):
        from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError

        self.calls.append(kw)
        if kw.get("local_files_only"):
            raise IncompleteSnapshotError(
                "3 file(s) are missing (README.md, ...)", snapshot_path=str(self.root)
            )
        if not self.online:
            raise LocalEntryNotFoundError("An error happened while trying to locate the files on the Hub")
        return str(self.root)


def test_with_the_hub_out_of_reach_a_voice_model_missing_only_its_readme_still_loads(monkeypatch, tmp_path):
    """No network, or HF_HUB_OFFLINE: either way the call as it was fails on the missing README."""
    _voice_files(tmp_path)
    hub = _Partial(monkeypatch, tmp_path, online=False)
    assert models.local_path(_STT) == str(tmp_path)  # today this start failed: no README, no voice
    assert [kw.get("local_files_only") for kw in hub.calls] == [True, None]  # the Hub was still asked first


def test_online_the_hub_still_completes_the_folder_as_before(monkeypatch, tmp_path):
    _voice_files(tmp_path)
    hub = _Partial(monkeypatch, tmp_path, online=True)
    assert models.local_path(_STT) == str(tmp_path)
    assert hub.calls[1] == {"revision": models.REVISIONS[_STT]}


def test_an_incomplete_folder_with_nothing_usable_in_it_is_the_hubs_error(monkeypatch, tmp_path):
    from huggingface_hub.errors import LocalEntryNotFoundError

    _Partial(monkeypatch, tmp_path, online=False)  # an empty folder
    with pytest.raises(LocalEntryNotFoundError, match="locate the files on the Hub"):
        models.local_path(_STT)


def test_laya_out_of_reach_of_the_hub_loads_from_the_cache_when_its_files_are_there(monkeypatch, tmp_path):
    from huggingface_hub.errors import LocalEntryNotFoundError

    hub = _Partial(monkeypatch, tmp_path, online=False)
    with pytest.raises(LocalEntryNotFoundError):  # not there: the download's own error, as before
        models.laya_dir()
    _laya_files(tmp_path)
    assert models.laya_dir() == str(tmp_path)
    assert [kw.get("local_files_only") for kw in hub.calls] == [True, None, True, None]


# ---- "[brain] <name> unavailable": once per process, per brain and kind of error ----------------
def test_a_brain_that_isnt_set_up_says_so_once(monkeypatch, caplog):
    from neo2.core import providers as prov

    monkeypatch.setattr(prov, "_said_unavailable", set())
    caplog.set_level(logging.INFO, logger="neo")
    for _ in range(3):  # a chain is built for every step
        prov.Chain(["claude"])._providers()
    prov.Chain(["gemini_lite"], eyes=["claude"])._eye_brains()  # the seeing fallbacks: the same rule
    lines = [r.getMessage() for r in caplog.records if "unavailable" in r.getMessage()]
    assert lines == ["[brain] claude unavailable: ANTHROPIC_API_KEY is not set"]


def test_a_different_kind_of_error_from_the_same_brain_is_said_too(monkeypatch, caplog):
    from neo2.core import providers as prov
    from neo2.core.providers.base import ProviderError

    monkeypatch.setattr(prov, "_said_unavailable", set())
    caplog.set_level(logging.INFO, logger="neo")
    errors = iter([ProviderError("no key"), ProviderError("no key, again"), ImportError("no package")])

    def make(name):
        raise next(errors)

    monkeypatch.setattr(prov, "make", make)
    prov.Chain(["claude"])._providers()
    prov.Chain(["claude"])._providers()
    prov.Chain(["gemini"], eyes=["claude"])._eye_brains()
    lines = [r.getMessage() for r in caplog.records if "unavailable" in r.getMessage()]
    assert lines == ["[brain] claude unavailable: no key", "[brain] claude unavailable: no package"]


def test_the_chain_still_skips_a_brain_that_isnt_set_up_every_time(monkeypatch):
    """Only the log line is once: the brain is left out of every chain, and its reason kept."""
    from neo2.core import providers as prov

    monkeypatch.setattr(prov, "_said_unavailable", set())
    for _ in range(2):
        c = prov.Chain(["claude"])
        assert c._providers() == [] and [n for n, _ in c._missing] == ["claude"]
