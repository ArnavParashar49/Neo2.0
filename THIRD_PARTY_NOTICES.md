# Third-party notices

NEO's own code is under the MIT License ([LICENSE](LICENSE)). The parts below are other people's work
and keep their own licences.

## YOLOE-26s (Ultralytics): AGPL-3.0

- **What it is:** YOLOE-26s prompt-free (`yoloe-26s-seg-pf.pt`), an open-vocabulary object detector
  by [Ultralytics](https://github.com/ultralytics/ultralytics). Its vocabulary has 4,585 words and is
  stored in the model.
- **Licence:** [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html)
  (Ultralytics also sells an [Enterprise licence](https://ultralytics.com/license)). NEO's author
  chose this model for NEO's camera under these terms.
- **Where it comes from:**
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg-pf.pt`, 40,003,064
  bytes, SHA-256 `01605da3f2e0e5325df45d5f5eef8e40b5b5c7d085399be3c33ac638258b150a`.
- **How NEO uses it:**
  - `scripts/get_yoloe.py` downloads the weights. It downloads only when it is passed `--yes`, and it
    checks the size and hash. `scripts/install-app.sh` runs it once, with `--yes`, when the converted
    model isn't in `~/.neo/models` yet.
  - It then converts them with Ultralytics 8.4.165 and coremltools 9.0 into a Core ML model,
    `yoloe-26s-pf.mlpackage`, in NEO's models folder (`~/.neo/models`). The converted model is a
    derivative of the weights and is AGPL-3.0 too. Its `license` field says so.
  - The conversion changes three things. The weights are stored at half precision (fp16). A
    boxes-only output head, written for NEO, replaces the original one, so the model gives one word
    per box and no masks. Tags for each word are added to the model's metadata: its part of speech
    and whether empty frames get it.
  - `neo2/core/vision/yolo.py` runs the converted model on camera frames, on the Mac, through
    Core ML. By default it lets the model name only about 230 everyday things of its vocabulary
    (through the model's own per-word input); the setting `vision_words=all` allows all of it. The
    results are the boxes on the chat's camera card and the object hints NEO gives the brain that
    answers a camera question.
  - Frames never leave the Mac for this, and nothing is written to disk.
- **Source for these parts:**
  - Ultralytics' code, at the link above.
  - The conversion, in `scripts/get_yoloe.py`.
  - The code that runs the model, in `neo2/core/vision/yolo.py`.
  - Neither the weights nor the converted model are stored in this repository.

## shazamio and shazamio-core (dotX12): MIT

- **What they are:** [shazamio](https://github.com/dotX12/ShazamIO) 0.8.1, an unofficial Python client
  for Shazam's song recognition, and shazamio-core 1.1.2, its Rust part, which computes Shazam's
  fingerprint of a piece of audio. Neither is made by, or affiliated with, Shazam or Apple.
- **Licence:** MIT, both. shazamio: Copyright (c) 2021 dotX12. shazamio-core: Copyright (c) 2024
  dotX12. Each package ships its licence text (`LICENSE.txt`, `LICENSE`).
- **How NEO uses them:** the `music` extra (in NEO.app's `app` extra) installs them, pinned in
  `requirements.lock`. When the user asks what song is playing, `neo2/core/tools/music/recognize.py`
  builds a WAV of the last seconds NEO heard, in memory, and shazamio-core turns it into Shazam's
  fingerprint. shazamio then sends that fingerprint, its length and a timestamp to Shazam, in one
  request through NEO's own HTTP client. The audio itself never leaves the Mac and is never written
  to disk. Shazam's answer (the song's title, artist, album, year, genre, links and cover art) is
  shown in the chat and said aloud.
- **Source:** the packages above; NEO's code that uses them is in `neo2/core/tools/music/`.

Other models NEO downloads (Laya, the on-device voice, D-FINE small for the camera if you allow that
download) keep their own licences, as stated where each one is fetched.
