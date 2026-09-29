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
  - `scripts/get_yoloe.py` downloads the weights. It downloads only when you pass `--yes`, and it
    checks the size and hash.
  - It then converts them with Ultralytics 8.4.165 and coremltools 9.0 into a Core ML model,
    `yoloe-26s-pf.mlpackage`, in NEO's models folder (`~/.neo/models`). The converted model is a
    derivative of the weights and is AGPL-3.0 too. Its `license` field says so.
  - The conversion changes three things. The weights are stored at half precision (fp16). A
    boxes-only output head, written for NEO, replaces the original one, so the model gives one word
    per box and no masks. Tags for each word are added to the model's metadata: its part of speech
    and whether empty frames get it.
  - `neo2/core/vision/yolo.py` runs the converted model on camera frames, on the Mac, through
    Core ML. The results are the boxes on the chat's camera card and the object hints NEO gives the
    brain that answers a camera question.
  - Frames never leave the Mac for this, and nothing is written to disk.
- **Source for these parts:**
  - Ultralytics' code, at the link above.
  - The conversion, in `scripts/get_yoloe.py`.
  - The code that runs the model, in `neo2/core/vision/yolo.py`.
  - Neither the weights nor the converted model are stored in this repository.

Other models NEO downloads (Laya, the on-device voice, D-FINE small for the camera if you allow that
download) keep their own licences, as stated where each one is fetched.
