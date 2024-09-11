# aria-cl

Efficient solo-piano audio classification

## Install 

Requires Python > 3.11.

```
git clone https://github.com/loubbrad/aria-cl
cd aria-cl
pip install -e .
```

Download the preliminary model weights:

```
wget https://storage.googleapis.com/aria-checkpoints/cl/large-0.1.safetensors
```

## Usage

After downloading the preliminary model weights, run inference with:

```bash
python ariacl/run.py process-files \
    --load_dir <dir-containing-audio-files> \
    --save_path <path-to-save-output-json> \
    --checkpoint_path <path-to-model-weights>
```

You can customize how aggressive the classification and segmentation is in the config file located at `config/config.json`. For example, if you don't want to perform segmentation, the following configuration will remove this functionality and instead just classify files according to their average score.

```json
"inference": {
    "batch_size": 256,
    "threshold_score": 0.0,
    "min_avg_score": 0.5,
    "min_valid_window_s": 0,
    "min_invalid_window_s": 5
}
```

The output json file is of the form path: List[[piano_segment_start (in seconds), piano_segment_end (in seconds)]]. You can provide this json file to [aria-amt](https://github.com/EleutherAI/aria-amt) using the `-segments_json_path` flag, and the segments will be transcribed independently, instead of the entire file.