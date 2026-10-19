# The registry

Every extensible component in glide is looked up by **name** at build time. The
names live in a handful of small `Registry` objects in `glide.registry`, and
`config.plugins` is the hook that gets your own code imported early enough to
add to them.

This page covers what the registries are, the exact call signature each one
expects, and how to add your own entry.

## 1. Why it exists

A config refers to components by string:

```yaml
speech:
  encoder:   { name: whisper }
  projector: { name: mlp_gelu }
rl:
  rewards:
    - { name: exact_match, weight: 1.0 }
```

Nothing in the YAML names a Python object. At build time glide resolves each
string through a registry, so a config stays portable and your own components
plug in exactly like the built-ins.

## 2. The registries

`glide.registry` exposes one `Registry` per component kind:

| Registry | Resolved from | Registered value is a… |
|---|---|---|
| `audio_encoders` | `speech.encoder.name` | builder returning an `AudioEncoder` |
| `projectors` | `speech.projector.name` | builder returning a `Projector` |
| `rewards` | `rl.rewards[].name` | builder returning a TRL reward function |
| `metrics` | `eval.metrics[]` | the metric function itself |
| `collators` | `template.collator` | builder returning a collator |
| `models` | — | *declared, currently unused* |
| `templates` | — | *declared, currently unused* |

> `models` and `templates` exist for symmetry but nothing resolves them yet, so
> registering into them has no effect today.

Each is a `Registry[Callable]` with a tiny API:

```python
reg.register(name, obj=None, *, exist_ok=False)  # decorator or direct call
reg.get(name)                                    # -> value, KeyError if absent
name in reg                                      # membership
reg.names()                                      # -> sorted list[str]
```

`register` raises `KeyError` on a duplicate name unless you pass
`exist_ok=True`. Built-ins register with `exist_ok=True` so re-importing a
module is harmless; use it in your own plugin when you deliberately **override**
a built-in name.

## 3. Signatures per registry

The registry stores whatever you give it, so the signature is a contract you
have to match. These are the calls glide actually makes.

### `audio_encoders`

Called as `builder(cfg, sample_rate)` with the **`AudioEncoderConfig`
sub-config** (`speech.encoder`) and `speech.sample_rate` — not the whole
`GlideConfig`.

The returned module must set `output_dim` (the projector reads it for its
`in_dim`) and its `forward` must return `(hidden_states, attention_mask)` with
`hidden_states` shaped `(batch, frames, hidden)`.

```python
from glide.registry import audio_encoders
from glide.models import AudioEncoder

class MyEncoder(AudioEncoder):
    def __init__(self, pretrained, freeze=False):
        super().__init__()
        ...
        self.output_dim = self.model.config.hidden_size
    def forward(self, input_features, attention_mask=None):
        return hidden_states, out_mask

@audio_encoders.register("my_encoder")
def build_my_encoder(cfg, sample_rate: int = 16000):
    return MyEncoder(cfg.pretrained or "my/default", freeze=cfg.freeze,
                     **cfg.extra_kwargs)
```

Give `sample_rate` a default: it is passed positionally today, but every
built-in declares it as `sample_rate: int = 16000`.

### `projectors`

Called as `builder(cfg, in_dim, out_dim)` — note this is the **`ProjectorConfig`
sub-config**, not the whole `GlideConfig`, plus the encoder and LLM widths.

```python
from glide.registry import projectors

@projectors.register("my_proj")
def build_my_proj(cfg, in_dim, out_dim):
    return MyProjector(in_dim, out_dim, downsample=cfg.downsample)
```

If your projector cannot downsample, raise on `cfg.downsample > 1` rather than
ignoring it — `SpeechLLM` divides `audio_lengths` by the configured factor
regardless, and a mismatch silently truncates audio.

### `rewards`

The registered value is a **builder**, called once as `builder(**spec.kwargs)`;
the function it returns is what TRL calls per batch.

```python
from glide.registry import rewards

@rewards.register("my_reward")
def build_my_reward(scale: float = 1.0):
    def _reward(prompts, completions, completion_ids=None, **columns):
        return [scale * len(c) for c in completions]
    return _reward
```

```yaml
rl:
  rewards:
    - name: my_reward
      weight: 1.0
      kwargs: { scale: 0.5 }     # -> build_my_reward(scale=0.5)
```

`**columns` receives the extra JSONL columns (one value per sample), so a
record with a `reference` field arrives as `reference=[...]`. glide sets
`fn.__name__` to the spec name so TRL logs a distinct `rewards/<name>/mean`.

### `metrics`

Unlike rewards, the registered callable **is** the metric — there is no builder
indirection. It takes `(predictions, references)` and returns a dict of
`{metric_name: float}`.

```python
from glide.registry import metrics

@metrics.register("my_metric")
def compute_my_metric(predictions, references, *, normalize=True):
    return {"my_metric": 0.0}
```

### `collators`

Called as `builder(config, processor)` with the full `GlideConfig` and the
loaded HF processor. Only consulted when `template.collator` is set; otherwise
glide picks its own collator from the modality.

```python
from glide.registry import collators

@collators.register("my_collator")
def build_my_collator(config, processor):
    return MyCollator(processor, config.data)
```

## 4. Getting your code imported

Registration only happens when your module is imported. List your plugin under
`plugins` in the config:

```yaml
plugins:
  - src/my_plugin.py        # file path
  - src.my_plugin           # or dotted module path
```

`load_plugins` runs before anything is built. Resolution rules:

- Anything containing `/` or ending in `.py` is imported **as a file**, with a
  module name derived from a hash of the resolved path. Two plugins with the
  same basename in different directories therefore do not collide, and
  importing the same file twice is a no-op instead of a duplicate-registration
  error.
- A bare dotted path goes through `importlib`. Only if that raises
  `ModuleNotFoundError` does glide fall back to inserting the current working
  directory on `sys.path` and retrying — so run `glide` from your project root.

## 5. Debugging

```python
from glide.registry import rewards
rewards.names()          # ['cer', 'exact_match', 'format', 'length']
"my_reward" in rewards   # False until the plugin module is imported
```

A miss raises with the full list of what *is* registered:

```
KeyError: "No reward function named 'my_rewrd'. Registered: ['cer', 'exact_match', ...]"
```

Almost always this means the plugin was never imported: check that `plugins`
names the right path and that you invoked `glide` from the directory that path
is relative to.
