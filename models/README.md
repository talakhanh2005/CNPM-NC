# Model checkpoint

The repository tracks the runtime checkpoint used by the OpenEmotion API:

- File: `openemotion_efficientnet_b0.pth`
- Architecture: EfficientNet-B0 with a 7-class classifier head
- Input size: `224 × 224`
- SHA-256: `95076144e742f8369cd16cd58fbc8e60ac90e9d81801cd025a4f346bf067a836`
- Class order: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

The checkpoint is the default configured by `openemotion.config.DEFAULT_MODEL_PATH`
and `AI_MODEL_PATH` in `.env.example`. CI/local tests verify that it can be loaded
by `AIService`; this smoke test is not an accuracy benchmark.

`emotion_model_quick_3ep.pth` is an experimental three-epoch checkpoint with low
evaluation accuracy and is intentionally not tracked.
