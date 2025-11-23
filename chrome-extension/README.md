# WebLLM Chrome Extension

![Chrome Extension](https://github.com/mlc-ai/mlc-llm/assets/11940172/0d94cc73-eff1-4128-a6e4-70dc879f04e0)

## Setup

#### 1. Install Dependencies

```bash
npm install
```

#### 2. Build Extension

```bash
npm run build
```

This will create a new directory at `chrome-extension/dist/` with the model files included.

### 3. Load Extension in Chrome

1. Go to Extensions > Manage Extensions
2. Make sure developer mode toggle is selected
3. Select Load Unpacked
4. Add the `chrome-extension/dist/` directory