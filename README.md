VulEx
=======

A reproduction package for the paper **"VulEx: Extracting Vulnerability Knowledge from Software Engineering Agent Memory."**

This package provides the **VulEx** implementation, the inputs required to reproduce it, model download instructions, CAP/SSS bank builders, and the experiment runner.

Contents
--------

- Overview
- Directory structure
- Key files
- Quick start
- Statement
- Citation

Project overview
----------------

VulEx is a black-box method for extracting vulnerability knowledge from the memory of software engineering agents. Each query is framed as a CWE-focused security review, and its public review seeds are built with two complementary components:

- **CWE-Anchored Probing (CAP)** selects repository anchors using public CWE priors and repository coverage.
- **Suspect-Style Synthesis (SSS)** selects public vulnerability functions and converts their evidence into one- or two-line project-styled suspect snippets.

The default configuration runs 30 queries per repetition: 18 CAP queries and 12 SSS queries, repeated three times.

Directory structure
-------------------

```text
.
|-- .gitignore                        # Local artifact rules
|-- README.md                         # Package documentation
|-- build.sh                          # Build CAP/SSS banks and validated snippets
|-- config.json                       # Fixed method, input, and runtime configuration
|-- experiments/
|   |-- __init__.py
|   `-- vulex/
|       |-- build_cap_anchor_bank.py  # CAP bank construction
|       |-- build_sss_bank.py         # SSS selection and embedding construction
|       |-- build_snippets.py         # Exact source-line snippet generation
|       |-- stagedvulbert_encoder.py  # MSP/pretrain selector inference
|       |-- agent_harness.py          # Memory-only OpenClaw harness
|       |-- run_experiment.py         # Per-repeat experiment runner
|       |-- score_run.py              # Output parsing and scoring
|       `-- ...                       # Other VulEx support modules
|-- generated/                        # Populated by build.sh
|-- inputs/                           # Experiment inputs and download guides
|-- outputs/                          # Populated by run.sh
|-- requirements.txt                  # Python dependencies
|-- run.sh                            # Run VulEx
|-- scripts/
|   `-- run.py                        # Three-repeat orchestration and aggregation
|-- tests/
`-- third_party/
    `-- StagedVulBERT/                # Required upstream model definition
```

Important file descriptions
---------------------------

- **`config.json` - Reproduction configuration.**  
  Defines the CAP and SSS parameters, the fixed query schedule, package-relative input paths, the CodeBERT and MSP/pretrain model paths, the target-agent model, and the OpenClaw runtime settings.

- **`build.sh` - Review-seed build script.**  
  Builds the CAP bank, encodes public functions with the MSP/pretrain selector, builds the SSS bank, and generates source-exact snippets.

- **`experiments/vulex/build_cap_anchor_bank.py` - CAP builder.**  
  Builds 12 public anchors per CWE with prior weight `0.30`, repository demand `512`, and candidate pool size `1024`.

- **`experiments/vulex/build_sss_bank.py` - SSS builder.**  
  Filters single-body-line functions and selects up to five public payload sources per CWE with prior weight `0.50` and repository demand `512`.

- **`experiments/vulex/build_snippets.py` - SSS snippet builder.**  
  Asks for one or two exact source lines for each selected public function and rejects any response that cannot be matched to the supplied source.

- **`run.sh` - Experiment entry point.**  
  Runs three independent 30-query repetitions, scores each repetition, and writes aggregate metrics to `outputs/`.

Quick start
-----------

### 1. Set up the environment

Use Python 3.11 on Linux or macOS. The OpenClaw target-agent runtime requires Docker.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Make sure the PyTorch wheel matches your host platform. If dependency resolution does not select a suitable wheel, install the appropriate build from [PyTorch](https://pytorch.org/) first.

Install [Docker Engine or Docker Desktop](https://docs.docker.com/get-docker/) for the OpenClaw target-agent runtime. Before starting the experiment, pull the image used by this package:

```bash
docker pull ghcr.io/openclaw/openclaw@sha256:e2482a66682de6f540dcfd9921e410c23fd060dcd441382ff952247ee911a672
```

`run.sh` passes this image to Docker for every repetition. Docker can also pull it automatically when it is missing, but pre-pulling exposes image and registry errors before a long run.

### 2. Download the Linux source archive

Download the Linux source archive and save it as `inputs/linux.zip` using the link below:

[Google Drive download link](https://drive.google.com/file/d/197FS98SjLgk63ry_IbO_VQPXEQRhYPPQ/view?usp=sharing)

Alternatively, download the archive directly with `gdown`:

```bash
gdown --id 197FS98SjLgk63ry_IbO_VQPXEQRhYPPQ -O inputs/linux.zip
```

Extract the archive from the package root:

```bash
unzip inputs/linux.zip -d inputs
```

The extracted source tree must be available at `inputs/linux/`.

### 3. Download the CodeBERT model

Official model page: [Hugging Face `microsoft/codebert-base`](https://huggingface.co/microsoft/codebert-base)

Download the complete official `microsoft/codebert-base` repository into `inputs/models/codebert-base` by running:

```bash
hf download microsoft/codebert-base \
  config.json merges.txt pytorch_model.bin \
  special_tokens_map.json tokenizer_config.json vocab.json \
  --local-dir inputs/models/codebert-base
```

### 4. Download the StagedVulBERT MSP model

Download the MSP pre-trained model from the [official StagedVulBERT link](https://drive.google.com/file/d/1frZLAmB2F0z1LLEwjVmoAtqKlPMg13uR/view?usp=sharing) and save it as:

```text
inputs/models/stagedvulbert-msp.bin
```

Alternatively, download it directly with `gdown`:

```bash
gdown --id 1frZLAmB2F0z1LLEwjVmoAtqKlPMg13uR \
  -O inputs/models/stagedvulbert-msp.bin
```

The model must be present at this path before you run `build.sh`.

### 5. Configure the model endpoint

Both review-seed generation and target-agent execution require an OpenAI-compatible API endpoint. Set the API key and endpoint before running the build or experiment:

```bash
export OPENAI_API_KEY="<your-api-key>"
export VULEX_OPENAI_API_BASE="<your-openai-compatible-v1-endpoint>"
```

The target-agent model defaults to the value in `config.json`. Set `VULEX_AGENT_MODEL` only when intentionally reproducing a different model configuration:

```bash
export VULEX_AGENT_MODEL="gpt-5.5"
```

### 6. Build the CAP and SSS artifacts

```bash
./build.sh
```

This command creates the following artifacts under `generated/`:

```text
generated/cap_bank.json
generated/sss_embeddings.json
generated/sss_bank.json
generated/snippet_cache/
generated/sss_snippets.json
```

The files are generated locally from the supplied experiment inputs and the downloaded CodeBERT and MSP models.

### 7. Run VulEx

```bash
./run.sh
```

Each repetition uses an empty agent workspace, OpenClaw native BM25 memory search, and only the `memory_search` and `memory_get` tools. Every `memory_search` call is limited to `maxResults <= 4`.

Results are written to:

```text
outputs/repeat_1/
outputs/repeat_2/
outputs/repeat_3/
outputs/aggregate.json
```

To resume a stopped reproduction from its completed checkpoints, run:

```bash
./run.sh --resume
```

Statement
---------

This standalone package contains the VulEx implementation, reproduction inputs, Linux and model download instructions, CAP/SSS builders, the OpenClaw execution harness, the scorer, and the three-repetition runner.

Citation
--------

Citation metadata will be added when the paper is published. Until then, please refer to the work by its title:

> VulEx: Extracting Vulnerability Knowledge from Software Engineering Agent Memory.
