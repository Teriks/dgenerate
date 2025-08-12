## Usage

dgenerate must be installed in the environment for static asset generation tasks.

### Build everything:
```bash
python -m assetgen.build
```

### Build only README.rst template:
```bash
python -m assetgen.build --target readme
```

### Build only ReadTheDocs templates:
```bash
python -m assetgen.build --target docs
```

### Build only Console UI schemas:
```bash
python -m assetgen.build --target console-schemas
```

### Build only Helsinki NLP model map
```bash
python -m assetgen.build --target helsinki-nlp-translation-map
```

### Build only vendored HF Hub configs
```bash
python -m assetgen.build --target hf-configs
```

### Cache management:
```bash
# Disable command cache completely (RST templating)
python -m assetgen.build --no-command-cache

# Clear specific command cache patterns (RST templating)
python -m assetgen.build --no-command-cache "dgenerate --help"

# Clear specific command cache patterns (RST templating)
python -m assetgen.build --no-command-cache-regex "..."
```

## Output Locations

- **README.rst**: Project root (for GitHub /  PyPI)
- **docs/intro.rst**: docs/ directory (for ReadTheDocs intro)
- **docs/manual.rst**: docs/ directory (for ReadTheDocs usage manual)
- **dgenerate/console/schemas**: static Console UI schemas
- **dgenerate/translators/data/helsinki-nlp-translation-map.json**: Helsinki NLP model map
- **dgenerate/pipelinewrapper/hub_configs**: Vendored HF Hub model configs