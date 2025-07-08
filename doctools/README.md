## Usage

dgenerate must be installed in the environment for documentation build tasks.

### Build everything:
```bash
python -m doctools.build
```

### Build only README.rst template:
```bash
python -m doctools.build --target readme
```

### Build only ReadTheDocs templates:
```bash
python -m doctools.build --target docs
```

### Build only Console UI schemas:
```bash
python -m doctools.build --target console-schemas
```

### Build only Helsinki NLP model map
```bash
python -m doctools.build --target helsinki-nlp-translation-map
```

### Cache management:
```bash
# Disable command cache completely (RST templating)
python -m doctools.build --no-command-cache

# Clear specific command cache patterns (RST templating)
python -m doctools.build --no-command-cache "dgenerate --help"

# Clear specific command cache patterns (RST templating)
python -m doctools.build --no-command-cache-regex "..."
```

## Output Locations

- **README.rst**: Project root (for GitHub /  PyPI)
- **docs/intro.rst**: docs/ directory (for ReadTheDocs intro)
- **docs/manual.rst**: docs/ directory (for ReadTheDocs usage manual)
- **dgenerate/console/schemas**: static Console UI schemas
- **dgenerate/translators/data/helsinki-nlp-translation-map.json**: Helsinki NLP model map