# Contributing to NEURON

Thank you for your interest in contributing to NEURON!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ertugrulakben/neuron.git
cd neuron

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"
```

## Code Style

We use:
- **Black** for formatting
- **Ruff** for linting
- **MyPy** for type checking

Run before committing:
```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Testing

```bash
pytest tests/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with clear messages
6. Push and open a PR

## Areas for Contribution

- **Core algorithms**: Improvements to SMTR, HCMC, or decay mechanisms
- **Benchmarks**: New evaluation datasets or metrics
- **Documentation**: Tutorials, examples, translations
- **Integrations**: New LLM backends or storage systems

## Questions?

Open an issue or reach out to [@ertugrulakben](https://twitter.com/ertugrulakben)
