# Contributing to Affective RAG

Thank you for your interest in contributing to Affective RAG! This library is designed to advance emotion-aware memory retrieval research and applications.

## Ways to Contribute

### 1. Report Issues
- **Bugs**: Describe what happened vs. what you expected
- **Feature Requests**: Explain your use case and proposed solution
- **Documentation**: Point out unclear sections or missing examples

### 2. Submit Code Contributions

#### Getting Started
```bash
# Clone the repository
git clone https://github.com/jeje1197/AffectiveRAG.git
cd AffectiveRAG/affective_rag

# Install dependencies
pip install -r requirements.txt

# Run examples to verify setup
python -m affective_rag.examples.spreading_demo
python -m affective_rag.examples.unified_sampling_demo
```

#### Code Guidelines
- **Style**: Follow PEP 8 conventions
- **Type Hints**: Add type annotations for function signatures
- **Documentation**: Include docstrings for public methods
- **Examples**: Update examples if adding new features

#### Pull Request Process
1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-new-feature`
3. **Make your changes** with clear, focused commits
4. **Test** your changes against the examples
5. **Update documentation** (README, docstrings, examples as needed)
6. **Submit a PR** with a clear description of the changes

### 3. Improve Documentation
- Add tutorials for common use cases
- Improve API documentation clarity
- Create integration examples (Pinecone, Weaviate, Qdrant, etc.)
- Fix typos or unclear explanations

### 4. Share Research
- Publish papers using Affective RAG and share citations
- Share benchmarks or evaluation datasets
- Contribute experimental configurations or scoring functions

## Development Setup

### Running Tests
```bash
# Run all examples as smoke tests
python -m affective_rag.examples.spreading_demo
python -m affective_rag.examples.no_spread_demo
python -m affective_rag.examples.topk_demo
python -m affective_rag.examples.unified_sampling_demo
python -m affective_rag.examples.logging_example
```

### Project Structure
```
affective_rag/
├── core/              # Core library code
│   ├── graph.py       # MemoryGraph implementation
│   ├── memory.py      # Event models
│   └── ...
├── examples/          # Usage examples
├── docs/              # Documentation
├── LICENSE            # Apache 2.0 License
├── PATENTS            # Patent grant information
└── README.md          # Main documentation
```

## Contribution Types Welcome

### Code Contributions
- **Bug fixes**: Always welcome
- **Performance improvements**: Optimize graph traversal or scoring
- **New features**: Discuss in an issue first for major features
- **Integration examples**: New vector DB adapters or production patterns

### Non-Code Contributions
- **Documentation**: Tutorials, guides, API improvements
- **Examples**: Real-world use cases and integration patterns
- **Testing**: Edge cases, performance benchmarks
- **Community**: Answer questions, help other users

## Commercial Use & Patents

This library is released under Apache 2.0 for maximum research and development benefit. The implemented algorithms are subject to patent applications.

- **Research/Academic Use**: Fully free and encouraged
- **Commercial Use**: See [PATENTS](PATENTS) file for details
- **Questions**: Contact focalways99@gmail.com

## Code of Conduct

A positive and respectful environment is essential for the Affective RAG community. Please read and follow our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) in all project spaces.

## Attribution

When contributing code, you agree that your contributions will be licensed under the Apache 2.0 License. For academic work using this library, please cite:

```
Evans, J. (2025). Beyond Semantics: Information Retrieval with Emotion and Time.
```

## Questions?

- **General Questions**: Open a GitHub Discussion
- **Bug Reports**: Open a GitHub Issue
- **Commercial Licensing**: Contact focalways99@gmail.com
- **Research Collaboration**: Contact focalways99@gmail.com

## Recognition

Contributors will be acknowledged in:
- GitHub contributor list
- Release notes for significant contributions
- Special mentions for major features or research contributions

Thank you for helping make Affective RAG better!
