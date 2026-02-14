# Publishing langchain-kalibr to PyPI

## Prerequisites

```bash
pip install build twine
```

## Build and Publish

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build
python -m build

# Test upload (optional)
twine upload --repository testpypi dist/*

# Production upload
twine upload dist/*
```

## PyPI Credentials

Use your existing Kalibr PyPI credentials (same account as the `kalibr` package).

Set up `~/.pypirc`:
```ini
[pypi]
username = kalibr
password = <your-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = kalibr
password = <your-test-token>
```

Or use environment variables:
```bash
export TWINE_USERNAME=kalibr
export TWINE_PASSWORD=<your-token>
twine upload dist/*
```

## Post-Publish Checklist

1. ✅ Verify on PyPI: https://pypi.org/project/langchain-kalibr/
2. ✅ Test install: `pip install langchain-kalibr`
3. ✅ Submit PR to LangChain docs (see docs/langchain-provider-page.md)
4. ✅ Submit PR to LangChain chat integrations (see docs/langchain-chat-integration.md)
5. ✅ Post on CrewAI community (see docs/crewai-community-post.md)
6. ✅ Submit awesome-list PRs (see docs/awesome-list-submissions.md)
7. ✅ Add AGENTS.md to kalibr-sdk-python repo
8. ✅ Add llms.txt to kalibr.systems docs site
