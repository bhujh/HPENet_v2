# Decisions - deploy-cpp-port

## Architecture Decisions
- Code in deploy/CPP/ separate from existing deploy/*.py
- CMake build system (not setup.py/PyTorch Extension)
- TDD with GoogleTest + golden data from Python
