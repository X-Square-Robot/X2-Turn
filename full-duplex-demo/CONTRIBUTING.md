# Contributing

Thank you for contributing.

1. Open an issue before large behavior or protocol changes.
2. Create a focused branch and keep model weights, recordings, credentials,
   generated certificates, logs, and local environment files out of Git.
3. Install development tools with `pip install -e '.[demo,llm,dev]'`.
4. Run `python3 -m compileall dialogue_system voxtral_bridge` and
   `bash -n start_demo.sh scripts/run_app.sh` before submitting a change.
5. Describe the hardware, model IDs, and manual full-duplex checks used.

Contributions must be your own work or compatible with Apache License 2.0.
Do not copy non-commercial, private, or otherwise incompatible material into
the repository. By submitting a contribution, you agree that it is licensed
under the repository license.
