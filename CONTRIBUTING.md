# Contributing

Thank you for contributing to X2 Turn.

## Before opening a change

1. Open an issue before large model, protocol, or architecture changes.
2. Create a focused branch and keep model weights, recordings, credentials,
   generated certificates, logs, caches, and local environment files out of
   Git.
3. Use the component-specific development extras defined in each
   `pyproject.toml`.
4. Run the root release checks:

   ```bash
   python scripts/check_release_language.py
   python scripts/check_public_release.py
   python scripts/check_environments.py
   ```

5. Run tests and Ruff for every changed component. For full-duplex changes,
   also report the hardware, model IDs, and manual browser checks used.

Contributions must be your own work or use material compatible with Apache
License 2.0 and the component notices. Do not copy private, non-commercial, or
otherwise incompatible material into this repository. By submitting a
contribution, you agree that it is licensed under the repository license.
