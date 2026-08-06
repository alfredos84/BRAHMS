# Security Policy

## Supported Versions

BRAHMS is developed on a single rolling `main` branch. Only the latest
commit on `main` is supported; there are no maintained older release
branches.

| Version         | Supported          |
| ---------------- | ------------------ |
| `main` (latest)  | :white_check_mark: |
| older commits    | :x:                |

## Reporting a Vulnerability

BRAHMS is a desktop simulation tool: it reads a local JSON configuration
file, runs a local C++/CUDA or C++/OpenMP binary, and writes local HDF5
output files. It does not open network ports, accept remote input, or run
with elevated privileges, so its attack surface is limited. That said, if
you find a security issue (e.g., unsafe handling of a malformed
configuration or crystal-database file), please report it privately
instead of opening a public issue:

- Email: alfredo.daniel.sanchez@gmail.com
- Please include: the version/commit hash, your OS, and steps to
  reproduce.

We aim to acknowledge reports within a few days. Once a fix is available,
it will be merged into `main` and credited in the commit message (unless
you prefer to remain anonymous).
