# PyPI release checklist for `kakeyalattice`

Status at 2026-04-25: package is build-verified + twine-check-PASSED,
but **NOT yet published to PyPI**. This doc is the exact sequence to
follow when ready.

## Pre-release

- [x] `pyproject.toml` has `version`, `description`, `classifiers`, URLs
- [x] `README.md` inside `kakeyalattice/` is PyPI-rendered (not the
      top-level repo README which contains binary artefacts / paper)
- [x] `LICENSE` file present
- [x] Unit tests pass: `pytest kakeyalattice/python/kakeyalattice/hf/test_cache.py`
- [x] `twine check dist/*` PASSED
- [x] Local install-from-wheel smoke test passes
- [ ] Maintainer agreement: who owns the PyPI namespace, who has 2FA
- [ ] First-time PyPI account signup and API token (one-off)
- [ ] Test upload to `testpypi.org` passes end-to-end
- [ ] Test install from `testpypi.org` in a fresh virtualenv:
      `pip install --index-url https://test.pypi.org/simple/ kakeyalattice`

## Release

```bash
# From repo root
cd kakeyalattice
rm -rf dist build *.egg-info python/kakeyalattice.egg-info
python -m build --installer pip
twine check dist/*

# First upload to testpypi to verify
twine upload --repository testpypi dist/*

# In a throwaway virtualenv:
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    'kakeyalattice[hf]'
python -c "from kakeyalattice.hf import KakeyaLatticeCache; print('OK')"

# If all good, upload to real PyPI
twine upload dist/*
```

## Post-release

- [ ] Tag the git release: `git tag v1.5.0 && git push origin v1.5.0`
- [ ] Create GitHub release with the same notes as `BLOG.md`
- [ ] Update Stage 0.75 FINDINGS / Stage 1 docs to include the
      `pip install kakeyalattice` one-liner
- [ ] Post to HF forum: `https://discuss.huggingface.co/c/research/7`
      with the blog post body
- [ ] Deploy the HF Space: follow `demos/hf_llama_kakeyalattice/README.md`
- [ ] Open a `huggingface/transformers` Discussion (not a PR) asking
      maintainers' interest in a future `KakeyaLatticeQuantizedCache`
      backend

## Version bump policy

- **Patch** (1.5.x): bug fixes, documentation, CI changes
- **Minor** (1.6.0): new public API, new codec variants, new host-model
  compatibility fixes
- **Major** (2.0.0): breaking API changes (e.g. codec interface change,
  rename of `V14/V15` classes)

The `v1.4` and `v1.5` naming in the package refers to **paper release
tags** (D4 and E8 codecs respectively), not to semver. Future codec
extensions (e.g. Leech lattice) would be `v1.6_leech` or similar,
published under package version `1.6.0`.

## Trusted publishing (recommended for future releases)

PyPI supports OpenID Connect / trusted publishing from GitHub Actions,
eliminating long-lived tokens. Setup in a follow-up:

1. Add a GitHub Actions workflow `.github/workflows/release.yml` that
   triggers on `v*` tags
2. Register the workflow as a trusted publisher on PyPI project settings
3. Use `pypa/gh-action-pypi-publish@release/v1`

This is the recommended path once the package has a first successful
manual release and maintainer agreement is in place.
