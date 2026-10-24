# Repository working conventions

- Commit titles use `X.X.X 改動說明` (version number followed by a concrete change description).
- Version numbering restarts at `0.1.0` for the ChatGPT-style interface/context update.
- Never increase the first (major) number unless the user explicitly calls the change a major release. Size or breaking changes alone do not authorize a major bump.
- Validate Python behavior with `python -m unittest discover -s tests -v`. Browser changes should also be checked at desktop and mobile widths.
