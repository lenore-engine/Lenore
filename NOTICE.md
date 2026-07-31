# Notice

Lenore — a Vulkan rendering engine in Zig.
Copyright (c) 2026 The Lenore Engine Authors, listed in `AUTHORS`.
Licensed under BSD-3-Clause; see `LICENSE`.

Named after Bürger's ballad *Lenore* (1773) by way of Poe's raven.

## Provenance

Every line of engine code and every test in this repository and its submodules
is written by hand. No machine-generated code is committed.

This is a deliberate policy with a practical reason behind it, not only a
stylistic one. Material produced without human authorship is treated as
uncopyrightable in several jurisdictions — the United States Copyright Office
declined registration for such material in its 2023 guidance, and *Thaler v.
Perlmutter* (D.D.C. 2023, aff'd D.C. Cir. 2025) held human authorship to be a
statutory requirement. A tree that mixes generated and written code therefore
carries patches over which no copyright can be asserted, and a licence grant is
only as sound as the copyright underneath it.

Assistants are used in this project for specifications, review, documentation
and tooling. They do not author engine code.

## Trademark

"Lenore" as the name of this project is not licensed by `LICENSE`, which covers
copyright only. BSD-3-Clause clause 3 already forbids using the copyright
holder's name to endorse or promote derived products; forks are welcome and are
asked to carry their own name.

## Intended relicensing

The project is expected to move to Apache-2.0 after v2/v3, for its explicit
patent grant and its well-defined `NOTICE` mechanism. Contributions are accepted
on terms that keep this possible — see `CONTRIBUTING.md`. Releases made before
that point remain available under BSD-3-Clause; nothing is withdrawn.

## Third-party dependencies

Dependencies are consumed unmodified and keep their own licences. Their terms
are not affected by this file:

| Dependency | Used for |
|---|---|
| vulkan-zig (Snektron) | Vulkan bindings generated from `vk.xml` |
| zmath (zig-gamedev) | SIMD math |
| zglfw (zig-gamedev) | the current platform backend, to be replaced |

The Vulkan registry (`vk/vk.xml`) is published by The Khronos Group under its
own terms.
