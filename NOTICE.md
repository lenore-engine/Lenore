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

Dependencies keep their own licences. Their terms are not affected by this file.

| Dependency | Licence | Used for |
|---|---|---|
| vulkan-zig | MIT, © Robin Voetter | Vulkan bindings generated from `vk.xml` |
| zmath (zig-gamedev) | MIT, © 2021 Michal Ziulek, © 2024 zig-gamedev contributors | SIMD math |
| zignal | MIT, © 2024–2026 B Factory Inc, © Zignal Contributors | PNG and JPEG decoding of source images |
| FreeType 2.14.3 | FreeType Licence, chosen over GPLv2 | glyph rasterisation, in `lenore-text` |
| HarfBuzz 14.3.0 | "Old MIT", as its `COPYING` calls it | text shaping, in `lenore-text` |
| bc7enc, Richard Geldreich, Jr. | MIT or public domain | BC7 encoding, vendored in `lenore-ktx` |

vulkan-zig and zmath are fetched from forks under `lenore-engine`, each one
commit over the upstream revision it pins, which builds it on Zig 0.17.0;
licence and copyright are upstream's. The rest are fetched as released, except bc7enc,
which `lenore-ktx` vendors unmodified. `lenore-text/NOTICE.md` and
`lenore-ktx/NOTICE.md` carry the detail for the code those modules compile.

The MIT-style licences ask that the copyright and permission notices travel with
copies and substantial portions; each fetched package carries its `LICENSE` and
this table names them. None of them imposes a term BSD-3-Clause does not already
accept, and none constrains the intended move to Apache-2.0.

The FreeType Licence asks a program that uses FreeType to say so in its
documentation, and a build of this engine compiles it:

> Portions of this software are copyright © 2026 The FreeType Project
> (https://freetype.org). All rights reserved.

The Vulkan registry (`lenore-gpu/vk/vk.xml`) is published by The Khronos Group
under Apache-2.0 OR MIT, as its own header states.
