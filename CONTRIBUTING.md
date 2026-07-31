# Contributing

## Licence of contributions

By submitting a contribution you certify the Developer Certificate of Origin
1.1 (<https://developercertificate.org/>), and you sign off each commit:

```
git commit -s
```

which appends `Signed-off-by: Your Name <your@email>`. There is no CLA and
nothing to sign separately.

Contributions are made under **BSD-3-Clause**, the licence of this project. In
addition, you agree that the maintainer may release your contribution under any
licence approved by the Open Source Initiative.

Add yourself to `AUTHORS` in your first pull request. That file is what
"The Lenore Engine Authors" in `LICENSE` resolves to, so it is a copyright
record rather than a credits list.

That second sentence exists for one reason and it is worth stating plainly: the
project intends to move to Apache-2.0 after v2/v3, for its patent grant. A
licence change requires the agreement of every copyright holder, and each
contributor holds copyright in their own patch — so a project that does not ask
for this permission up front loses the ability to change licence the moment it
becomes popular enough to want to. Asking later means finding every contributor
who ever landed a patch. See `NOTICE.md`.

## No machine-generated code

Contributions must be written by their author. Machine-generated code is not
accepted, in whole or in part, and the DCO sign-off is your statement that the
contribution is your own work.

This is not an aesthetic rule. Material without human authorship is treated as
uncopyrightable in several jurisdictions, so generated code cannot be licensed
by anyone — it would leave holes in the grant that `LICENSE` makes. The
reasoning is in `NOTICE.md`.

Assistants are welcome for the things around the code: specifications, review,
documentation, build tooling.

## Before you open a pull request

- Every submodule builds and tests **from its own directory**, with no umbrella:
  `cd lenore-<module> && zig build test`. A module that only builds from the
  umbrella has lost its boundary, and that is a defect on its own.
- `zig fmt` is the formatter. No other style discussion is needed.
- Source files start with `// SPDX-License-Identifier: BSD-3-Clause`.
- Vulkan work is developed with validation layers enabled. A validation error is
  a bug, not a warning.
- Behaviour that only a GPU can demonstrate is not asserted from a build. Say
  what you ran and on what device.

## Scope

Each module owns one thing; `README.md` has the table. A change that does not
belong to the module it is proposed against will be asked to move rather than
merged, even when it is correct — the boundary between modules is the design.
