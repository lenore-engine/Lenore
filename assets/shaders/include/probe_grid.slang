#pragma once

// Where the probes stand. Read by the field's lookup in probe_field.slang,
// carried in the lights block and in the probe update's push constants; a
// module that includes both of the first two declares it once.

// Mirrors the grid the host fitted: probe (0, 0, 0) at `origin`, `spacing`
// between neighbours, `counts` per axis, X fastest. `enabled` zero is no
// field, and the field's lookup is not called.
struct ProbeGridParams
{
    float3 origin;
    float spacing;
    uint3 counts;
    uint enabled;
};
