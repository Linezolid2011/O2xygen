// Shared client constants. The client-side N-body propagator was DELETED in
// 0.1.5 (ADR-0021): the server is the only computer - positions arrive as
// SSE poses, and no client code may integrate orbits anymore.
const DAY = 86400.0;
const AU = 1.495978707e11;

export { DAY, AU };
