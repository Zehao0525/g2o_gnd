# Style / standards review: `slam_system_base`

Living checklist against `slam_system_base.hpp` / `.inl`.

---

## Done

1. ~~Add missing includes; drop unused ones.~~
2. ~~`virtual ~SlamSystemBase() = default`; `= delete` copy/move; `optimizer()` returns `SparseOptimizer&`.~~
3. ~~`.hpp` / `.inl` split; ctor calls `loadConfig` + `setupOptimizer`.~~
4. ~~Algorithm ownership: `make_unique` + `release()` into `setAlgorithm` (g2o owns it).~~
5. ~~Spacing / Doxygen typos / “Levenburg” TODO~~ — those lived on old `slam_system_base.h`; the new files are clean.

---

## Still open

1. clang-tidy (`cppcoreguidelines-*`, `bugprone-*`) pass — optional hygiene only.

---

## Config approach (intentional — not a defect)

- **JSON at runtime** is the right design: ~20 knobs in derived systems, edit without recompile.
- Base owns shared keys (`verbose`, opt period/algorithm/counts); derived own their own (GND, sensor offset, …).
- When wiring children onto this base: **parse the file once**, apply base fields, then a derived hook (e.g. `loadDerivedConfig(const nlohmann::json&)`) — avoid each layer re-opening the same file (old UTISA pattern).
- Named key constants / JSON Schema are **not** required for this project.

---

## Optional / not bugs

- Out-params `x`, `P` on `platformEstimateMarginals`: fine for g2o-style code.
- `[[nodiscard]]` on `optimize` is **not** recommended: side effect is the purpose.
- Soft-fail on marginals (`catch ...`, warn, zero `P`) vs fail-fast — document once if you care.
- Ctor throws on bad config — intentional.

---

## Design context

- Role: **templated abstract base + Template Method** (`processEvents` → `processEvent`), not a pure interface.
- File: header-only template → `.hpp` + `.inl`.
- Hierarchy intent: `SlamSystemBase` → `MultiRobotSlamSystemBase`; reuse `optimize` / `processEvents` / `setFixOlderPlatformVertices` on the base.
- Leaner long-term: keep the abstract base thin; derived configs stay in derived classes + JSON.
