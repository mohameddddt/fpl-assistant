// Shared localStorage key names, so app-provider.tsx, chips/page.tsx and
// lib/data-portability.ts (export/import) all agree on where each piece of
// state actually lives instead of duplicating the literal strings.
export const SQUAD_STORAGE_KEY = "fpl-assistant-squad";
export const CHIP_INVENTORY_KEY = "fpl-assistant-chip-inventory";
// The chip plan the engine last returned for a full squad, so the My Team
// deadline board can name it without re-running the (slow) chip solve.
export const CHIP_PLAN_KEY = "fpl-assistant-chip-plan";

export interface StoredChipPlan {
  version: 1;
  season: string;
  gameweek: number;
  /** Sorted squad element ids the plan was computed for. */
  squad: string;
  plan: { chip: string; gw: number; gain: number | null }[];
}

export function chipPlanSquadKey(elements: number[]): string {
  return [...elements].sort((a, b) => a - b).join(",");
}

export function isStoredChipPlan(value: unknown): value is StoredChipPlan {
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  return record.version === 1 && typeof record.season === "string" &&
    typeof record.gameweek === "number" && typeof record.squad === "string" &&
    Array.isArray(record.plan) &&
    record.plan.every((entry) => typeof entry === "object" && entry !== null &&
      typeof (entry as Record<string, unknown>).chip === "string" &&
      typeof (entry as Record<string, unknown>).gw === "number");
}
