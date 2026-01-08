const normalizeBase = (value?: string | null): string | undefined => {
  if (!value) return undefined;
  const trimmed = value.trim();
  if (!trimmed || trimmed === "undefined" || trimmed === "null") {
    return undefined;
  }
  return trimmed.replace(/\/+$/, "");
};

const uniqueBases = (values: Array<string | undefined | null>): string[] => {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const normalized = normalizeBase(value);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    result.push(normalized);
  }
  return result;
};

const ENV_BASE =
  normalizeBase(import.meta.env.VITE_API_BASE_URL) ??
  normalizeBase(import.meta.env.VITE_API_BASE);

const FALLBACK_BASES = [
  "http://localhost:8080",
  "http://127.0.0.1:8080",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
];

const deriveWindowOrigin = (): string | undefined => {
  if (typeof window === "undefined") {
    return undefined;
  }
  try {
    return `${window.location.protocol}//${window.location.host}`;
  } catch {
    return undefined;
  }
};

const BASE_CANDIDATES = (() => {
  if (ENV_BASE) {
    return [ENV_BASE];
  }
  const candidates = uniqueBases([deriveWindowOrigin(), ...FALLBACK_BASES]);
  return candidates.length > 0 ? candidates : [FALLBACK_BASES[0]];
})();

let resolvedBase: string | null = ENV_BASE ?? null;

const joinUrl = (base: string, path: string): string => {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  const normalizedBase = base.endsWith("/") ? base.slice(0, -1) : base;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${normalizedBase}${normalizedPath}`;
};

const buildPreferenceList = (): string[] => {
  const ordered = resolvedBase
    ? uniqueBases([resolvedBase, ...BASE_CANDIDATES])
    : BASE_CANDIDATES;
  return ordered.length > 0 ? ordered : [FALLBACK_BASES[0]];
};

export const getActiveApiBase = (): string => {
  return resolvedBase ?? BASE_CANDIDATES[0];
};

export const apiFetch = async (
  path: string,
  init?: RequestInit
): Promise<Response> => {
  const candidates = buildPreferenceList();
  let lastError: unknown;

  for (const base of candidates) {
    try {
      const response = await fetch(joinUrl(base, path), init);
      resolvedBase = base;
      return response;
    } catch (error) {
      lastError = error;
      if (import.meta.env.DEV) {
        console.warn(
          `[PatentSearch] Failed to reach API at ${base}:`,
          error
        );
      }
    }
  }

  throw lastError ?? new Error("Patent Search API is unreachable");
};
