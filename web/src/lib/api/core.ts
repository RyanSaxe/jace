/** Core typed JSON fetch utilities and card image URL builder for Jace Studio API modules. */

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export async function fetchJson<T>(
  url: string,
  guard: (data: unknown) => data is T,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`API ${response.status}: ${response.statusText}`);
  }
  const data: unknown = await response.json();
  if (!guard(data)) {
    throw new Error(`Unexpected response shape from ${url}`);
  }
  return data;
}

export function extractApiDetail(data: unknown, fallback: string): string {
  if (isRecord(data) && typeof data.detail === "string" && data.detail.length > 0) {
    return data.detail;
  }
  return fallback;
}

export function cardImageUrl(name: string): string {
  return `/api/cards/${encodeURIComponent(name)}/image`;
}
