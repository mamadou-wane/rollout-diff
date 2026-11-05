import { useEffect, useState } from "react";
const TOKEN_KEY = "rolloutdiff.session.v1";
export function initializeToken(): string {
  const token = new URLSearchParams(window.location.hash.slice(1)).get("token");
  if (window.location.hash)
    window.history.replaceState(
      null,
      "",
      window.location.pathname + window.location.search,
    );
  if (token) sessionStorage.setItem(TOKEN_KEY, token);
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}
export async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set(
    "Authorization",
    `Bearer ${sessionStorage.getItem(TOKEN_KEY) ?? ""}`,
  );
  if (init.body) headers.set("Content-Type", "application/json");
  const response = await fetch(path, {
    ...init,
    headers,
    credentials: "omit",
    cache: "no-store",
    redirect: "error",
  });
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* Keep the status when no JSON error is provided. */
    }
    if (response.status === 401)
      message = "Session authorization expired. Reopen the local launch URL.";
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
export function post<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
}
export function useResource<T>(path: string | null) {
  const [state, setState] = useState<{
    path: string | null;
    data: T | null;
    error: string | null;
    loading: boolean;
  }>({ path: null, data: null, error: null, loading: false });
  const [revision, refresh] = useState(0);
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    let active = true;
    setState({ path, data: null, error: null, loading: true });
    request<T>(path, { signal: controller.signal })
      .then((data) => {
        if (active) setState({ path, data, error: null, loading: false });
      })
      .catch((error) => {
        if (active)
          setState({
            path,
            data: null,
            error: error instanceof Error ? error.message : "Request failed.",
            loading: false,
          });
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [path, revision]);
  return {
    ...(state.path === path
      ? state
      : { data: null, error: null, loading: !!path }),
    refresh: () => refresh((value) => value + 1),
  };
}
export async function downloadReport(path: string, name: string) {
  const response = await fetch(path, {
    headers: {
      Authorization: `Bearer ${sessionStorage.getItem(TOKEN_KEY) ?? ""}`,
    },
    credentials: "omit",
    cache: "no-store",
    redirect: "error",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `Export failed (${response.status}).`,
    );
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
