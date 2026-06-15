/**
 * Ambient type shims for Tauri modules. When this package is built
 * inside the Atomic-Chat host app, the real `@tauri-apps/api` types
 * take precedence (via node_modules). These shims exist so the package
 * also typechecks in a minimal CI where the peer dependency isn't
 * installed.
 */
declare module "@tauri-apps/api/core" {
  export function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T>;
}

declare module "@tauri-apps/api/event" {
  export interface Event<T> {
    event: string;
    id: number;
    payload: T;
  }
  export type UnlistenFn = () => void;
  export function listen<T>(
    event: string,
    handler: (e: Event<T>) => void,
  ): Promise<UnlistenFn>;
}
