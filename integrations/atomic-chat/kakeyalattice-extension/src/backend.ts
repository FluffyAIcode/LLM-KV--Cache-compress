/**
 * KakeyaLattice backend — Atomic-Chat `Backend` implementation.
 *
 * We do NOT import `@atomic-chat/core` at compile time (its exact
 * shape is governed by the host app). Instead we declare the minimal
 * interface we need as a local type, and the host will use structural
 * typing via `registerBackend(new KakeyaBackend())`.
 */
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  KakeyaModel,
  KakeyaModelList,
  SidecarStats,
} from "./types";

/**
 * Minimal shape of Atomic-Chat's backend contract. Keep in sync with
 * the host app's `core/src/backend.ts` as it evolves.
 */
export interface Backend {
  readonly id: string;
  readonly displayName: string;
  readonly kind: "local" | "cloud";
  listModels(): Promise<KakeyaModel[]>;
  chatCompletion(req: ChatCompletionRequest): Promise<ChatCompletionResponse>;
  chatCompletionStream(
    req: ChatCompletionRequest,
    onDelta: (chunk: string) => void,
  ): Promise<void>;
  healthCheck(): Promise<boolean>;
  getStats?(): Promise<SidecarStats>;
}

/**
 * Tauri invoke wrapper — resolved at runtime so the package can still
 * `tsc --noEmit` in a node-only CI where `@tauri-apps/api` isn't
 * available.
 */
type InvokeFn = <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;

async function tauriInvoke(): Promise<InvokeFn> {
  // Prefer the runtime-injected global when running inside Tauri.
  const win = globalThis as unknown as {
    __TAURI__?: { invoke: InvokeFn };
    __TAURI_INTERNALS__?: { invoke: InvokeFn };
  };
  if (win.__TAURI__?.invoke) return win.__TAURI__.invoke;
  if (win.__TAURI_INTERNALS__?.invoke) return win.__TAURI_INTERNALS__.invoke;
  // Fallback to the dynamic import (ESM, requires @tauri-apps/api).
  const mod = await import("@tauri-apps/api/core");
  return mod.invoke as InvokeFn;
}

export class KakeyaBackend implements Backend {
  readonly id = "kakeyalattice";
  readonly displayName = "KakeyaLattice (E8 KV-cache compression)";
  readonly kind = "local" as const;

  async listModels(): Promise<KakeyaModel[]> {
    const invoke = await tauriInvoke();
    const list = await invoke<KakeyaModelList>(
      "plugin:kakeyalattice|list_models",
    );
    return list.data;
  }

  async chatCompletion(req: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    const invoke = await tauriInvoke();
    return invoke<ChatCompletionResponse>(
      "plugin:kakeyalattice|chat_completion",
      { request: { ...req, stream: false } },
    );
  }

  async chatCompletionStream(
    req: ChatCompletionRequest,
    onDelta: (chunk: string) => void,
  ): Promise<void> {
    const invoke = await tauriInvoke();
    // The Rust plugin streams via a Tauri event named after the request id.
    const streamId = await invoke<string>(
      "plugin:kakeyalattice|chat_completion_stream_start",
      { request: { ...req, stream: true } },
    );

    // Listen on the event channel. `@tauri-apps/api/event` is resolved lazily.
    const eventMod = await import("@tauri-apps/api/event");
    const unlisten = await eventMod.listen<string>(
      `kakeyalattice:${streamId}`,
      (e) => {
        if (e.payload === "__DONE__") {
          unlisten();
          return;
        }
        onDelta(e.payload);
      },
    );

    return invoke<void>("plugin:kakeyalattice|chat_completion_stream_wait", {
      streamId,
    });
  }

  async healthCheck(): Promise<boolean> {
    try {
      const invoke = await tauriInvoke();
      const ok = await invoke<{ ok: boolean }>("plugin:kakeyalattice|health");
      return ok.ok === true;
    } catch {
      return false;
    }
  }

  async getStats(): Promise<SidecarStats> {
    const invoke = await tauriInvoke();
    return invoke<SidecarStats>("plugin:kakeyalattice|stats");
  }
}
