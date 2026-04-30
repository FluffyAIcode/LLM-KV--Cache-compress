/**
 * Atomic-Chat extension entry point.
 *
 * Usage (inside the Atomic-Chat host app bootstrap):
 *
 *   import { register as registerKakeya } from "@atomic-chat/kakeyalattice-extension";
 *   registerKakeya();
 *
 * The host app is expected to expose a global registry via
 * `window.AtomicChatCore?.registerBackend(backend)`. We fall back to
 * exporting the backend class directly so the host can wire it up
 * however it likes.
 */
import { KakeyaBackend } from "./backend";

export * from "./types";
export { KakeyaBackend };

type HostAPI = {
  registerBackend?: (backend: unknown) => void;
};

export function register(): KakeyaBackend {
  const backend = new KakeyaBackend();
  const host = (globalThis as unknown as { AtomicChatCore?: HostAPI }).AtomicChatCore;
  if (host?.registerBackend) {
    host.registerBackend(backend);
  }
  return backend;
}

export default register;
