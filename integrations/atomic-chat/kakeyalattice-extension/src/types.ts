/**
 * Types mirroring the Python sidecar's `/v1/models` response.
 * Kept in sync with `kakeya_sidecar/schemas.py` and
 * `kakeya_sidecar/model_registry.py`.
 */

export interface KakeyaModelMeta {
  hf_repo_id: string;
  head_dim: number | number[];
  num_hidden_layers: number;
  variant: "d4" | "e8";
  q_range: number;
  boundary: number;
  est_compression: number;
  est_delta_ppl_pct: number | null;
  label: string;
  is_default: boolean;
  notes: string;
}

export interface KakeyaModel {
  id: string;                  // "<short>@<variant>-q<Q>[-b<B>]"
  object: "model";
  owned_by: "kakeyalattice";
  x_kakeya: KakeyaModelMeta;
}

export interface KakeyaModelList {
  object: "list";
  data: KakeyaModel[];
}

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string | string[];
  /** Runtime override for the per-model default channel. */
  x_kakeya_override?: {
    variant?: "d4" | "e8";
    q_range?: number;
    boundary?: number;
  };
}

export interface ChatCompletionChoice {
  index: number;
  message: { role: "assistant"; content: string };
  finish_reason: "stop" | "length";
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  x_kakeya?: Record<string, unknown>;
}

export interface SidecarStats {
  engine_loaded: boolean;
  device?: "mps" | "cuda" | "cpu";
  resident_models?: string[];
  max_resident?: number;
}
