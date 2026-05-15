"use client";

import { useState, useEffect, useRef, useCallback } from "react";

// ================================================================
// TYPES
// ================================================================

interface Prediction {
  theme: string;
  confidence: number;
}

interface TopPrediction {
  theme: string;
  confidence: number;
  is_uncertain: boolean;
}

interface PreprocessingSteps {
  segmentation?: string;
  case_folding?: string;
  tokenization?: string[];
  stopword_removal?: string[];
  stemming?: string[];
  final?: string;
}

interface RelatedPantun {
  tema: string;
  pantun: string;
}

interface ClassifyResponse {
  predictions: Prediction[];
  top_prediction: TopPrediction;
  model_used: string;
  use_pembayang: boolean;
  related_pantun: RelatedPantun[];
  preprocessing_steps?: PreprocessingSteps;
}

type ChatMessage =
  | { id: string; role: "welcome" }
  | { id: string; role: "user"; text: string }
  | { id: string; role: "loading" }
  | { id: string; role: "assistant"; results: Record<string, ClassifyResponse>; usedModels: string[] };

// ================================================================
// CONSTANTS
// ================================================================

const MODELS = [
  { id: "textcnn",                  name: "TextCNN",    short: "CNN"  },
  { id: "svm_90-10_no_pembayang",   name: "SVM 90/10",  short: "SVM"  },
  { id: "malaybert",                name: "MalayBERT",  short: "BERT" },
];

const CHAT_KEY  = "pantun-ai-chat";
const THEME_KEY = "pantun-ai-theme";

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

// ================================================================
// SUB-COMPONENTS
// ================================================================

function WelcomeBubble() {
  return (
    <div className="chat-row ai-row">
      <div className="ai-avatar">🪻</div>
      <div className="ai-bubble welcome-bubble">
        <p className="welcome-title">
          Selamat datang ke <strong>Pantun AI</strong>
        </p>
        <p className="welcome-body">
          Saya boleh mengenal pasti tema pantun Melayu menggunakan tiga model AI — TextCNN, SVM, dan MalayBERT. Taip atau tampal pantun anda di bawah.
        </p>
        <div className="welcome-examples">
          <p className="welcome-example-label">Contoh format</p>
          <code className="welcome-code">
            Buah manggis di dalam peti;<br />
            bawa ke pasar dijual orang;<br />
            Melentur buluh biarlah dari muda;<br />
            melentur sudah tidak payang.
          </code>
        </div>
      </div>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  const lines = text
    .split(/[;\n]/)
    .map((l) => l.trim())
    .filter(Boolean);

  return (
    <div className="chat-row user-row">
      <div className="user-bubble">
        <div className="pantun-lines">
          {lines.map((line, i) => (
            <p key={i} className={`pantun-line ${i < 2 ? "pembayang" : "maksud"}`}>
              {line}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}

function LoadingBubble() {
  return (
    <div className="chat-row ai-row">
      <div className="ai-avatar">🪻</div>
      <div className="ai-bubble">
        <div className="typing-indicator">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}

function ConfBar({ confidence }: { confidence: number }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(confidence), 60);
    return () => clearTimeout(t);
  }, [confidence]);

  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ width: `${width}%` }} />
      </div>
      <span className="conf-label">{confidence.toFixed(1)}%</span>
    </div>
  );
}

function AssistantBubble({
  results,
  usedModels,
}: {
  results: Record<string, ClassifyResponse>;
  usedModels: string[];
}) {
  const available = usedModels.filter((m) => results[m]);
  const [active, setActive] = useState(available[0] ?? "");

  if (!active || !results[active]) return null;

  const result = results[active];
  const top = result.top_prediction;
  const others = result.predictions.slice(1);

  return (
    <div className="chat-row ai-row">
      <div className="ai-avatar">🪻</div>
      <div className="ai-bubble result-bubble">
        {/* Model tabs */}
        {available.length > 1 && (
          <div className="model-tabs">
            {available.map((mid) => {
              const m = MODELS.find((x) => x.id === mid);
              return (
                <button
                  key={mid}
                  className={`model-tab ${active === mid ? "active" : ""}`}
                  onClick={() => setActive(mid)}
                >
                  {m?.short ?? mid}
                </button>
              );
            })}
          </div>
        )}

        {/* Theme badge */}
        <div className={`theme-badge-new ${top.is_uncertain ? "uncertain" : ""}`}>
          <span className="theme-badge-icon">{top.is_uncertain ? "⚠️" : "🏷️"}</span>
          <span className="theme-badge-text">{top.theme}</span>
        </div>

        {/* Confidence bar */}
        <ConfBar confidence={top.confidence} />

        {/* Alternate predictions */}
        {others.length > 0 && (
          <div className="alt-predictions">
            {others.map((pred, i) => (
              <div key={i} className="alt-pred-row">
                <span className="alt-pred-theme">{pred.theme}</span>
                <span className="alt-pred-conf">{pred.confidence.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}

        {/* Preprocessing steps */}
        {result.preprocessing_steps &&
          Object.keys(result.preprocessing_steps).length > 0 && (
            <details className="steps-details">
              <summary className="steps-summary">Langkah Pemprosesan</summary>
              <div className="steps-list">
                {Object.entries(result.preprocessing_steps).map(([key, val]) => (
                  <div key={key} className="step-row">
                    <span className="step-key">{key}</span>
                    <span className="step-val">
                      {Array.isArray(val) ? val.join(" → ") : String(val)}
                    </span>
                  </div>
                ))}
              </div>
            </details>
          )}

        {/* Related pantun */}
        {result.related_pantun?.length > 0 && (
          <div className="related-section">
            <p className="related-header">Pantun Berkaitan</p>
            {result.related_pantun.map((p, i) => (
              <div key={i} className="related-item">
                {p.pantun
                  .split(/[;,]/)
                  .map((line) => line.trim())
                  .filter(Boolean)
                  .map((line, li) => (
                    <span key={li} className="related-line">
                      {line}
                      <br />
                    </span>
                  ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ================================================================
// MAIN PAGE
// ================================================================

export default function Home() {
  const [messages, setMessages]         = useState<ChatMessage[]>([]);
  const [input, setInput]               = useState("");
  const [isDarkMode, setIsDarkMode]     = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [isLoading, setIsLoading]       = useState(false);

  // Settings
  const [selectedModels, setSelectedModels]             = useState<string[]>(["textcnn", "svm_90-10_no_pembayang", "malaybert"]);
  const [topK, setTopK]                                 = useState(3);
  const [showSteps, setShowSteps]                       = useState(false);
  const [confidenceThreshold, setConfidenceThreshold]   = useState(20);
  const [relatedCount, setRelatedCount]                 = useState(0);

  const chatEndRef  = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const hydratedRef = useRef(false);

  // ---- Hydrate from localStorage ----
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    const savedTheme = localStorage.getItem(THEME_KEY);
    if (savedTheme) setIsDarkMode(savedTheme === "dark");

    const saved = localStorage.getItem(CHAT_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as ChatMessage[];
        setMessages(parsed.length > 0 ? parsed : [{ id: "welcome", role: "welcome" }]);
      } catch {
        setMessages([{ id: "welcome", role: "welcome" }]);
      }
    } else {
      setMessages([{ id: "welcome", role: "welcome" }]);
    }
    hydratedRef.current = true;
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  // ---- Persist theme ----
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", isDarkMode ? "dark" : "light");
    if (hydratedRef.current) localStorage.setItem(THEME_KEY, isDarkMode ? "dark" : "light");
  }, [isDarkMode]);

  // ---- Persist chat (skip loading bubbles) ----
  useEffect(() => {
    if (!hydratedRef.current || messages.length === 0) return;
    const toSave = messages.filter((m) => m.role !== "loading");
    localStorage.setItem(CHAT_KEY, JSON.stringify(toSave));
  }, [messages]);

  // ---- Auto-scroll ----
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ---- Toggle model selection ----
  const toggleModel = useCallback((modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.length > 1 ? prev.filter((m) => m !== modelId) : prev
        : [...prev, modelId]
    );
  }, []);

  // ---- Clear history ----
  const clearHistory = useCallback(() => {
    setMessages([{ id: "welcome", role: "welcome" }]);
    localStorage.removeItem(CHAT_KEY);
  }, []);

  // ---- Submit ----
  const handleSubmit = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    // Normalise: newlines → "; " for the backend
    const apiText = text.replace(/\n/g, "; ");

    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setIsLoading(true);

    setMessages((prev) => [
      ...prev,
      { id: uid(), role: "user",    text },
      { id: uid(), role: "loading"        },
    ]);

    const results: Record<string, ClassifyResponse> = {};
    await Promise.all(
      selectedModels.map(async (modelId) => {
        try {
          const resp = await fetch("/api/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              pantun: apiText,
              model: modelId,
              use_pembayang: false,
              top_k: topK,
              show_steps: showSteps,
              related_count: relatedCount,
              confidence_threshold: confidenceThreshold,
            }),
          });
          if (resp.ok) results[modelId] = await resp.json();
        } catch {
          // Model unavailable — silently skip
        }
      })
    );

    setMessages((prev) => [
      ...prev.filter((m) => m.role !== "loading"),
      { id: uid(), role: "assistant", results, usedModels: [...selectedModels] },
    ]);
    setIsLoading(false);
  }, [input, isLoading, selectedModels, topK, showSteps, relatedCount, confidenceThreshold]);

  // ---- Keyboard shortcut ----
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // ---- Auto-grow textarea ----
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  };

  // ================================================================
  // RENDER
  // ================================================================
  return (
    <div className="chat-app">

      {/* ---- HEADER ---- */}
      <header className="chat-header">
        <div className="chat-header-inner">
          <div className="logo-group">
            <span className="logo-emoji">🪻</span>
            <div className="logo-text">
              <span className="logo-name">Pantun AI</span>
              <span className="logo-tagline">Pengelas Tema Melayu</span>
            </div>
          </div>

          <div className="header-controls">
            <button
              className={`icon-btn ${settingsOpen ? "active" : ""}`}
              onClick={() => setSettingsOpen((v) => !v)}
              title="Tetapan"
            >
              ⚙️
            </button>
            <button
              className="icon-btn"
              onClick={() => setIsDarkMode((v) => !v)}
              title={isDarkMode ? "Mod Cerah" : "Mod Gelap"}
            >
              {isDarkMode ? "☀️" : "🌙"}
            </button>
            <button
              className="icon-btn"
              onClick={clearHistory}
              title="Kosongkan Sejarah"
            >
              🗑️
            </button>
          </div>
        </div>

        {/* Settings Drawer */}
        {settingsOpen && (
          <div className="settings-drawer">
            <div className="settings-section">
              <p className="settings-label">Model Aktif</p>
              <div className="model-checkboxes">
                {MODELS.map((m) => (
                  <label
                    key={m.id}
                    className={`model-checkbox ${selectedModels.includes(m.id) ? "checked" : ""}`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(m.id)}
                      onChange={() => toggleModel(m.id)}
                    />
                    {m.name}
                  </label>
                ))}
              </div>
            </div>

            <div className="settings-row">
              <div className="settings-field">
                <label className="settings-label">Ramalan Teratas (K)</label>
                <select
                  className="settings-select"
                  value={topK}
                  onChange={(e) => setTopK(+e.target.value)}
                >
                  <option value={1}>1</option>
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                </select>
              </div>

              <div className="settings-field">
                <label className="settings-label">Pantun Berkaitan</label>
                <select
                  className="settings-select"
                  value={relatedCount}
                  onChange={(e) => setRelatedCount(+e.target.value)}
                >
                  <option value={0}>Tiada</option>
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                </select>
              </div>

              <div className="settings-field">
                <label className="settings-label">Had Keyakinan (%)</label>
                <select
                  className="settings-select"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(+e.target.value)}
                >
                  <option value={0}>Tiada</option>
                  <option value={20}>20%</option>
                  <option value={40}>40%</option>
                  <option value={60}>60%</option>
                </select>
              </div>

              <div className="settings-field toggle-field">
                <label className="settings-label">Langkah Pemprosesan</label>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={showSteps}
                    onChange={(e) => setShowSteps(e.target.checked)}
                  />
                  <span className="toggle-track" />
                </label>
              </div>
            </div>
          </div>
        )}
      </header>

      {/* ---- CHAT WINDOW ---- */}
      <div className="chat-window">
        {messages.map((msg) => {
          if (msg.role === "welcome")   return <WelcomeBubble key={msg.id} />;
          if (msg.role === "user")      return <UserBubble    key={msg.id} text={msg.text} />;
          if (msg.role === "loading")   return <LoadingBubble key={msg.id} />;
          if (msg.role === "assistant") return (
            <AssistantBubble key={msg.id} results={msg.results} usedModels={msg.usedModels} />
          );
          return null;
        })}
        <div ref={chatEndRef} />
      </div>

      {/* ---- INPUT PANEL ---- */}
      <div className="input-panel">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Masukkan pantun anda… (pisahkan baris dengan ';' atau Enter)"
            rows={1}
            disabled={isLoading}
          />
          <button
            className="send-btn"
            onClick={handleSubmit}
            disabled={isLoading || !input.trim()}
            title="Hantar (Ctrl+Enter)"
          >
            {isLoading ? <span className="send-spinner" /> : <span>→</span>}
          </button>
        </div>
        <p className="input-hint">Ctrl+Enter untuk hantar • Baris dipisah dengan &apos;;&apos; atau Enter</p>
      </div>
    </div>
  );
}
