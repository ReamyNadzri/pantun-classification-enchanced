"use client";

import { useState, useEffect, useCallback } from "react";

// Types
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

interface ModelInfo {
  key: string;
  type: string;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    split?: string;
  };
}

// API base URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

export default function Home() {
  // Input state
  const [lines, setLines] = useState(["", "", "", ""]);

  // Settings
  const [usePembayang, setUsePembayang] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  const [topK, setTopK] = useState(3);
  const [showSteps, setShowSteps] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(20);
  const [relatedCount, setRelatedCount] = useState(5);
  const [splitRatio, setSplitRatio] = useState("90-10");
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Results
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // Available models
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Load available models on mount
  useEffect(() => {
    fetch(`${API_URL}/api/models`)
      .then((res) => res.json())
      .then((data) => {
        setAvailableModels(data.models || []);
        if (data.models?.length > 0) {
          // Default to best model
          const bestModel = data.models.find(
            (m: ModelInfo) => m.key.includes("90-10") && m.key.includes("no_pembayang")
          );
          setSelectedModel(bestModel?.key || data.models[0].key);
        }
      })
      .catch(() => { });
  }, []);

  // Theme toggle
  useEffect(() => {
    document.documentElement.setAttribute(
      "data-theme",
      isDarkMode ? "dark" : "light"
    );
  }, [isDarkMode]);

  // Auto-select model based on settings
  useEffect(() => {
    const suffix = usePembayang ? "pembayang" : "no_pembayang";
    const key = `svm_${splitRatio}_${suffix}`;
    const exists = availableModels.find((m) => m.key === key);
    if (exists) {
      setSelectedModel(key);
    }
  }, [usePembayang, splitRatio, availableModels]);

  const handleLineChange = (index: number, value: string) => {
    const newLines = [...lines];
    newLines[index] = value;
    setLines(newLines);
  };

  const isValid = lines.every((line) => line.trim().length > 0);

  const handleClassify = useCallback(async () => {
    if (!isValid) return;

    setIsLoading(true);
    setError("");
    setResult(null);

    const pantunText = lines.join("; ");

    try {
      const response = await fetch(`${API_URL}/api/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pantun: pantunText,
          model: selectedModel,
          use_pembayang: usePembayang,
          top_k: topK,
          show_steps: showSteps,
          related_count: relatedCount,
          confidence_threshold: confidenceThreshold,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: ClassifyResponse = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Ralat berlaku. Pastikan backend sedang berjalan."
      );
    } finally {
      setIsLoading(false);
    }
  }, [lines, isValid, selectedModel, usePembayang, topK, showSteps, relatedCount, confidenceThreshold]);

  const handleClear = () => {
    setLines(["", "", "", ""]);
    setResult(null);
    setError("");
  };

  const getModelMetrics = () => {
    const model = availableModels.find((m) => m.key === selectedModel);
    return model?.metrics || {};
  };

  return (
    <>
      {/* Header */}
      <header className="header">
        <div className="container header-content">
          <div className="logo">
            <span className="logo-icon">🪻</span>
            <div>
              <h1>Klasifikasi Pantun</h1>
              <span className="logo-subtitle">SVM Theme Classifier</span>
            </div>
          </div>
          <button
            className="theme-toggle-btn"
            onClick={() => setIsDarkMode(!isDarkMode)}
            title={isDarkMode ? "Light Mode" : "Dark Mode"}
          >
            {isDarkMode ? "☀️" : "🌙"}
          </button>
        </div>
      </header>

      <main>
        <div className="container">
          {/* Hero */}
          <section className="hero">
            <h2>Klasifikasi Tema Pantun Melayu</h2>
            <p>
              Masukkan empat baris pantun anda untuk mengenal pasti tema menggunakan
              algoritma Support Vector Machine (SVM).
            </p>
          </section>

          {/* Main Grid */}
          <div className="main-grid">
            {/* Left Column - Input */}
            <div>
              {/* Pantun Input Card */}
              <div className="card">
                <h3 className="card-title">✍️ Masukkan Pantun</h3>
                <div className="pantun-input-group">
                  {lines.map((line, index) => (
                    <div className="line-input-wrapper" key={index}>
                      <span className="line-label">
                        {index < 2 ? "Pemb" : "Maks"} {index + 1}
                      </span>
                      <input
                        type="text"
                        className={`line-input ${index < 2 ? "pembayang" : "maksud"}`}
                        placeholder={
                          index === 0
                            ? "Baris pertama (pembayang)..."
                            : index === 1
                              ? "Baris kedua (pembayang)..."
                              : index === 2
                                ? "Baris ketiga (maksud)..."
                                : "Baris keempat (maksud)..."
                        }
                        value={line}
                        onChange={(e) => handleLineChange(index, e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && isValid) handleClassify();
                        }}
                      />
                    </div>
                  ))}
                </div>

                {/* Buttons */}
                <div className="btn-group">
                  <button
                    className="btn btn-primary"
                    onClick={handleClassify}
                    disabled={!isValid || isLoading}
                  >
                    {isLoading ? (
                      <>
                        <span className="spinner" /> Mengklasifikasi...
                      </>
                    ) : (
                      <>🔍 Klasifikasi</>
                    )}
                  </button>
                  <button className="btn btn-secondary" onClick={handleClear}>
                    🗑️ Padam
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => setSettingsOpen(!settingsOpen)}
                  >
                    ⚙️ Tetapan
                  </button>
                </div>
              </div>

              {/* Settings Card (collapsible) */}
              {settingsOpen && (
                <div className="card" style={{ marginTop: "1rem" }}>
                  <h3 className="card-title">⚙️ Tetapan Klasifikasi</h3>
                  <div className="settings-grid">
                    {/* Model Selector */}
                    <div className="setting-item">
                      <span className="setting-label">🔀 Model</span>
                      <select
                        className="setting-select"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                      >
                        {availableModels.map((m) => (
                          <option key={m.key} value={m.key}>
                            {m.key} ({((m.metrics.f1_score || 0) * 100).toFixed(1)}% F1)
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Split Ratio */}
                    <div className="setting-item">
                      <span className="setting-label">📈 Nisbah Data</span>
                      <select
                        className="setting-select"
                        value={splitRatio}
                        onChange={(e) => setSplitRatio(e.target.value)}
                      >
                        <option value="70-30">70% Latihan - 30% Ujian</option>
                        <option value="80-20">80% Latihan - 20% Ujian</option>
                        <option value="90-10">90% Latihan - 10% Ujian</option>
                      </select>
                    </div>

                    {/* Pembayang Toggle */}
                    <div className="setting-item">
                      <div className="toggle-wrapper">
                        <span className="setting-label">🔧 Guna Pembayang</span>
                        <label className="toggle">
                          <input
                            type="checkbox"
                            checked={usePembayang}
                            onChange={(e) => setUsePembayang(e.target.checked)}
                          />
                          <span className="toggle-slider"></span>
                        </label>
                      </div>
                    </div>

                    {/* Show Preprocessing */}
                    <div className="setting-item">
                      <div className="toggle-wrapper">
                        <span className="setting-label">🔍 Tunjuk Proses</span>
                        <label className="toggle">
                          <input
                            type="checkbox"
                            checked={showSteps}
                            onChange={(e) => setShowSteps(e.target.checked)}
                          />
                          <span className="toggle-slider"></span>
                        </label>
                      </div>
                    </div>

                    {/* Top-K */}
                    <div className="setting-item">
                      <span className="setting-label">📊 Top-K Ramalan</span>
                      <select
                        className="setting-select"
                        value={topK}
                        onChange={(e) => setTopK(Number(e.target.value))}
                      >
                        <option value={1}>Top 1</option>
                        <option value={3}>Top 3</option>
                        <option value={5}>Top 5</option>
                      </select>
                    </div>

                    {/* Related Count */}
                    <div className="setting-item">
                      <span className="setting-label">📚 Pantun Berkaitan</span>
                      <select
                        className="setting-select"
                        value={relatedCount}
                        onChange={(e) => setRelatedCount(Number(e.target.value))}
                      >
                        <option value={0}>Tiada</option>
                        <option value={3}>3 pantun</option>
                        <option value={5}>5 pantun</option>
                        <option value={10}>10 pantun</option>
                      </select>
                    </div>

                    {/* Confidence Threshold */}
                    <div className="setting-item" style={{ gridColumn: "1 / -1" }}>
                      <span className="setting-label">
                        🎯 Ambang Keyakinan: {confidenceThreshold}%
                      </span>
                      <input
                        type="range"
                        className="setting-range"
                        min={0}
                        max={80}
                        step={5}
                        value={confidenceThreshold}
                        onChange={(e) =>
                          setConfidenceThreshold(Number(e.target.value))
                        }
                      />
                    </div>
                  </div>

                  {/* Current Model Metrics */}
                  {selectedModel && (
                    <div style={{ marginTop: "1.25rem" }}>
                      <span
                        className="setting-label"
                        style={{ marginBottom: "0.5rem", display: "block" }}
                      >
                        📋 Prestasi Model Semasa
                      </span>
                      <div className="metrics-grid">
                        <div className="metric-card">
                          <div className="metric-value">
                            {((getModelMetrics().accuracy || 0) * 100).toFixed(1)}%
                          </div>
                          <div className="metric-label">Accuracy</div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-value">
                            {((getModelMetrics().precision || 0) * 100).toFixed(1)}%
                          </div>
                          <div className="metric-label">Precision</div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-value">
                            {((getModelMetrics().recall || 0) * 100).toFixed(1)}%
                          </div>
                          <div className="metric-label">Recall</div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-value">
                            {((getModelMetrics().f1_score || 0) * 100).toFixed(1)}%
                          </div>
                          <div className="metric-label">F1-Score</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right Column - Results */}
            <div>
              {/* Error */}
              {error && (
                <div
                  className="card"
                  style={{
                    borderColor: "rgba(239, 68, 68, 0.3)",
                    marginBottom: "1rem",
                  }}
                >
                  <h3 className="card-title" style={{ color: "#ef4444" }}>
                    ❌ Ralat
                  </h3>
                  <p style={{ color: "var(--text-secondary)" }}>{error}</p>
                </div>
              )}

              {/* Result */}
              {result && (
                <div className="result-section">
                  {/* Top Prediction */}
                  <div className="card" style={{ marginBottom: "1rem" }}>
                    <h3 className="card-title">🎯 Keputusan Klasifikasi</h3>
                    <div style={{ textAlign: "center", margin: "1rem 0" }}>
                      <span
                        className={`theme-badge ${result.top_prediction.is_uncertain ? "uncertain" : ""
                          }`}
                      >
                        {result.top_prediction.is_uncertain
                          ? "⚠️ Tidak Pasti"
                          : `🏷️ ${result.top_prediction.theme}`}
                      </span>
                    </div>

                    {/* Confidence Bar */}
                    <div className="confidence-bar-container">
                      <div className="confidence-text">
                        <span>Keyakinan</span>
                        <span style={{ fontWeight: 600 }}>
                          {result.top_prediction.confidence.toFixed(1)}%
                        </span>
                      </div>
                      <div className="confidence-bar">
                        <div
                          className="confidence-bar-fill"
                          style={{
                            width: `${result.top_prediction.confidence}%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* Model Info */}
                    <div
                      style={{
                        marginTop: "1rem",
                        fontSize: "0.8rem",
                        color: "var(--text-muted)",
                      }}
                    >
                      Model: <strong>{result.model_used}</strong> | Pembayang:{" "}
                      <strong>{result.use_pembayang ? "ON" : "OFF"}</strong>
                    </div>
                  </div>

                  {/* Top-K Predictions */}
                  {result.predictions.length > 1 && (
                    <div className="card" style={{ marginBottom: "1rem" }}>
                      <h3 className="card-title">📊 Top Ramalan</h3>
                      <div className="predictions-list">
                        {result.predictions.map((pred, i) => (
                          <div className="prediction-item" key={i}>
                            <span className="theme-name">
                              {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : `#${i + 1}`}{" "}
                              {pred.theme}
                            </span>
                            <span className="confidence">
                              {pred.confidence.toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Preprocessing Steps */}
                  {result.preprocessing_steps && (
                    <div className="card" style={{ marginBottom: "1rem" }}>
                      <h3 className="card-title">🔬 Langkah Prapemprosesan</h3>
                      <div className="steps-container">
                        {Object.entries(result.preprocessing_steps).map(
                          ([name, value]) => (
                            <div className="step-item" key={name}>
                              <div className="step-name">{name.replace(/_/g, " ")}</div>
                              <div className="step-value">
                                {Array.isArray(value)
                                  ? value.join(" → ")
                                  : String(value)}
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}

                  {/* Related Pantun */}
                  {result.related_pantun && result.related_pantun.length > 0 && (
                    <div className="card">
                      <h3 className="card-title">
                        📚 Pantun Berkaitan ({result.top_prediction.theme})
                      </h3>
                      <div className="related-pantun-list">
                        {result.related_pantun.map((rp, i) => (
                          <div className="related-pantun-item" key={i}>
                            {rp.pantun}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Empty State */}
              {!result && !error && !isLoading && (
                <div className="card">
                  <div className="empty-state">
                    <div className="empty-state-icon">📝</div>
                    <p>
                      Masukkan pantun empat baris di sebelah kiri dan tekan
                      &quot;Klasifikasi&quot; untuk melihat keputusan.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
