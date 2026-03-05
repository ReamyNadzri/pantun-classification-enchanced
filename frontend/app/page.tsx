"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";

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

// Hardcoded for UI simplicity as requested
const AVAILABLE_MODELS = [
  { id: "textcnn", name: "TextCNN (Deep Learning)" },
  { id: "svm_90-10_no_pembayang", name: "SVM 90/10 (Machine Learning)" },
  { id: "malaybert", name: "MalayBERT (Transformer)" },
];

const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

export default function Home() {
  // Input state
  const [inputMode, setInputMode] = useState<"4-lines" | "1-line">("4-lines");
  const [lines, setLines] = useState(["", "", "", ""]);
  const [singleLineInput, setSingleLineInput] = useState("");

  // Settings
  const [selectedModels, setSelectedModels] = useState<string[]>(["textcnn", "svm_90-10_no_pembayang", "malaybert"]);
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Advanced Settings
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [topK, setTopK] = useState(3);
  const [showSteps, setShowSteps] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(20);
  const [relatedCount, setRelatedCount] = useState(0);

  // Results
  // Keyed by model ID
  const [results, setResults] = useState<Record<string, ClassifyResponse>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // Theme toggle
  useEffect(() => {
    document.documentElement.setAttribute(
      "data-theme",
      isDarkMode ? "dark" : "light"
    );
  }, [isDarkMode]);

  const handleLineChange = (index: number, value: string) => {
    const newLines = [...lines];
    newLines[index] = value;
    setLines(newLines);
  };

  const toggleModelSelection = (modelId: string) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const getPantunText = () => {
    if (inputMode === "4-lines") {
      return lines.join("; ");
    }
    return singleLineInput;
  };

  const isValid = inputMode === "4-lines"
    ? lines.every((line) => line.trim().length > 0)
    : singleLineInput.trim().length > 0;

  const handleClassify = useCallback(async () => {
    if (!isValid || selectedModels.length === 0) return;

    setIsLoading(true);
    setError("");
    setResults({});

    const pantunText = getPantunText();

    try {
      const fetchPromises = selectedModels.map(async (modelId) => {
        const response = await fetch(`${API_URL}/api/classify`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pantun: pantunText,
            model: modelId,
            use_pembayang: false,
            top_k: topK,
            show_steps: showSteps,
            related_count: relatedCount,
            confidence_threshold: confidenceThreshold,
          }),
        });

        if (!response.ok) {
          throw new Error(`API error from ${modelId}: ${response.status}`);
        }

        const data: ClassifyResponse = await response.json();
        return { modelId, data };
      });

      const settledResults = await Promise.allSettled(fetchPromises);

      const newResults: Record<string, ClassifyResponse> = {};
      settledResults.forEach((result) => {
        if (result.status === "fulfilled") {
          newResults[result.value.modelId] = result.value.data;
        } else {
          console.error(result.reason);
          setError("Terdapat ralat pada salah satu model. Sila cuba lagi.");
        }
      });

      setResults(newResults);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Ralat berlaku. Pastikan backend sedang berjalan."
      );
    } finally {
      setIsLoading(false);
    }
  }, [lines, singleLineInput, inputMode, isValid, selectedModels, topK, showSteps, relatedCount, confidenceThreshold]);

  const handleClear = () => {
    setLines(["", "", "", ""]);
    setSingleLineInput("");
    setResults({});
    setError("");
  };

  return (
    <>
      <header className="header">
        <div className="container header-content">
          <Link href="/" style={{ textDecoration: 'none' }}>
            <div className="logo cursor-pointer">
              <span className="logo-icon">🪻</span>
              <div>
                <h1>Klasifikasi Pantun</h1>
                <span className="logo-subtitle">Mengenal Pasti Tema Pantun</span>
              </div>
            </div>
          </Link>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <Link href="/" className="nav-link active">Klasifikasi</Link>
            <Link href="/insights" className="nav-link">Insights</Link>
            <button
              className="theme-toggle-btn"
              onClick={() => setIsDarkMode(!isDarkMode)}
              title={isDarkMode ? "Light Mode" : "Dark Mode"}
            >
              {isDarkMode ? "☀️" : "🌙"}
            </button>
          </div>
        </div>
      </header>

      <main>
        <div className="container">
          <section className="hero" style={{ paddingBottom: '1rem' }}>
            <h2>Klasifikasi Berbilang Model AI</h2>
            <p>
              Pilih model di bawah. Bandingkan bagaimana Deep Learning, Machine Learning, dan Transformer berfikir.
            </p>
          </section>

          <div style={{ maxWidth: '1000px', margin: '0 auto' }}>

            {/* Model Selection Containers */}
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap', justifyContent: 'center' }}>
              {AVAILABLE_MODELS.map(model => {
                const isSelected = selectedModels.includes(model.id);
                return (
                  <div
                    key={model.id}
                    onClick={() => toggleModelSelection(model.id)}
                    style={{
                      flex: '1 1 250px',
                      padding: '1rem',
                      borderRadius: 'var(--radius-md)',
                      background: isSelected ? 'var(--bg-tertiary)' : 'var(--bg-card)',
                      border: `2px solid ${isSelected ? 'var(--primary)' : 'var(--border)'}`,
                      cursor: 'pointer',
                      transition: 'var(--transition)',
                      textAlign: 'center',
                      boxShadow: isSelected ? '0 0 15px var(--primary-glow)' : 'var(--shadow-sm)'
                    }}
                  >
                    <div style={{
                      width: '24px', height: '24px', borderRadius: '50%',
                      background: isSelected ? 'var(--primary)' : 'transparent',
                      border: `2px solid ${isSelected ? 'var(--primary)' : 'var(--text-muted)'}`,
                      margin: '0 auto 0.5rem',
                      display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}>
                      {isSelected && <span style={{ color: 'white', fontSize: '12px' }}>✓</span>}
                    </div>
                    <span style={{ fontWeight: 600, color: isSelected ? 'var(--primary-light)' : 'var(--text-primary)' }}>
                      {model.name}
                    </span>
                  </div>
                )
              })}
            </div>

            {/* Input Section */}
            <div className="card" style={{ marginBottom: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
                <h3 className="card-title" style={{ marginBottom: 0 }}>✍️ Masukkan Pantun</h3>

                {/* Mode Toggle Tabs */}
                <div style={{ display: 'flex', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)', padding: '0.25rem' }}>
                  <button
                    onClick={() => setInputMode('4-lines')}
                    style={{
                      padding: '0.4rem 1rem', border: 'none', borderRadius: 'var(--radius-sm)',
                      background: inputMode === '4-lines' ? 'var(--primary)' : 'transparent',
                      color: inputMode === '4-lines' ? 'white' : 'var(--text-secondary)',
                      cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem', transition: '0.2s'
                    }}
                  >4 Baris</button>
                  <button
                    onClick={() => setInputMode('1-line')}
                    style={{
                      padding: '0.4rem 1rem', border: 'none', borderRadius: 'var(--radius-sm)',
                      background: inputMode === '1-line' ? 'var(--primary)' : 'transparent',
                      color: inputMode === '1-line' ? 'white' : 'var(--text-secondary)',
                      cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem', transition: '0.2s'
                    }}
                  >Teks Penuh</button>
                </div>
              </div>

              {inputMode === "4-lines" ? (
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
                          index === 0 ? "Baris pertama (pembayang)..." :
                            index === 1 ? "Baris kedua (pembayang)..." :
                              index === 2 ? "Baris ketiga (maksud)..." :
                                "Baris keempat (maksud)..."
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
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <textarea
                    className="line-input"
                    style={{ minHeight: '150px', resize: 'vertical' }}
                    placeholder={`Contoh:\nPulau pandan jauh ke tengah,\nGunung Daik bercabang tiga;\nHancur badan dikandung tanah,\nBudi yang baik dikenang juga.`}
                    value={singleLineInput}
                    onChange={(e) => setSingleLineInput(e.target.value)}
                  />
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ color: 'var(--accent)' }}>💡 Peringatan:</span>
                    Pastikan anda memisahkan hujung ayat menggunakan koma ( , ) atau semikolon ( ; ) atau enter ( baris baru ).
                  </div>
                </div>
              )}

              {/* Buttons */}
              <div className="btn-group" style={{ flexDirection: 'row', marginTop: '1.5rem' }}>
                <button
                  className="btn btn-primary"
                  onClick={handleClassify}
                  disabled={!isValid || selectedModels.length === 0 || isLoading}
                  style={{ flex: 1 }}
                >
                  {isLoading ? (
                    <><span className="spinner" /> Sedang Menilai...</>
                  ) : (
                    <>🔍 Nilai Menggunakan {selectedModels.length} Model</>
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
              <div className="card" style={{ marginBottom: "2rem" }}>
                <h3 className="card-title">⚙️ Tetapan Klasifikasi Tambahan</h3>
                <div className="settings-grid">
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
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div
                className="card"
                style={{
                  borderColor: "rgba(239, 68, 68, 0.3)",
                  marginBottom: "1rem",
                }}
              >
                <h3 className="card-title" style={{ color: "#ef4444" }}>❌ Ralat</h3>
                <p style={{ color: "var(--text-secondary)" }}>{error}</p>
              </div>
            )}

            {/* Multi-Model Results Output */}
            {Object.keys(results).length > 0 && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: `repeat(auto-fit, minmax(280px, 1fr))`,
                gap: '1rem'
              }}>
                {AVAILABLE_MODELS.map(model => {
                  const modelResult = results[model.id];
                  if (!modelResult) return null;

                  return (
                    <div className="card" key={model.id}>
                      <div style={{
                        fontSize: '0.85rem',
                        fontWeight: 600,
                        color: 'var(--text-muted)',
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px',
                        marginBottom: '1rem',
                        borderBottom: '1px solid var(--border)',
                        paddingBottom: '0.5rem'
                      }}>
                        {model.name}
                      </div>

                      <div style={{ textAlign: "center", margin: "1rem 0" }}>
                        <span
                          className={`theme-badge ${modelResult.top_prediction.is_uncertain ? "uncertain" : ""
                            }`}
                        >
                          {modelResult.top_prediction.is_uncertain
                            ? "⚠️ Tidak Pasti"
                            : `🏷️ ${modelResult.top_prediction.theme}`}
                        </span>
                      </div>

                      <div className="confidence-bar-container">
                        <div className="confidence-text">
                          <span>Keyakinan</span>
                          <span style={{ fontWeight: 600 }}>
                            {modelResult.top_prediction.confidence.toFixed(1)}%
                          </span>
                        </div>
                        <div className="confidence-bar">
                          <div
                            className="confidence-bar-fill"
                            style={{
                              width: `${modelResult.top_prediction.confidence}%`,
                            }}
                          />
                        </div>
                      </div>

                      {/* Other Predictions */}
                      <div style={{ marginTop: '1.5rem' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Ramalan Lain:</div>
                        {modelResult.predictions.slice(1).map((pred, i) => (
                          <div key={i} style={{
                            display: 'flex', justifyContent: 'space-between',
                            fontSize: '0.85rem', padding: '0.25rem 0',
                            color: 'var(--text-secondary)'
                          }}>
                            <span>{pred.theme}</span>
                            <span style={{ fontWeight: 600 }}>{pred.confidence.toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>

                      {/* Preprocessing Steps */}
                      {showSteps && modelResult.preprocessing_steps && (
                        <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
                          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>🔍 Langkah Pre-pemprosesan</span>
                          <div style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                            {Object.entries(modelResult.preprocessing_steps).map(([step, val]) => (
                              <div key={step} style={{ marginBottom: '0.25rem' }}>
                                <strong style={{ color: 'var(--primary-light)' }}>{step}:</strong> {Array.isArray(val) ? val.join(" → ") : String(val)}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Related Pantun */}
                      {relatedCount > 0 && modelResult.related_pantun && modelResult.related_pantun.length > 0 && (
                        <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
                          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>📚 Pantun Berkaitan ({modelResult.top_prediction.theme})</span>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                            {modelResult.related_pantun.map((rp, i) => (
                              <div key={i} style={{
                                background: 'var(--bg-tertiary)', padding: '0.75rem',
                                borderRadius: 'var(--radius-sm)', fontSize: '0.85rem',
                                color: 'var(--text-secondary)', borderLeft: '2px solid var(--accent)',
                                whiteSpace: 'pre-wrap', fontStyle: 'italic'
                              }}>
                                {rp.pantun.split("; ").join("\n")}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                    </div>
                  );
                })}
              </div>
            )}

          </div>
        </div>
      </main>
    </>
  );
}
