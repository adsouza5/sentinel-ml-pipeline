import { useState, useEffect, useRef, useCallback } from "react";

const COLORS = {
  bg: "#0a0a0f",
  surface: "#12121a",
  surfaceHover: "#1a1a25",
  border: "#1e1e2e",
  borderActive: "#3b82f6",
  text: "#e2e8f0",
  textMuted: "#64748b",
  textDim: "#475569",
  accent: "#3b82f6",
  accentGlow: "rgba(59, 130, 246, 0.15)",
  green: "#22c55e",
  greenGlow: "rgba(34, 197, 94, 0.2)",
  amber: "#f59e0b",
  amberGlow: "rgba(245, 158, 11, 0.15)",
  red: "#ef4444",
  purple: "#a855f7",
  purpleGlow: "rgba(168, 85, 247, 0.15)",
  cyan: "#06b6d4",
  cyanGlow: "rgba(6, 182, 212, 0.15)",
};

const FONTS = {
  mono: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
  sans: "'DM Sans', 'Segoe UI', system-ui, sans-serif",
  display: "'Space Grotesk', 'DM Sans', system-ui, sans-serif",
};

// Pipeline stages
const STAGES = [
  { id: "ingest", label: "Pub/Sub Ingest", icon: "⟐", color: COLORS.cyan, glow: COLORS.cyanGlow },
  { id: "preprocess", label: "Cloud Run Preprocessor", icon: "⚙", color: COLORS.amber, glow: COLORS.amberGlow },
  { id: "inference", label: "Vertex AI Inference", icon: "◈", color: COLORS.purple, glow: COLORS.purpleGlow },
  { id: "store", label: "BigQuery Store", icon: "⬡", color: COLORS.green, glow: COLORS.greenGlow },
  { id: "serve", label: "API Response", icon: "↗", color: COLORS.accent, glow: COLORS.accentGlow },
];

// Simulated data samples
const SAMPLE_DATA = [
  { input: '{"ticker": "AAPL", "price": 198.5, "volume": 12400}', prediction: "BULLISH", confidence: 0.87, latency: 23 },
  { input: '{"ticker": "TSLA", "price": 245.2, "volume": 38200}', prediction: "BEARISH", confidence: 0.72, latency: 31 },
  { input: '{"ticker": "MSFT", "price": 412.8, "volume": 8900}', prediction: "NEUTRAL", confidence: 0.91, latency: 18 },
  { input: '{"ticker": "NVDA", "price": 875.3, "volume": 54100}', prediction: "BULLISH", confidence: 0.94, latency: 27 },
  { input: '{"ticker": "AMZN", "price": 186.4, "volume": 15700}', prediction: "BEARISH", confidence: 0.68, latency: 35 },
  { input: '{"ticker": "META", "price": 502.1, "volume": 22300}', prediction: "BULLISH", confidence: 0.82, latency: 21 },
];

function MetricCard({ label, value, unit, trend, color }) {
  return (
    <div style={{
      background: COLORS.surface,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      padding: "12px 16px",
      minWidth: 120,
      flex: 1,
    }}>
      <div style={{ fontFamily: FONTS.sans, fontSize: 11, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <span style={{ fontFamily: FONTS.mono, fontSize: 22, fontWeight: 700, color: color || COLORS.text }}>
          {value}
        </span>
        {unit && <span style={{ fontFamily: FONTS.mono, fontSize: 12, color: COLORS.textMuted }}>{unit}</span>}
      </div>
      {trend && (
        <div style={{ fontFamily: FONTS.mono, fontSize: 11, color: trend > 0 ? COLORS.green : COLORS.red, marginTop: 2 }}>
          {trend > 0 ? "▲" : "▼"} {Math.abs(trend)}%
        </div>
      )}
    </div>
  );
}

function PipelineNode({ stage, isActive, isComplete, activeProgress, onClick }) {
  const nodeStyle = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 8,
    cursor: "pointer",
    transition: "all 0.3s ease",
    position: "relative",
  };

  const iconContainerStyle = {
    width: 56,
    height: 56,
    borderRadius: 12,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 22,
    background: isActive || isComplete ? stage.glow : COLORS.surface,
    border: `2px solid ${isActive ? stage.color : isComplete ? stage.color + "80" : COLORS.border}`,
    boxShadow: isActive ? `0 0 20px ${stage.glow}, 0 0 40px ${stage.glow}` : "none",
    transition: "all 0.4s ease",
    position: "relative",
    overflow: "hidden",
  };

  return (
    <div style={nodeStyle} onClick={onClick}>
      <div style={iconContainerStyle}>
        {isActive && (
          <div style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            height: `${(activeProgress || 0) * 100}%`,
            background: `linear-gradient(to top, ${stage.color}30, transparent)`,
            transition: "height 0.3s ease",
          }} />
        )}
        <span style={{ position: "relative", zIndex: 1 }}>{stage.icon}</span>
      </div>
      <span style={{
        fontFamily: FONTS.mono,
        fontSize: 10,
        color: isActive ? stage.color : isComplete ? COLORS.text : COLORS.textDim,
        textAlign: "center",
        maxWidth: 90,
        lineHeight: 1.3,
        transition: "color 0.3s ease",
      }}>
        {stage.label}
      </span>
      {isActive && (
        <div style={{
          position: "absolute",
          top: -6,
          right: -6,
          width: 12,
          height: 12,
          borderRadius: "50%",
          background: stage.color,
          animation: "pulse 1.5s ease-in-out infinite",
        }} />
      )}
    </div>
  );
}

function PipelineConnector({ isActive, isComplete, color }) {
  return (
    <div style={{
      flex: 1,
      height: 2,
      background: isComplete ? `${color}60` : COLORS.border,
      position: "relative",
      minWidth: 20,
      maxWidth: 60,
      alignSelf: "center",
      marginBottom: 20,
      borderRadius: 1,
      overflow: "hidden",
    }}>
      {isActive && (
        <div style={{
          position: "absolute",
          top: 0,
          left: 0,
          height: "100%",
          width: "30%",
          background: `linear-gradient(90deg, transparent, ${color}, transparent)`,
          animation: "flowRight 1s ease-in-out infinite",
          borderRadius: 1,
        }} />
      )}
    </div>
  );
}

function LogEntry({ message, type, timestamp }) {
  const typeColors = {
    info: COLORS.accent,
    success: COLORS.green,
    warning: COLORS.amber,
    processing: COLORS.purple,
    data: COLORS.cyan,
  };

  return (
    <div style={{
      fontFamily: FONTS.mono,
      fontSize: 11,
      lineHeight: 1.6,
      display: "flex",
      gap: 8,
      opacity: 0.9,
    }}>
      <span style={{ color: COLORS.textDim, flexShrink: 0 }}>{timestamp}</span>
      <span style={{ color: typeColors[type] || COLORS.textMuted, flexShrink: 0 }}>
        {type === "success" ? "✓" : type === "warning" ? "!" : type === "processing" ? "◌" : type === "data" ? "›" : "·"}
      </span>
      <span style={{ color: COLORS.textMuted }}>{message}</span>
    </div>
  );
}

function LatencyChart({ history }) {
  const maxVal = Math.max(...history.map(h => h.latency), 50);
  const chartH = 80;
  const barW = Math.max(4, Math.min(12, 200 / history.length));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ fontFamily: FONTS.sans, fontSize: 11, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Inference Latency (ms)
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: chartH, padding: "0 4px" }}>
        {history.map((h, i) => {
          const height = (h.latency / maxVal) * chartH;
          const color = h.latency < 25 ? COLORS.green : h.latency < 35 ? COLORS.amber : COLORS.red;
          return (
            <div
              key={i}
              title={`${h.latency}ms — ${h.ticker}`}
              style={{
                width: barW,
                height,
                background: `linear-gradient(to top, ${color}, ${color}80)`,
                borderRadius: "2px 2px 0 0",
                transition: "height 0.4s ease",
                opacity: i === history.length - 1 ? 1 : 0.6,
              }}
            />
          );
        })}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span style={{ fontFamily: FONTS.mono, fontSize: 9, color: COLORS.textDim }}>0ms</span>
        <span style={{ fontFamily: FONTS.mono, fontSize: 9, color: COLORS.textDim }}>{maxVal}ms</span>
      </div>
    </div>
  );
}

function TechBadge({ label }) {
  return (
    <span style={{
      fontFamily: FONTS.mono,
      fontSize: 10,
      color: COLORS.textMuted,
      background: COLORS.surface,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 4,
      padding: "3px 8px",
      whiteSpace: "nowrap",
    }}>
      {label}
    </span>
  );
}

export default function MLPipelineShowcase() {
  const [view, setView] = useState("architecture");
  const [isRunning, setIsRunning] = useState(false);
  const [activeStage, setActiveStage] = useState(-1);
  const [stageProgress, setStageProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [processedCount, setProcessedCount] = useState(0);
  const [latencyHistory, setLatencyHistory] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [totalLatency, setTotalLatency] = useState(0);
  const [avgConfidence, setAvgConfidence] = useState(0);
  const logRef = useRef(null);
  const runningRef = useRef(false);
  const sampleIdxRef = useRef(0);

  const addLog = useCallback((message, type = "info") => {
    const now = new Date();
    const ts = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}.${String(now.getMilliseconds()).padStart(3, "0")}`;
    setLogs(prev => [...prev.slice(-40), { message, type, timestamp: ts }]);
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  const runPipeline = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setIsRunning(true);
    setLogs([]);
    setActiveStage(-1);
    setProcessedCount(0);
    setLatencyHistory([]);
    setCurrentPrediction(null);
    setTotalLatency(0);
    setAvgConfidence(0);
    sampleIdxRef.current = 0;

    addLog("Initializing ML Inference Pipeline...", "info");
    await sleep(600);
    addLog("Connected to Pub/Sub topic: market-data-stream", "success");
    await sleep(400);
    addLog("Vertex AI endpoint healthy — model: sentiment-classifier-v3", "success");
    await sleep(400);
    addLog("BigQuery dataset: predictions.market_signals ready", "success");
    await sleep(500);
    addLog("Pipeline active — awaiting incoming data...", "info");
    await sleep(700);

    let totalLat = 0;
    let totalConf = 0;

    for (let i = 0; i < SAMPLE_DATA.length; i++) {
      if (!runningRef.current) break;
      const sample = SAMPLE_DATA[i];
      const parsed = JSON.parse(sample.input);

      // Stage 0: Ingest
      setActiveStage(0);
      addLog(`Pub/Sub message received — ${parsed.ticker} @ $${parsed.price}`, "data");
      for (let p = 0; p <= 10; p++) { setStageProgress(p / 10); await sleep(60); }
      await sleep(200);

      // Stage 1: Preprocess
      setActiveStage(1);
      addLog(`Preprocessing: normalizing price/volume for ${parsed.ticker}`, "processing");
      for (let p = 0; p <= 10; p++) { setStageProgress(p / 10); await sleep(50); }
      addLog("Feature vector: [0.72, -0.15, 0.88, 0.33, -0.41]", "data");
      await sleep(300);

      // Stage 2: Inference
      setActiveStage(2);
      addLog(`Running inference on Vertex AI endpoint...`, "processing");
      for (let p = 0; p <= 10; p++) { setStageProgress(p / 10); await sleep(70); }
      const pred = sample.prediction;
      const conf = sample.confidence;
      addLog(`Prediction: ${pred} (confidence: ${(conf * 100).toFixed(1)}%) — ${sample.latency}ms`, "success");
      setCurrentPrediction({ ticker: parsed.ticker, prediction: pred, confidence: conf, latency: sample.latency });
      await sleep(300);

      // Stage 3: Store
      setActiveStage(3);
      addLog(`Writing to BigQuery: predictions.market_signals`, "processing");
      for (let p = 0; p <= 10; p++) { setStageProgress(p / 10); await sleep(40); }
      addLog(`Row inserted — partition: ${new Date().toISOString().split("T")[0]}`, "success");
      await sleep(200);

      // Stage 4: Serve
      setActiveStage(4);
      addLog(`API response served — 200 OK`, "success");
      for (let p = 0; p <= 10; p++) { setStageProgress(p / 10); await sleep(30); }
      await sleep(200);

      totalLat += sample.latency;
      totalConf += conf;
      const count = i + 1;
      setProcessedCount(count);
      setTotalLatency(Math.round(totalLat / count));
      setAvgConfidence(Math.round((totalConf / count) * 100));
      setLatencyHistory(prev => [...prev, { latency: sample.latency, ticker: parsed.ticker }]);

      if (i < SAMPLE_DATA.length - 1) {
        setActiveStage(-1);
        addLog("Awaiting next message...", "info");
        await sleep(800);
      }
    }

    setActiveStage(-1);
    addLog(`Pipeline batch complete — ${SAMPLE_DATA.length} predictions processed`, "success");
    setIsRunning(false);
    runningRef.current = false;
  }, [addLog]);

  const stopPipeline = useCallback(() => {
    runningRef.current = false;
    setIsRunning(false);
    setActiveStage(-1);
    addLog("Pipeline stopped by user", "warning");
  }, [addLog]);

  const predColor = currentPrediction?.prediction === "BULLISH" ? COLORS.green : currentPrediction?.prediction === "BEARISH" ? COLORS.red : COLORS.amber;

  return (
    <div style={{
      background: COLORS.bg,
      minHeight: "100vh",
      color: COLORS.text,
      fontFamily: FONTS.sans,
      padding: "24px 16px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.3); }
        }
        @keyframes flowRight {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 2px; }
      `}</style>

      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <span style={{ fontFamily: FONTS.mono, fontSize: 12, color: COLORS.accent, background: COLORS.accentGlow, padding: "2px 8px", borderRadius: 4, border: `1px solid ${COLORS.accent}30` }}>
              PROJECT
            </span>
            <span style={{ fontFamily: FONTS.mono, fontSize: 11, color: COLORS.textDim }}>GCP · Vertex AI · Cloud Run</span>
          </div>
          <h1 style={{
            fontFamily: FONTS.display,
            fontSize: 28,
            fontWeight: 700,
            color: COLORS.text,
            lineHeight: 1.2,
            marginBottom: 8,
          }}>
            Real-Time ML Inference Pipeline
          </h1>
          <p style={{
            fontFamily: FONTS.sans,
            fontSize: 14,
            color: COLORS.textMuted,
            lineHeight: 1.6,
            maxWidth: 600,
          }}>
            Streaming market data through a serverless ML pipeline — ingestion via Pub/Sub,
            preprocessing on Cloud Run, inference on Vertex AI, storage in BigQuery.
          </p>
          <div style={{ display: "flex", gap: 6, marginTop: 12, flexWrap: "wrap" }}>
            {["Vertex AI", "Cloud Run", "Pub/Sub", "BigQuery", "Terraform", "Python", "FastAPI"].map(t => (
              <TechBadge key={t} label={t} />
            ))}
          </div>
        </div>

        {/* Tab Switcher */}
        <div style={{
          display: "flex",
          gap: 2,
          marginBottom: 24,
          background: COLORS.surface,
          borderRadius: 8,
          padding: 3,
          border: `1px solid ${COLORS.border}`,
          width: "fit-content",
        }}>
          {[
            { id: "architecture", label: "Architecture" },
            { id: "simulation", label: "Live Simulation" },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setView(tab.id)}
              style={{
                fontFamily: FONTS.mono,
                fontSize: 12,
                padding: "8px 20px",
                borderRadius: 6,
                border: "none",
                cursor: "pointer",
                background: view === tab.id ? COLORS.accent : "transparent",
                color: view === tab.id ? "#fff" : COLORS.textMuted,
                transition: "all 0.2s ease",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Architecture View */}
        {view === "architecture" && (
          <div style={{ animation: "fadeIn 0.4s ease" }}>
            {/* Pipeline Diagram */}
            <div style={{
              background: COLORS.surface,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 12,
              padding: 32,
              marginBottom: 20,
            }}>
              <div style={{ fontFamily: FONTS.mono, fontSize: 11, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 24 }}>
                Pipeline Architecture
              </div>
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "center", gap: 0, flexWrap: "wrap" }}>
                {STAGES.map((stage, i) => (
                  <div key={stage.id} style={{ display: "flex", alignItems: "flex-start" }}>
                    <PipelineNode stage={stage} isActive={false} isComplete={false} />
                    {i < STAGES.length - 1 && (
                      <PipelineConnector isActive={false} isComplete={false} color={COLORS.border} />
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Architecture Details */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {[
                {
                  title: "Data Ingestion",
                  desc: "Pub/Sub topic receives streaming market data events. Cloud Functions trigger on new messages, batching for throughput optimization.",
                  color: COLORS.cyan,
                  details: ["Message ordering guarantees", "Dead-letter queue for failures", "Auto-scaling subscribers"],
                },
                {
                  title: "Preprocessing",
                  desc: "Cloud Run service normalizes raw data into feature vectors. Stateless containers scale to zero when idle.",
                  color: COLORS.amber,
                  details: ["Feature normalization pipeline", "Schema validation", "Scale-to-zero cost optimization"],
                },
                {
                  title: "Model Inference",
                  desc: "Vertex AI endpoint hosts the sentiment classifier. Custom prediction routines handle pre/post processing at the model layer.",
                  color: COLORS.purple,
                  details: ["Custom container serving", "A/B model deployment", "Auto-scaling on GPU"],
                },
                {
                  title: "Storage & Serving",
                  desc: "Predictions land in BigQuery partitioned by date. FastAPI serves real-time results via Cloud Run with Redis caching.",
                  color: COLORS.green,
                  details: ["Partitioned tables by date", "Sub-second API responses", "Grafana monitoring dashboard"],
                },
              ].map((section) => (
                <div key={section.title} style={{
                  background: COLORS.surface,
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: 10,
                  padding: 20,
                  borderTop: `2px solid ${section.color}`,
                }}>
                  <div style={{ fontFamily: FONTS.display, fontSize: 15, fontWeight: 600, color: COLORS.text, marginBottom: 8 }}>
                    {section.title}
                  </div>
                  <div style={{ fontFamily: FONTS.sans, fontSize: 13, color: COLORS.textMuted, lineHeight: 1.6, marginBottom: 12 }}>
                    {section.desc}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {section.details.map(d => (
                      <div key={d} style={{ fontFamily: FONTS.mono, fontSize: 11, color: COLORS.textDim, display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ color: section.color, fontSize: 8 }}>●</span> {d}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* IaC Note */}
            <div style={{
              background: COLORS.surface,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 10,
              padding: 16,
              marginTop: 16,
              display: "flex",
              alignItems: "center",
              gap: 12,
            }}>
              <span style={{ fontSize: 20 }}>⎔</span>
              <div>
                <div style={{ fontFamily: FONTS.mono, fontSize: 12, color: COLORS.text, marginBottom: 2 }}>
                  Infrastructure as Code
                </div>
                <div style={{ fontFamily: FONTS.sans, fontSize: 12, color: COLORS.textMuted }}>
                  Entire pipeline provisioned via Terraform modules — Pub/Sub topics, Cloud Run services, Vertex AI endpoints, BigQuery datasets, IAM bindings, and monitoring alerts.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Simulation View */}
        {view === "simulation" && (
          <div style={{ animation: "fadeIn 0.4s ease" }}>
            {/* Pipeline Progress */}
            <div style={{
              background: COLORS.surface,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 12,
              padding: 24,
              marginBottom: 16,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                <div style={{ fontFamily: FONTS.mono, fontSize: 11, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  Pipeline Status
                </div>
                <button
                  onClick={isRunning ? stopPipeline : runPipeline}
                  style={{
                    fontFamily: FONTS.mono,
                    fontSize: 12,
                    padding: "8px 20px",
                    borderRadius: 6,
                    border: "none",
                    cursor: "pointer",
                    background: isRunning ? COLORS.red : COLORS.accent,
                    color: "#fff",
                    transition: "all 0.2s ease",
                  }}
                >
                  {isRunning ? "■ Stop" : "▶ Run Pipeline"}
                </button>
              </div>

              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "center", gap: 0, flexWrap: "wrap" }}>
                {STAGES.map((stage, i) => (
                  <div key={stage.id} style={{ display: "flex", alignItems: "flex-start" }}>
                    <PipelineNode
                      stage={stage}
                      isActive={activeStage === i}
                      isComplete={activeStage > i}
                      activeProgress={activeStage === i ? stageProgress : 0}
                    />
                    {i < STAGES.length - 1 && (
                      <PipelineConnector
                        isActive={activeStage === i}
                        isComplete={activeStage > i}
                        color={stage.color}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Metrics Row */}
            <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
              <MetricCard label="Processed" value={processedCount} unit={`/ ${SAMPLE_DATA.length}`} color={COLORS.accent} />
              <MetricCard label="Avg Latency" value={totalLatency || "—"} unit="ms" color={totalLatency < 25 ? COLORS.green : COLORS.amber} />
              <MetricCard label="Avg Confidence" value={avgConfidence ? `${avgConfidence}` : "—"} unit="%" color={COLORS.purple} />
              <MetricCard label="Uptime" value="99.9" unit="%" color={COLORS.green} trend={0.2} />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {/* Live Prediction */}
              <div style={{
                background: COLORS.surface,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 10,
                padding: 20,
              }}>
                <div style={{ fontFamily: FONTS.mono, fontSize: 11, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 16 }}>
                  Latest Prediction
                </div>
                {currentPrediction ? (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                      <span style={{ fontFamily: FONTS.display, fontSize: 24, fontWeight: 700, color: COLORS.text }}>
                        {currentPrediction.ticker}
                      </span>
                      <span style={{
                        fontFamily: FONTS.mono,
                        fontSize: 12,
                        fontWeight: 600,
                        color: predColor,
                        background: predColor + "15",
                        padding: "4px 12px",
                        borderRadius: 4,
                        border: `1px solid ${predColor}30`,
                      }}>
                        {currentPrediction.prediction}
                      </span>
                    </div>
                    <div style={{ display: "flex", gap: 16 }}>
                      <div>
                        <div style={{ fontFamily: FONTS.mono, fontSize: 10, color: COLORS.textDim, marginBottom: 2 }}>Confidence</div>
                        <div style={{ fontFamily: FONTS.mono, fontSize: 16, fontWeight: 600, color: COLORS.purple }}>
                          {(currentPrediction.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <div style={{ fontFamily: FONTS.mono, fontSize: 10, color: COLORS.textDim, marginBottom: 2 }}>Latency</div>
                        <div style={{ fontFamily: FONTS.mono, fontSize: 16, fontWeight: 600, color: currentPrediction.latency < 25 ? COLORS.green : COLORS.amber }}>
                          {currentPrediction.latency}ms
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ fontFamily: FONTS.mono, fontSize: 13, color: COLORS.textDim, padding: "20px 0", textAlign: "center" }}>
                    Run pipeline to see predictions
                  </div>
                )}

                {latencyHistory.length > 0 && (
                  <div style={{ marginTop: 20 }}>
                    <LatencyChart history={latencyHistory} />
                  </div>
                )}
              </div>

              {/* Log Stream */}
              <div style={{
                background: COLORS.surface,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 10,
                padding: 20,
                display: "flex",
                flexDirection: "column",
              }}>
                <div style={{ fontFamily: FONTS.mono, fontSize: 11, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }}>
                  Pipeline Logs
                </div>
                <div
                  ref={logRef}
                  style={{
                    flex: 1,
                    minHeight: 220,
                    maxHeight: 300,
                    overflowY: "auto",
                    background: COLORS.bg,
                    borderRadius: 6,
                    padding: 12,
                    border: `1px solid ${COLORS.border}`,
                  }}
                >
                  {logs.length === 0 ? (
                    <div style={{ fontFamily: FONTS.mono, fontSize: 12, color: COLORS.textDim, padding: "40px 0", textAlign: "center" }}>
                      Waiting for pipeline execution...
                    </div>
                  ) : (
                    logs.map((log, i) => (
                      <LogEntry key={i} message={log.message} type={log.type} timestamp={log.timestamp} />
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
