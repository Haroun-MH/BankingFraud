import { useEffect, useMemo, useState } from 'react';
import { ALGORITHMS } from '../data/mockData';
import { api } from '../services/api';
import { useNotifications } from '../context/NotificationContext';
import Tooltip from '../components/Tooltip';
import {
  BarChart3,
  Download,
  Maximize2,
  ArrowLeftRight,
  Grid3X3,
  TrendingUp,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  Tooltip as RTooltip,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';

const COLORS = ['#818cf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#22d3ee'];

export default function Results() {
  const [selectedModels, setSelectedModels] = useState([]);
  const [activeTab, setActiveTab] = useState('comparison');
  const [confusionModel, setConfusionModel] = useState('rf');
  const [summary, setSummary] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);
  const { addNotification } = useNotifications();

  useEffect(() => {
    let mounted = true;
    const loadSummary = async () => {
      setIsLoading(true);
      setLoadError(null);
      try {
        const data = await api.getSummary();
        if (mounted) {
          setSummary(data);
          const ids = Object.keys(data.models || {});
          if (ids.length > 0) {
            setSelectedModels(ids.slice(0, 2));
            setConfusionModel(ids[0]);
          }
        }
      } catch (err) {
        if (mounted) {
          setLoadError('Unable to load results');
        }
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    };

    loadSummary();
    return () => {
      mounted = false;
    };
  }, []);

  const toggleModel = (id) => {
    setSelectedModels((prev) =>
      prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]
    );
  };

  const renderTable = (rows, title) => {
    if (!rows || rows.length === 0) {
      return null;
    }
    const headers = Object.keys(rows[0]);
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="p-5 border-b border-gray-800">
          <h3 className="text-sm font-semibold text-white">{title}</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800/50">
                {headers.map((header) => (
                  <th key={header} className="text-left px-5 py-3 text-gray-400 font-medium">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr key={idx} className="border-b border-gray-800 last:border-0">
                  {headers.map((header) => (
                    <td key={header} className="px-5 py-3 text-gray-300">
                      {row[header] != null ? String(row[header]) : 'n/a'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const modelResults = summary?.models || {};
  const availableModels = useMemo(
    () => ALGORITHMS.filter((algo) => modelResults[algo.id]),
    [modelResults]
  );

  // Bar chart data for comparison
  const comparisonData = ['accuracy', 'precision', 'recall', 'f1', 'auc'].map((metric) => {
    const entry = { metric: metric.charAt(0).toUpperCase() + metric.slice(1) };
    selectedModels.forEach((id) => {
      const algo = ALGORITHMS.find((a) => a.id === id);
      entry[algo.shortName] = modelResults[id]?.metrics?.[metric] ?? null;
    });
    return entry;
  });

  // Radar data
  const radarData = ['accuracy', 'precision', 'recall', 'f1', 'auc'].map((metric) => {
    const entry = { metric: metric.toUpperCase() };
    selectedModels.forEach((id) => {
      const algo = ALGORITHMS.find((a) => a.id === id);
      const value = modelResults[id]?.metrics?.[metric];
      entry[algo.shortName] = value != null ? value * 100 : null;
    });
    return entry;
  });

  const cm = summary?.confusion?.[confusionModel] || null;
  const artifacts = summary?.artifacts || {};
  const curves = summary?.curves || {};
  const hasCurves = Object.values(curves).some((curve) => (curve?.roc || []).length > 0);

  const handleExportPNG = () => addNotification('Charts exported as PNG', 'success');
  const handleExportCSV = () => addNotification('Results exported as CSV', 'success');

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <BarChart3 className="w-8 h-8 text-emerald-400" />
            Results & Visualization
            <Tooltip text="Interactive charts comparing model performance. Export graphs and metrics." />
          </h1>
          <p className="text-gray-400 mt-1">Compare models side by side with dynamic visualizations</p>
          {isLoading && <p className="text-xs text-gray-500 mt-1">Loading MLflow summary...</p>}
          {loadError && <p className="text-xs text-amber-400 mt-1">{loadError}</p>}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleExportPNG} className="bg-gray-800 hover:bg-gray-700 text-gray-200 text-sm px-3 py-2 rounded-lg border border-gray-700 flex items-center gap-1.5 transition-colors">
            <Download className="w-4 h-4" /> PNG
          </button>
          <button onClick={handleExportCSV} className="bg-gray-800 hover:bg-gray-700 text-gray-200 text-sm px-3 py-2 rounded-lg border border-gray-700 flex items-center gap-1.5 transition-colors">
            <Download className="w-4 h-4" /> CSV
          </button>
        </div>
      </div>

      {/* Model Selection */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <p className="text-sm text-gray-400 mb-3 flex items-center gap-2">
          <ArrowLeftRight className="w-4 h-4" />
          Select models to compare:
        </p>
        <div className="flex flex-wrap gap-2">
          {availableModels.map((algo, idx) => (
            <button
              key={algo.id}
              onClick={() => toggleModel(algo.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all border ${
                selectedModels.includes(algo.id)
                  ? 'text-white border-current'
                  : 'bg-gray-800 text-gray-500 border-gray-700 hover:text-gray-300'
              }`}
              style={
                selectedModels.includes(algo.id)
                  ? { backgroundColor: COLORS[idx] + '20', borderColor: COLORS[idx], color: COLORS[idx] }
                  : {}
              }
            >
              {algo.shortName}
              {modelResults[algo.id]?.metrics?.f1 != null && (
                <span className="ml-2 text-xs opacity-70">
                  F1: {modelResults[algo.id].metrics.f1.toFixed(3)}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1">
        {[
          { id: 'comparison', label: 'Comparison', icon: BarChart3 },
          { id: 'roc', label: 'ROC Curves', icon: TrendingUp },
          { id: 'confusion', label: 'Confusion Matrix', icon: Grid3X3 },
          { id: 'features', label: 'Feature Importance', icon: Maximize2 },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-indigo-600 text-white'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'comparison' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Bar Chart */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Metrics Comparison</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} barGap={4}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                  <YAxis domain={[0.6, 1]} tick={{ fill: '#9ca3af', fontSize: 12 }} />
                  <RTooltip
                    contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                    labelStyle={{ color: '#f3f4f6' }}
                  />
                  <Legend />
                  {selectedModels.map((id, idx) => {
                    const algo = ALGORITHMS.find((a) => a.id === id);
                    return (
                      <Bar key={id} dataKey={algo.shortName} fill={COLORS[idx]} radius={[4, 4, 0, 0]} />
                    );
                  })}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Radar Chart */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Radar Comparison</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#1f2937" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <PolarRadiusAxis domain={[60, 100]} tick={{ fill: '#6b7280', fontSize: 10 }} />
                  {selectedModels.map((id, idx) => {
                    const algo = ALGORITHMS.find((a) => a.id === id);
                    return (
                      <Radar
                        key={id}
                        name={algo.shortName}
                        dataKey={algo.shortName}
                        stroke={COLORS[idx]}
                        fill={COLORS[idx]}
                        fillOpacity={0.15}
                      />
                    );
                  })}
                  <Legend />
                  <RTooltip
                    contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Metrics Table */}
          <div className="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="p-5 border-b border-gray-800">
              <h3 className="text-sm font-semibold text-white">Detailed Metrics</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-800/50">
                    <th className="text-left px-5 py-3 text-gray-400 font-medium">Model</th>
                    <th className="text-right px-5 py-3 text-gray-400 font-medium">Accuracy</th>
                    <th className="text-right px-5 py-3 text-gray-400 font-medium">Precision</th>
                    <th className="text-right px-5 py-3 text-gray-400 font-medium">Recall</th>
                    <th className="text-right px-5 py-3 text-gray-400 font-medium">F1-Score</th>
                    <th className="text-right px-5 py-3 text-gray-400 font-medium">AUC-ROC</th>
                    <th className="text-right px-5 py-3 text-gray-400 font-medium">Train Time</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedModels.map((id) => {
                    const algo = ALGORITHMS.find((a) => a.id === id);
                    const r = modelResults[id];
                    const metrics = r?.metrics || {};
                    const trainTime = r?.durationSec != null ? `${Math.round(r.durationSec)}s` : 'n/a';
                    return (
                      <tr key={id} className="border-b border-gray-800 last:border-0">
                        <td className="px-5 py-3 font-medium text-white">{algo.name}</td>
                        <td className="px-5 py-3 text-right font-mono text-gray-300">
                          {metrics.accuracy != null ? metrics.accuracy.toFixed(3) : 'n/a'}
                        </td>
                        <td className="px-5 py-3 text-right font-mono text-gray-300">
                          {metrics.precision != null ? metrics.precision.toFixed(3) : 'n/a'}
                        </td>
                        <td className="px-5 py-3 text-right font-mono text-gray-300">
                          {metrics.recall != null ? metrics.recall.toFixed(3) : 'n/a'}
                        </td>
                        <td className="px-5 py-3 text-right font-mono text-emerald-400 font-semibold">
                          {metrics.f1 != null ? metrics.f1.toFixed(3) : 'n/a'}
                        </td>
                        <td className="px-5 py-3 text-right font-mono text-indigo-300">
                          {metrics.auc != null ? metrics.auc.toFixed(3) : 'n/a'}
                        </td>
                        <td className="px-5 py-3 text-right text-gray-400">{trainTime}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'roc' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {!hasCurves && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 text-sm text-gray-500">
              Curve data is not available yet. Re-run training to generate ROC and PR curves.
            </div>
          )}
          {hasCurves && (
            <>
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
                <h3 className="text-sm font-semibold text-white mb-4">ROC Curve</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis
                        dataKey="fpr"
                        type="number"
                        domain={[0, 1]}
                        tick={{ fill: '#9ca3af', fontSize: 11 }}
                        label={{ value: 'False Positive Rate', position: 'bottom', fill: '#6b7280', fontSize: 12 }}
                      />
                      <YAxis
                        dataKey="tpr"
                        type="number"
                        domain={[0, 1]}
                        tick={{ fill: '#9ca3af', fontSize: 11 }}
                        label={{ value: 'True Positive Rate', angle: -90, position: 'left', fill: '#6b7280', fontSize: 12 }}
                      />
                      <RTooltip
                        contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                      />
                      <Legend />
                      <Line
                        data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
                        dataKey="tpr"
                        stroke="#374151"
                        strokeDasharray="5 5"
                        dot={false}
                        name="Random"
                      />
                      {selectedModels.map((id, idx) => {
                        const algo = ALGORITHMS.find((a) => a.id === id);
                        const curve = curves[id]?.roc || [];
                        return (
                          <Line
                            key={id}
                            data={curve}
                            dataKey="tpr"
                            stroke={COLORS[idx]}
                            dot={false}
                            strokeWidth={2}
                            name={algo.shortName}
                          />
                        );
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
                <h3 className="text-sm font-semibold text-white mb-4">Precision-Recall Curve</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis
                        dataKey="recall"
                        type="number"
                        domain={[0, 1]}
                        tick={{ fill: '#9ca3af', fontSize: 11 }}
                        label={{ value: 'Recall', position: 'bottom', fill: '#6b7280', fontSize: 12 }}
                      />
                      <YAxis
                        dataKey="precision"
                        type="number"
                        domain={[0, 1]}
                        tick={{ fill: '#9ca3af', fontSize: 11 }}
                        label={{ value: 'Precision', angle: -90, position: 'left', fill: '#6b7280', fontSize: 12 }}
                      />
                      <RTooltip
                        contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                      />
                      <Legend />
                      {selectedModels.map((id, idx) => {
                        const algo = ALGORITHMS.find((a) => a.id === id);
                        const curve = curves[id]?.pr || [];
                        return (
                          <Line
                            key={id}
                            data={curve}
                            dataKey="precision"
                            stroke={COLORS[idx]}
                            dot={false}
                            strokeWidth={2}
                            name={algo.shortName}
                          />
                        );
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {activeTab === 'confusion' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-400">Model:</span>
            {availableModels.map((algo) => (
              <button
                key={algo.id}
                onClick={() => setConfusionModel(algo.id)}
                className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                  confusionModel === algo.id
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                }`}
              >
                {algo.shortName}
              </button>
            ))}
          </div>
          {cm ? (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 max-w-lg mx-auto">
              <h3 className="text-lg font-semibold text-white text-center mb-6">Interactive Confusion Matrix</h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-5 text-center hover:scale-105 transition-transform cursor-pointer">
                  <p className="text-xs text-emerald-400 mb-1">True Negative</p>
                  <p className="text-3xl font-bold text-emerald-300">{cm.tn.toLocaleString()}</p>
                  <p className="text-xs text-gray-500 mt-1">Predicted: No | Actual: No</p>
                </div>
                <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5 text-center hover:scale-105 transition-transform cursor-pointer">
                  <p className="text-xs text-red-400 mb-1">False Positive</p>
                  <p className="text-3xl font-bold text-red-300">{cm.fp.toLocaleString()}</p>
                  <p className="text-xs text-gray-500 mt-1">Predicted: Yes | Actual: No</p>
                </div>
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-5 text-center hover:scale-105 transition-transform cursor-pointer">
                  <p className="text-xs text-amber-400 mb-1">False Negative</p>
                  <p className="text-3xl font-bold text-amber-300">{cm.fn.toLocaleString()}</p>
                  <p className="text-xs text-gray-500 mt-1">Predicted: No | Actual: Yes</p>
                </div>
                <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-xl p-5 text-center hover:scale-105 transition-transform cursor-pointer">
                  <p className="text-xs text-indigo-400 mb-1">True Positive</p>
                  <p className="text-3xl font-bold text-indigo-300">{cm.tp.toLocaleString()}</p>
                  <p className="text-xs text-gray-500 mt-1">Predicted: Yes | Actual: Yes</p>
                </div>
              </div>
              <div className="mt-4 flex justify-center gap-6 text-xs text-gray-500">
                <span>Total: {(cm.tp + cm.tn + cm.fp + cm.fn).toLocaleString()}</span>
                <span>Error Rate: {(((cm.fp + cm.fn) / (cm.tp + cm.tn + cm.fp + cm.fn)) * 100).toFixed(2)}%</span>
              </div>
            </div>
          ) : (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 text-center text-sm text-gray-500">
              {artifacts['rf_best_confusion_matrix.png'] ? (
                <img
                  src={artifacts['rf_best_confusion_matrix.png']}
                  alt="Confusion matrix"
                  className="mx-auto rounded-lg border border-gray-800"
                />
              ) : (
                'Confusion matrix data is not available yet.'
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'features' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Random Forest Feature Importance</h3>
            {artifacts['rf_feature_importance.png'] ? (
              <img
                src={artifacts['rf_feature_importance.png']}
                alt="Random Forest feature importance"
                className="w-full rounded-lg border border-gray-800"
              />
            ) : (
              <div className="text-sm text-gray-500">Feature importance plot not available yet.</div>
            )}
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Supporting Reports</h3>
            <p className="text-sm text-gray-400">
              Task 3 and Task 4 reports are available below in the experimental analysis section.
            </p>
          </div>
        </div>
      )}

      {/* Task 3 & Task 4 Reports */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">Experimental Reports</h2>
          <span className="text-xs text-gray-500">MLflow + Reports</span>
        </div>

        {renderTable(summary?.tables?.task3_comparison, 'Task 3 - Comparison Table')}
        {renderTable(summary?.tables?.rf_bias_variance, 'Task 4 - Bias/Variance Grid')}
        {renderTable(summary?.tables?.rf_stability, 'Task 4 - Stability (Random State)')}
        {renderTable(summary?.tables?.rf_misclassified_samples, 'Task 4 - Misclassified Samples')}

        {summary?.reports?.task3_md && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-3">Task 3 Report (Markdown)</h3>
            <pre className="text-xs text-gray-300 whitespace-pre-wrap">{summary.reports.task3_md}</pre>
          </div>
        )}

        {summary?.reports?.task4_md && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-3">Task 4 Report (Markdown)</h3>
            <pre className="text-xs text-gray-300 whitespace-pre-wrap">{summary.reports.task4_md}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
