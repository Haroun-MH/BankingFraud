import { useEffect, useMemo, useState } from 'react';
import { api } from '../services/api';
import { useNotifications } from '../context/NotificationContext';
import Tooltip from '../components/Tooltip';
import {
  GitBranch,
  RotateCcw,
  Download,
  CheckCircle,
  Clock,
  ArrowLeftRight,
  Tag,
  Database,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  RefreshCw,
  Play,
} from 'lucide-react';

export default function MLOps() {
  const [experiments, setExperiments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [expandedExp, setExpandedExp] = useState(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareSelection, setCompareSelection] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  const { addNotification } = useNotifications();

  const loadRuns = async (showToast = false) => {
    setIsLoading(true);
    setLoadError(null);
    try {
      const runs = await api.getRuns(50);
      setExperiments(Array.isArray(runs) ? runs : []);
      if (showToast) {
        addNotification('Experiments refreshed', 'success');
      }
    } catch (err) {
      setLoadError('Unable to load MLflow runs');
      if (showToast) {
        addNotification('Failed to refresh experiments', 'error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const loadActiveModel = async () => {
    try {
      const data = await api.getActiveModel();
      setActiveModel(data?.run_id ? data : null);
    } catch (err) {
      setActiveModel(null);
    }
  };

  useEffect(() => {
    loadRuns();
    loadActiveModel();
  }, []);

  const handleRollback = async (exp) => {
    addNotification(`Setting active model to "${exp.name}"...`, 'info', 3000);
    try {
      await api.setActiveModel(exp.id);
      setActiveModel({ run_id: exp.id });
      addNotification(`Active model set to ${exp.modelVersion}`, 'success');
    } catch (err) {
      addNotification('Failed to set active model', 'error');
    }
  };

  const handleExport = (exp) => {
    const url = api.exportRunModelUrl(exp.id);
    window.open(url, '_blank', 'noreferrer');
    addNotification(`Exporting ${exp.name} model...`, 'info');
  };

  const handleTrain = async () => {
    addNotification('Starting training job...', 'info');
    try {
      const result = await api.startTraining({ run_task4: true });
      addNotification(`Training job queued: ${result.job_id}`, 'success', 6000);
    } catch (err) {
      addNotification('Failed to start training job', 'error');
    }
  };

  const toggleCompare = (id) => {
    setCompareSelection((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 3 ? [...prev, id] : prev
    );
  };

  const compareExps = experiments.filter((e) => compareSelection.includes(e.id));
  const bestExperiment = useMemo(() => {
    if (activeModel?.run_id) {
      const match = experiments.find((exp) => exp.id === activeModel.run_id);
      if (match) return match;
    }
    const completed = experiments.filter((exp) => exp.status === 'completed');
    return completed.sort((a, b) => (b.metrics?.f1 || 0) - (a.metrics?.f1 || 0))[0];
  }, [experiments, activeModel]);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <GitBranch className="w-8 h-8 text-purple-400" />
            MLOps & Versioning
            <Tooltip text="Track experiment history, model and dataset versions. Rollback to previous versions when needed." />
          </h1>
          <p className="text-gray-400 mt-1">Experiment tracking, model versioning, and rollback</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleTrain}
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all border border-emerald-500/40 bg-emerald-600/20 text-emerald-300 hover:bg-emerald-600/30"
          >
            <Play className="w-4 h-4" />
            Run Training
          </button>
          <button
            onClick={() => loadRuns(true)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all border border-gray-700 bg-gray-800 text-gray-400 hover:text-gray-200"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={() => {
              setCompareMode(!compareMode);
              setCompareSelection([]);
            }}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all border ${
              compareMode
                ? 'bg-purple-600/20 border-purple-500/40 text-purple-300'
                : 'bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-200'
            }`}
          >
            <ArrowLeftRight className="w-4 h-4" />
            {compareMode ? 'Exit Compare' : 'Compare Experiments'}
          </button>
        </div>
      </div>

      {/* Version Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-2">
            <Tag className="w-5 h-5 text-indigo-400" />
            <p className="font-medium text-white">Current Model</p>
          </div>
          <p className="text-2xl font-bold text-indigo-300">
            {bestExperiment?.modelVersion || 'v1.0'}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            {bestExperiment?.algorithm || 'Model'} · {bestExperiment?.id || 'n/a'}
          </p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-2">
            <Database className="w-5 h-5 text-cyan-400" />
            <p className="font-medium text-white">Dataset Version</p>
          </div>
          <p className="text-2xl font-bold text-cyan-300">{bestExperiment?.datasetVersion || 'v1.0'}</p>
          <p className="text-sm text-gray-500 mt-1">IEEE-CIS · Processed</p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-2">
            <GitBranch className="w-5 h-5 text-emerald-400" />
            <p className="font-medium text-white">Total Experiments</p>
          </div>
          <p className="text-2xl font-bold text-emerald-300">{experiments.length}</p>
          <p className="text-sm text-gray-500 mt-1">{experiments.filter((e) => e.status === 'completed').length} completed · {experiments.filter((e) => e.status === 'running').length} running</p>
        </div>
      </div>

      {/* Comparison Panel */}
      {compareMode && compareSelection.length >= 2 && (
        <div className="bg-purple-500/5 border border-purple-500/20 rounded-xl overflow-hidden">
          <div className="p-5 border-b border-purple-500/20">
            <h2 className="text-lg font-semibold text-white">Experiment Comparison</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-800/30">
                  <th className="text-left px-5 py-3 text-gray-400 font-medium"></th>
                  {compareExps.map((exp) => (
                    <th key={exp.id} className="text-center px-5 py-3 text-white font-medium">{exp.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {['algorithm', 'date', 'modelVersion', 'datasetVersion'].map((field) => (
                  <tr key={field} className="border-b border-gray-800">
                    <td className="px-5 py-2.5 text-gray-400 capitalize">{field.replace(/([A-Z])/g, ' $1')}</td>
                    {compareExps.map((exp) => (
                      <td key={exp.id} className="px-5 py-2.5 text-center text-gray-300">{exp[field]}</td>
                    ))}
                  </tr>
                ))}
                {['accuracy', 'f1', 'auc'].map((metric) => (
                  <tr key={metric} className="border-b border-gray-800">
                    <td className="px-5 py-2.5 text-gray-400">{metric.toUpperCase()}</td>
                    {compareExps.map((exp) => {
                      const metrics = exp.metrics || {};
                      const val = metrics[metric];
                      const best = Math.max(...compareExps.map((e) => (e.metrics || {})[metric] || 0));
                      return (
                        <td key={exp.id} className={`px-5 py-2.5 text-center font-mono ${val === best ? 'text-emerald-400 font-semibold' : 'text-gray-300'}`}>
                          {val ? val.toFixed(3) : 'n/a'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Experiments List */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="p-5 border-b border-gray-800">
          <h2 className="text-lg font-semibold text-white">Experiment History</h2>
          <p className="text-sm text-gray-500 mt-1">All training runs with versioning metadata</p>
          {loadError && (
            <p className="text-xs text-amber-400 mt-2">{loadError}</p>
          )}
        </div>

        <div className="divide-y divide-gray-800">
          {experiments.map((exp) => {
            const isExpanded = expandedExp === exp.id;
            const isSelected = compareSelection.includes(exp.id);
            return (
              <div key={exp.id} className={`${isSelected ? 'bg-purple-500/5' : ''}`}>
                <div className="flex items-center gap-4 px-5 py-4">
                  {/* Compare checkbox */}
                  {compareMode && (
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleCompare(exp.id)}
                      className="accent-purple-500"
                      disabled={!isSelected && compareSelection.length >= 3}
                    />
                  )}

                  {/* Status */}
                  {exp.status === 'completed' ? (
                    <CheckCircle className="w-5 h-5 text-emerald-400 shrink-0" />
                  ) : (
                    <Clock className="w-5 h-5 text-amber-400 animate-pulse shrink-0" />
                  )}

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="font-medium text-white">{exp.name}</p>
                      <span className="px-2 py-0.5 bg-gray-800 rounded text-xs text-gray-400">{exp.id}</span>
                    </div>
                    <p className="text-sm text-gray-500 mt-0.5">
                      {exp.algorithm} · {exp.date}
                    </p>
                  </div>

                  {/* Versions */}
                  <div className="hidden sm:flex items-center gap-3">
                    <span className="px-2 py-0.5 bg-indigo-500/10 text-indigo-300 rounded text-xs">Model {exp.modelVersion}</span>
                    <span className="px-2 py-0.5 bg-cyan-500/10 text-cyan-300 rounded text-xs">Data {exp.datasetVersion}</span>
                  </div>

                  {/* Metrics */}
                  <div className="text-right">
                    {exp.metrics?.f1 ? (
                      <p className="font-mono text-emerald-400 text-sm">F1: {exp.metrics.f1.toFixed(3)}</p>
                    ) : (
                      <p className="text-amber-400 text-sm">Training...</p>
                    )}
                  </div>

                  {/* Expand */}
                  <button
                    onClick={() => setExpandedExp(isExpanded ? null : exp.id)}
                    className="text-gray-500 hover:text-gray-300 transition-colors"
                  >
                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="px-5 pb-5 space-y-4">
                    <div className="bg-gray-800/50 rounded-lg p-4 grid grid-cols-2 sm:grid-cols-4 gap-4">
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Accuracy</p>
                        <p className="font-mono text-white">{exp.metrics?.accuracy?.toFixed(3) || 'n/a'}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">F1-Score</p>
                        <p className="font-mono text-white">{exp.metrics?.f1?.toFixed(3) || 'n/a'}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">AUC-ROC</p>
                        <p className="font-mono text-white">{exp.metrics?.auc?.toFixed(3) || 'n/a'}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Parameters</p>
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(exp.params).map(([k, v]) => (
                            <span key={k} className="text-xs bg-gray-900 rounded px-1.5 py-0.5 text-gray-400">
                              {k}={String(v)}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {exp.status === 'completed' && (
                        <>
                          <button
                            onClick={() => handleRollback(exp)}
                            className="bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 text-sm px-3 py-1.5 rounded-lg flex items-center gap-1.5 border border-amber-500/20 transition-colors"
                          >
                            <RotateCcw className="w-3.5 h-3.5" />
                            Rollback to This Version
                          </button>
                          <button
                            onClick={() => handleExport(exp)}
                            className="bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm px-3 py-1.5 rounded-lg flex items-center gap-1.5 border border-gray-700 transition-colors"
                          >
                            <Download className="w-3.5 h-3.5" />
                            Export Model (zip)
                          </button>
                        </>
                      )}
                      {exp.status === 'running' && (
                        <div className="flex items-center gap-2 text-sm text-amber-400">
                          <AlertCircle className="w-4 h-4" />
                          This experiment is still running
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* MLflow/DVC Info */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 flex items-start gap-4">
        <GitBranch className="w-5 h-5 text-gray-500 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm text-gray-300 font-medium">MLflow / DVC Integration</p>
          <p className="text-xs text-gray-500 mt-1">
            This view tracks model versions, dataset versions, and experiment parameters. When connected to a backend,
            it integrates with MLflow for experiment tracking and DVC for data/model versioning. Each experiment is logged
            with full reproducibility metadata.
          </p>
          <div className="flex gap-3 mt-3">
            <span className="px-2.5 py-1 bg-indigo-500/10 text-indigo-300 rounded text-xs">MLflow Compatible</span>
            <span className="px-2.5 py-1 bg-emerald-500/10 text-emerald-300 rounded text-xs">DVC Compatible</span>
            <span className="px-2.5 py-1 bg-amber-500/10 text-amber-300 rounded text-xs">Git Tracked</span>
          </div>
        </div>
      </div>
    </div>
  );
}
