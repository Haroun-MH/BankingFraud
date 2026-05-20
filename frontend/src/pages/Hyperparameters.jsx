import { useEffect, useState } from 'react';
import { api } from '../services/api';
import { ALGORITHMS, SAVED_CONFIGS } from '../data/mockData';
import { useNotifications } from '../context/NotificationContext';
import Tooltip from '../components/Tooltip';
import {
  SlidersHorizontal,
  Save,
  RotateCcw,
  Search,
  Zap,
  FolderOpen,
  Trash2,
  Play,
} from 'lucide-react';

export default function Hyperparameters() {
  const [selectedAlgo, setSelectedAlgo] = useState('rf');
  const [params, setParams] = useState(() => {
    const algo = ALGORITHMS.find((a) => a.id === 'rf');
    return Object.fromEntries(algo.hyperparameters.map((hp) => [hp.name, hp.default]));
  });
  const [savedConfigs, setSavedConfigs] = useState(SAVED_CONFIGS);
  const [configName, setConfigName] = useState('');
  const [tuningMethod, setTuningMethod] = useState('grid');
  const [tuningTrials, setTuningTrials] = useState(12);
  const [tuningJobId, setTuningJobId] = useState(null);
  const [tuningJob, setTuningJob] = useState(null);
  const [isTuning, setIsTuning] = useState(false);
  const { addNotification } = useNotifications();

  const currentAlgo = ALGORITHMS.find((a) => a.id === selectedAlgo);

  const handleAlgoChange = (id) => {
    setSelectedAlgo(id);
    const algo = ALGORITHMS.find((a) => a.id === id);
    setParams(Object.fromEntries(algo.hyperparameters.map((hp) => [hp.name, hp.default])));
  };

  const handleParamChange = (name, value) => {
    setParams((prev) => ({ ...prev, [name]: value }));
  };

  const resetDefaults = () => {
    setParams(Object.fromEntries(currentAlgo.hyperparameters.map((hp) => [hp.name, hp.default])));
    addNotification('Parameters reset to defaults', 'info');
  };

  const saveConfig = () => {
    const name = configName.trim() || `${currentAlgo.shortName} Config ${savedConfigs.length + 1}`;
    setSavedConfigs((prev) => [
      ...prev,
      { id: Date.now(), name, algorithm: selectedAlgo, params: { ...params }, date: '2026-03-04' },
    ]);
    setConfigName('');
    addNotification(`Configuration "${name}" saved!`, 'success');
  };

  const loadConfig = (config) => {
    setSelectedAlgo(config.algorithm);
    setParams(config.params);
    addNotification(`Loaded "${config.name}"`, 'info');
  };

  const deleteConfig = (id) => {
    setSavedConfigs((prev) => prev.filter((c) => c.id !== id));
    addNotification('Configuration deleted', 'warning');
  };

  const startTuning = async () => {
    const modelMap = {
      rf: 'random_forest',
      svm: 'svm',
      knn: 'knn',
      lr: 'logreg',
      ada: 'adaboost',
      xgb: 'xgboost',
    };
    const mapped = modelMap[currentAlgo.id];
    if (!mapped) {
      addNotification('This algorithm is not wired to backend tuning yet.', 'warning');
      return;
    }

    const searchSpace = currentAlgo.hyperparameters.map((hp) => ({
      name: hp.name,
      type: hp.type,
      min: hp.min,
      max: hp.max,
      step: hp.step,
      options: hp.options,
    }));

    setIsTuning(true);
    addNotification(`Starting ${tuningMethod} tuning for ${currentAlgo.shortName}...`, 'info', 6000);
    try {
      const result = await api.tuneModel({
        model_type: mapped,
        search_method: tuningMethod,
        max_trials: tuningTrials,
        search_space: searchSpace,
      });
      setTuningJobId(result.job_id);
    } catch (err) {
      setIsTuning(false);
      addNotification('Failed to start tuning job', 'error');
    }
  };

  useEffect(() => {
    if (!tuningJobId) return;
    let mounted = true;
    const poll = async () => {
      try {
        const data = await api.getJob(tuningJobId);
        if (mounted) {
          setTuningJob(data);
          if (['completed', 'failed', 'cancelled'].includes(data.status)) {
            setIsTuning(false);
          }
        }
      } catch (err) {
        if (mounted) {
          setIsTuning(false);
        }
      }
    };
    poll();
    const interval = setInterval(poll, 2000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [tuningJobId]);

  useEffect(() => {
    if (tuningJob?.status !== 'completed') return;
    const best = tuningJob?.result?.best_params;
    if (best) {
      setParams((prev) => ({ ...prev, ...best }));
      addNotification('Tuning completed! Optimal parameters applied.', 'success', 5000);
    } else {
      addNotification('Tuning completed but no parameters were returned.', 'warning');
    }
  }, [tuningJob, addNotification]);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <SlidersHorizontal className="w-8 h-8 text-indigo-400" />
          Hyperparameters
          <Tooltip text="Configure hyperparameters for your chosen algorithm. Use auto-tuning for optimal settings." />
        </h1>
        <p className="text-gray-400 mt-1">Fine-tune model parameters or use automatic tuning</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel: Algorithm Selection + Parameters */}
        <div className="lg:col-span-2 space-y-6">
          {/* Algorithm selector */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h2 className="text-sm font-medium text-gray-300 mb-3">Algorithm</h2>
            <div className="flex flex-wrap gap-2">
              {ALGORITHMS.map((algo) => (
                <button
                  key={algo.id}
                  onClick={() => handleAlgoChange(algo.id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    selectedAlgo === algo.id
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200 border border-gray-700'
                  }`}
                >
                  {algo.shortName}
                </button>
              ))}
            </div>
          </div>

          {/* Hyperparameter Form */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-lg font-semibold text-white">{currentAlgo.name} Parameters</h2>
              <button
                onClick={resetDefaults}
                className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-200 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                Reset Defaults
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {currentAlgo.hyperparameters.map((hp) => (
                <div key={hp.name}>
                  <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-1.5">
                    {hp.name}
                    <Tooltip text={hp.description} />
                  </label>
                  {hp.type === 'number' ? (
                    <div className="space-y-2">
                      <input
                        type="range"
                        min={hp.min}
                        max={hp.max}
                        step={hp.step}
                        value={params[hp.name] ?? hp.default}
                        onChange={(e) => handleParamChange(hp.name, parseFloat(e.target.value))}
                        className="w-full accent-indigo-500"
                      />
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500">{hp.min}</span>
                        <input
                          type="number"
                          min={hp.min}
                          max={hp.max}
                          step={hp.step}
                          value={params[hp.name] ?? hp.default}
                          onChange={(e) => handleParamChange(hp.name, parseFloat(e.target.value))}
                          className="w-24 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-center text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        />
                        <span className="text-xs text-gray-500">{hp.max}</span>
                      </div>
                    </div>
                  ) : hp.type === 'select' ? (
                    <select
                      value={params[hp.name] ?? hp.default}
                      onChange={(e) => handleParamChange(hp.name, e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      {hp.options.map((opt) => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={params[hp.name] ?? hp.default}
                      onChange={(e) => handleParamChange(hp.name, e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-3 mt-6 pt-5 border-t border-gray-800">
              <button
                onClick={async () => {
                  addNotification(`Starting ${currentAlgo.shortName} training with custom parameters...`, 'info');
                  try {
                    const modelMap = {
                      rf: 'random_forest',
                      svm: 'svm',
                      knn: 'knn',
                      lr: 'logreg',
                      ada: 'adaboost',
                      xgb: 'xgboost',
                    };
                    const mapped = modelMap[currentAlgo.id];
                    if (!mapped) {
                      addNotification('This algorithm is not wired to backend training yet.', 'warning');
                      return;
                    }
                    const result = await api.startTraining({
                      selected_models: [mapped],
                      run_task4: currentAlgo.id === 'rf',
                      custom_params: { ...params },
                    });
                    addNotification(`Training job queued: ${result.job_id}`, 'success', 6000);
                  } catch (err) {
                    addNotification('Failed to start training job', 'error');
                  }
                }}
                className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-5 py-2.5 rounded-lg flex items-center gap-2 transition-colors"
              >
                <Play className="w-4 h-4" />
                Train Model
              </button>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="Config name..."
                  value={configName}
                  onChange={(e) => setConfigName(e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 w-40"
                />
                <button
                  onClick={saveConfig}
                  className="bg-gray-800 hover:bg-gray-700 text-gray-200 font-medium px-4 py-2.5 rounded-lg flex items-center gap-2 border border-gray-700 transition-colors"
                >
                  <Save className="w-4 h-4" />
                  Save
                </button>
              </div>
            </div>
          </div>

          {/* Auto-Tuning */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-400" />
              Automatic Tuning
              <Tooltip text="Let the system find optimal hyperparameters automatically using search methods." />
            </h2>
            <p className="text-sm text-gray-400 mb-4">
              Select a tuning method and the system will search for the best hyperparameters.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              {[
                { id: 'grid', label: 'GridSearch', desc: 'Exhaustive search' },
                { id: 'random', label: 'RandomSearch', desc: 'Random sampling' },
                { id: 'optuna', label: 'Optuna', desc: 'Bayesian optimization' },
              ].map((method) => (
                <button
                  key={method.id}
                  onClick={() => setTuningMethod(method.id)}
                  className={`px-4 py-2.5 rounded-lg text-sm transition-all border ${
                    tuningMethod === method.id
                      ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                      : 'bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <span className="font-medium">{method.label}</span>
                  <span className="block text-xs opacity-70">{method.desc}</span>
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-3 mt-4 pt-4 border-t border-gray-800">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-400">Trials</label>
                <input
                  type="number"
                  min={3}
                  max={50}
                  value={tuningTrials}
                  onChange={(e) => setTuningTrials(parseInt(e.target.value, 10) || 12)}
                  className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
                />
              </div>
              <button
                onClick={startTuning}
                disabled={isTuning}
                className="bg-amber-500 hover:bg-amber-400 disabled:opacity-60 disabled:cursor-not-allowed text-black font-semibold px-5 py-2.5 rounded-lg flex items-center gap-2 transition-colors"
              >
                <Search className="w-4 h-4" />
                {isTuning ? 'Tuning...' : 'Start Tuning'}
              </button>
              {tuningJob?.status && (
                <p className="text-xs text-gray-500">
                  Tuning status: {tuningJob.status} {tuningJob.message ? `- ${tuningJob.message}` : ''}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel: Saved Configs */}
        <div className="space-y-4">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <FolderOpen className="w-5 h-5 text-indigo-400" />
              Saved Configurations
            </h2>
            {savedConfigs.length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-6">No saved configurations yet.</p>
            ) : (
              <div className="space-y-3">
                {savedConfigs.map((config) => {
                  const algo = ALGORITHMS.find((a) => a.id === config.algorithm);
                  return (
                    <div
                      key={config.id}
                      className="bg-gray-800/50 border border-gray-700 rounded-lg p-3"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <p className="text-sm font-medium text-white">{config.name}</p>
                          <p className="text-xs text-gray-500">{algo?.shortName} · {config.date}</p>
                        </div>
                        <button
                          onClick={() => deleteConfig(config.id)}
                          className="text-gray-600 hover:text-red-400 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="flex flex-wrap gap-1 mb-2">
                        {Object.entries(config.params).map(([k, v]) => (
                          <span key={k} className="px-1.5 py-0.5 bg-gray-900 rounded text-xs text-gray-400">
                            <span className="text-indigo-400">{k}</span>={String(v)}
                          </span>
                        ))}
                      </div>
                      <button
                        onClick={() => loadConfig(config)}
                        className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
                      >
                        Load Configuration →
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Current Config Summary */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h2 className="text-sm font-medium text-gray-300 mb-3">Current Configuration</h2>
            <div className="bg-gray-950 rounded-lg p-3 font-mono text-xs space-y-1">
              <p className="text-indigo-400"># {currentAlgo.name}</p>
              {Object.entries(params).map(([key, val]) => (
                <p key={key}>
                  <span className="text-gray-400">{key}</span>
                  <span className="text-gray-600"> = </span>
                  <span className="text-amber-300">{String(val)}</span>
                </p>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
