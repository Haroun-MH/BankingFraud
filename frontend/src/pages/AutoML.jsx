import { useEffect, useMemo, useState } from 'react';
import { api } from '../services/api';
import { useNotifications } from '../context/NotificationContext';
import Tooltip from '../components/Tooltip';
import {
  Wand2,
  Play,
  Trophy,
  Loader2,
  TrendingUp,
} from 'lucide-react';

export default function AutoML() {
  const [isRunning, setIsRunning] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [summary, setSummary] = useState(null);
  const [searchMethod, setSearchMethod] = useState('optuna');
  const [maxTrials, setMaxTrials] = useState(20);
  const [timeLimit, setTimeLimit] = useState(20);
  const { addNotification } = useNotifications();

  const startAutoML = async () => {
    setIsRunning(true);
    setJob(null);
    setSummary(null);
    addNotification('AutoML training started...', 'info', 5000);
    try {
      const result = await api.startTraining({
        run_task4: false,
        selected_models: null,
        use_sample: true,
        search_method: searchMethod,
        max_trials: maxTrials,
        time_limit: timeLimit,
      });
      setJobId(result.job_id);
    } catch (err) {
      setIsRunning(false);
      addNotification('Failed to start AutoML job', 'error');
    }
  };

  useEffect(() => {
    if (!jobId) return;
    let mounted = true;

    const poll = async () => {
      try {
        const data = await api.getJob(jobId);
        if (mounted) {
          setJob(data);
          if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
            setIsRunning(false);
          }
        }
      } catch (err) {
        if (mounted) {
          addNotification('Unable to fetch AutoML status', 'warning');
        }
      }
    };

    poll();
    const interval = setInterval(poll, 2000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [jobId, addNotification]);

  useEffect(() => {
    if (job?.status !== 'completed') return;
    const loadSummary = async () => {
      try {
        const data = await api.getSummary();
        setSummary(data);
      } catch (err) {
        setSummary(null);
      }
    };
    loadSummary();
  }, [job?.status]);

  const progress = job?.progress ? job.progress * 100 : 0;
  const history = job?.history || [];

  const bestModel = useMemo(() => {
    const models = summary?.models || {};
    const entries = Object.values(models);
    if (entries.length === 0) return null;
    return entries.sort((a, b) => (b.metrics?.f1 || 0) - (a.metrics?.f1 || 0))[0];
  }, [summary]);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Wand2 className="w-8 h-8 text-purple-400" />
            AutoML Mode
            <Tooltip text="AutoML runs a compact multi-model search and ranks the best run." />
          </h1>
          <p className="text-gray-400 mt-1">
            One click to find the best model based on live MLflow runs
          </p>
        </div>
      </div>

      {/* Configuration */}
      {!isRunning && (!job || job.status !== 'completed') && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-5">
          <h2 className="text-lg font-semibold text-white">AutoML Configuration</h2>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
                Search Method
                <Tooltip text="Controls how candidate hyperparameters are sampled" />
              </label>
              <select
                value={searchMethod}
                onChange={(e) => setSearchMethod(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="optuna">Optuna (Bayesian)</option>
                <option value="grid">GridSearch (Exhaustive)</option>
                <option value="random">RandomSearch (Sampling)</option>
              </select>
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
                Max Trials
                <Tooltip text="Upper bound on parameter trials across models" />
              </label>
              <input
                type="number"
                value={maxTrials}
                onChange={(e) => setMaxTrials(parseInt(e.target.value, 10) || 10)}
                min={5}
                max={200}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
                Time Limit (min)
                <Tooltip text="Stops the AutoML search after the time limit" />
              </label>
              <input
                type="number"
                value={timeLimit}
                onChange={(e) => setTimeLimit(parseInt(e.target.value, 10) || 10)}
                min={5}
                max={120}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
          </div>

          <button
            onClick={startAutoML}
            className="bg-purple-600 hover:bg-purple-500 text-white font-semibold px-8 py-3 rounded-xl flex items-center gap-3 transition-colors text-lg shadow-lg shadow-purple-600/20"
          >
            <Play className="w-5 h-5" />
            Launch AutoML
          </button>
        </div>
      )}

      {/* Running State */}
      {isRunning && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
              AutoML Running...
            </h2>
            <span className="text-sm font-mono text-purple-300">{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-purple-600 to-indigo-500 h-3 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="text-sm text-gray-400">
            {job?.message || 'Evaluating candidate models...'}
          </div>
        </div>
      )}

      {/* Results */}
      {job?.status === 'completed' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Trophy className="w-6 h-6 text-amber-400" />
              <h2 className="text-lg font-semibold text-white">Best Model</h2>
            </div>
            {bestModel ? (
              <div>
                <p className="text-xl font-semibold text-white">{bestModel.algorithm}</p>
                <p className="text-sm text-gray-500 mt-1">Run ID: {bestModel.run_id}</p>
                <div className="mt-4 space-y-2">
                  {[
                    { label: 'Accuracy', value: bestModel.metrics?.accuracy },
                    { label: 'F1 Score', value: bestModel.metrics?.f1 },
                    { label: 'AUC-ROC', value: bestModel.metrics?.auc },
                  ].map((metric) => (
                    <div key={metric.label} className="flex justify-between text-sm">
                      <span className="text-gray-400">{metric.label}</span>
                      <span className="text-white font-mono">
                        {metric.value != null ? metric.value.toFixed(3) : 'n/a'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-500">No best model available yet.</p>
            )}
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <TrendingUp className="w-6 h-6 text-emerald-400" />
              <h2 className="text-lg font-semibold text-white">Run History</h2>
            </div>
            {history.length === 0 && <p className="text-sm text-gray-500">No history entries yet.</p>}
            {history.length > 0 && (
              <div className="space-y-2">
                {history.slice(-6).map((entry, idx) => (
                  <div key={idx} className="text-sm text-gray-300 flex justify-between">
                    <span>{entry.model || 'Model'} · step {entry.step}</span>
                    <span className="font-mono">F1 {entry.metrics?.val?.f1?.toFixed(3) || 'n/a'}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {job?.status === 'failed' && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-5 text-sm text-red-300">
          AutoML failed: {job?.message || 'Unknown error'}
        </div>
      )}

      {job?.status === 'cancelled' && (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-5 text-sm text-amber-300">
          AutoML stopped: {job?.message || 'Cancelled'}
        </div>
      )}
    </div>
  );
}
