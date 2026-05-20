import { useEffect, useMemo, useState } from 'react';
import { useNotifications } from '../context/NotificationContext';
import Tooltip from '../components/Tooltip';
import { api } from '../services/api';
import {
  Activity,
  Play,
  Pause,
  Square,
  Cpu,
  HardDrive,
  MemoryStick,
  Clock,
  TrendingUp,
  Zap,
  BarChart,
  RefreshCw,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RTooltip,
  AreaChart,
  Area,
} from 'recharts';

const TRAINING_OPTIONS = [
  { id: 'all', label: 'All Models (Task 3 + Task 4)', selectedModels: null, runTask4: true },
  { id: 'rf', label: 'Random Forest Only', selectedModels: ['random_forest'], runTask4: true },
  { id: 'knn', label: 'KNN Only', selectedModels: ['knn'], runTask4: false },
  { id: 'svm', label: 'SVM Only', selectedModels: ['svm'], runTask4: false },
  { id: 'lr', label: 'Logistic Regression Only', selectedModels: ['logreg'], runTask4: false },
  { id: 'ada', label: 'AdaBoost Only', selectedModels: ['adaboost'], runTask4: false },
  { id: 'xgb', label: 'XGBoost Only', selectedModels: ['xgboost'], runTask4: false },
];

export default function TrainingMonitor() {
  const [activeJobId, setActiveJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [trainingOption, setTrainingOption] = useState('all');
  const [costlyMaxSamples, setCostlyMaxSamples] = useState(20000);  // KNN/SVM cap — keep low for speed
  const [systemMetrics, setSystemMetrics] = useState({ cpu: 0, gpu: 0, ram: 0, gpu_mem: 0 });
  const [elapsedTime, setElapsedTime] = useState(0);
  const { addNotification } = useNotifications();

  // Continuously poll /api/jobs to pick up any active job, including ones
  // started from other pages. Runs every 3s regardless of activeJobId.
  useEffect(() => {
    let mounted = true;

    const discover = async () => {
      try {
        const jobs = await api.listJobs();
        if (!mounted || !Array.isArray(jobs)) return;

        const active = jobs
          .filter((j) => ['running', 'queued', 'paused', 'cancelling'].includes(j.status))
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0));

        if (active.length > 0) {
          // Switch to this job if we don't already have one tracked
          setActiveJobId((prev) => prev ?? active[0].id);
        }
      } catch (_) {
        // backend not ready — ignore silently
      }
    };

    discover();
    const interval = setInterval(discover, 3000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const selectedConfig = TRAINING_OPTIONS.find((opt) => opt.id === trainingOption);
  const isTraining = ['running', 'queued', 'paused', 'cancelling'].includes(job?.status);

  const stepData = useMemo(() => {
    const history = job?.history || [];
    return history.map((entry, index) => {
      const trainAcc = entry.metrics?.train?.accuracy ?? null;
      const valAcc = entry.metrics?.val?.accuracy ?? null;
      return {
        step: entry.step ?? index + 1,
        trainLoss: trainAcc != null ? 1 - trainAcc : null,
        valLoss: valAcc != null ? 1 - valAcc : null,
        trainAcc,
        valAcc,
        f1Score: entry.metrics?.val?.f1 ?? null,
      };
    });
  }, [job]);

  const currentStep = stepData.length;
  const totalSteps = job?.total_steps || (currentStep > 0 ? currentStep : 1);
  const progress = job?.progress ? job.progress * 100 : 0;
  const latestData = stepData.length > 0 ? stepData[stepData.length - 1] : null;

  const startTraining = async () => {
    const payload = {
      run_task4: selectedConfig.runTask4,
      selected_models: selectedConfig.selectedModels,
      use_sample: true,
      costly_max_samples: costlyMaxSamples,
      run_pca: false,
      run_tsne: false,
    };

    addNotification('Starting training job...', 'info');
    try {
      const result = await api.startTraining(payload);
      setActiveJobId(result.job_id);
      addNotification(`Training job queued: ${result.job_id}`, 'success', 6000);
    } catch (err) {
      addNotification('Failed to start training job', 'error');
    }
  };

  const refreshJob = async () => {
    if (!activeJobId) {
      return;
    }
    try {
      const data = await api.getJob(activeJobId);
      setJob(data);
    } catch (err) {
      addNotification('Failed to refresh job status', 'warning');
    }
  };

  const pauseTraining = async () => {
    if (!activeJobId) return;
    try {
      await api.pauseJob(activeJobId);
      await refreshJob();
      addNotification('Training paused', 'info');
    } catch (err) {
      addNotification('Failed to pause training', 'error');
    }
  };

  const resumeTraining = async () => {
    if (!activeJobId) return;
    try {
      await api.resumeJob(activeJobId);
      await refreshJob();
      addNotification('Training resumed', 'success');
    } catch (err) {
      addNotification('Failed to resume training', 'error');
    }
  };

  const stopTraining = async () => {
    if (!activeJobId) return;
    try {
      await api.cancelJob(activeJobId);
      await refreshJob();
      addNotification('Training cancellation requested', 'warning');
    } catch (err) {
      addNotification('Failed to cancel training', 'error');
    }
  };

  useEffect(() => {
    if (!activeJobId) {
      return;
    }

    let mounted = true;

    const poll = async () => {
      try {
        const data = await api.getJob(activeJobId);
        if (mounted) {
          setJob(data);
          // When job reaches a terminal state, release the lock so the
          // discovery loop can pick up the next job started from any page
          if (['completed', 'failed', 'cancelled'].includes(data.status)) {
            setTimeout(() => {
              if (mounted) setActiveJobId(null);
            }, 5000); // keep showing results for 5s before releasing
          }
        }
      } catch (err) {
        if (mounted) {
          addNotification('Unable to fetch job status', 'warning');
        }
      }
    };

    poll();
    const interval = setInterval(poll, 2000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [activeJobId, addNotification]);

  useEffect(() => {
    if (!job?.started_at) {
      return;
    }
    const timer = setInterval(() => {
      const now = Date.now() / 1000;
      const end = job.ended_at || now;
      setElapsedTime(Math.max(0, Math.floor(end - job.started_at)));
    }, 1000);
    return () => clearInterval(timer);
  }, [job?.started_at, job?.ended_at]);

  useEffect(() => {
    if (!isTraining) {
      return;
    }
    let mounted = true;
    const poll = async () => {
      try {
        const metrics = await api.getSystemMetrics();
        if (mounted) {
          setSystemMetrics(metrics || { cpu: null, gpu: null, ram: null, gpu_mem: null });
        }
      } catch (err) {
        if (mounted) {
          setSystemMetrics({ cpu: null, gpu: null, ram: null, gpu_mem: null });
        }
      }
    };
    poll();
    const interval = setInterval(poll, 2000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [isTraining]);

  const formatTime = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const eta = currentStep > 0 ? Math.round((elapsedTime / currentStep) * (totalSteps - currentStep)) : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Activity className="w-8 h-8 text-cyan-400" />
            Training Monitor
            <Tooltip text="Monitor training jobs and MLflow-backed progress updates." />
          </h1>
          <p className="text-gray-400 mt-1">Track training jobs started from the UI</p>
        </div>
        {job?.status === 'running' && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-sm text-emerald-400 font-medium">Live</span>
          </div>
        )}
      </div>

      {/* Control Panel */}
      {!isTraining && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">Training Configuration</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="text-sm text-gray-300 mb-1.5 block">Scope</label>
              <select
                value={trainingOption}
                onChange={(e) => setTrainingOption(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
              >
                {TRAINING_OPTIONS.map((opt) => (
                  <option key={opt.id} value={opt.id}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm text-gray-300 mb-1.5 block">Max Samples (KNN/SVM)</label>
              <input
                type="number"
                value={costlyMaxSamples}
                onChange={(e) => setCostlyMaxSamples(parseInt(e.target.value, 10) || 100000)}
                min={10000}
                max={500000}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
            </div>
            <div className="flex items-end gap-2">
              <button
                onClick={startTraining}
                className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold px-6 py-2.5 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                <Play className="w-4 h-4" />
                Start Training
              </button>
              <button
                onClick={refreshJob}
                className="bg-gray-800 hover:bg-gray-700 text-gray-300 px-4 py-2.5 rounded-lg border border-gray-700"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Status Bar */}
      {isTraining && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-4">
              <h2 className="text-lg font-semibold text-white">Training Job</h2>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-300">
                {job?.status}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {job?.status === 'paused' ? (
                <button onClick={resumeTraining} className="bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 p-2 rounded-lg transition-colors">
                  <Play className="w-4 h-4" />
                </button>
              ) : (
                <button onClick={pauseTraining} className="bg-amber-600/20 hover:bg-amber-600/30 text-amber-400 p-2 rounded-lg transition-colors">
                  <Pause className="w-4 h-4" />
                </button>
              )}
              <button onClick={stopTraining} className="bg-red-600/20 hover:bg-red-600/30 text-red-400 p-2 rounded-lg transition-colors">
                <Square className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mb-3">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">Step {currentStep} / {totalSteps}</span>
              <span className="font-mono text-cyan-300">{progress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-cyan-600 to-indigo-500 h-3 rounded-full transition-all duration-500 relative overflow-hidden"
                style={{ width: `${progress}%` }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse" />
              </div>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-gray-800/50 rounded-lg p-3 text-center">
              <Clock className="w-4 h-4 text-gray-500 mx-auto mb-1" />
              <p className="text-lg font-mono text-white">{formatTime(elapsedTime)}</p>
              <p className="text-xs text-gray-500">Elapsed</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-3 text-center">
              <Clock className="w-4 h-4 text-gray-500 mx-auto mb-1" />
              <p className="text-lg font-mono text-white">{formatTime(eta)}</p>
              <p className="text-xs text-gray-500">ETA</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-3 text-center">
              <TrendingUp className="w-4 h-4 text-emerald-500 mx-auto mb-1" />
              <p className="text-lg font-mono text-emerald-400">
                {latestData?.f1Score != null ? latestData.f1Score.toFixed(3) : 'n/a'}
              </p>
              <p className="text-xs text-gray-500">Current F1</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-3 text-center">
              <Zap className="w-4 h-4 text-amber-500 mx-auto mb-1" />
              <p className="text-lg font-mono text-amber-400">{job?.status}</p>
              <p className="text-xs text-gray-500">Status</p>
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      {stepData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Loss Curve */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <BarChart className="w-4 h-4 text-red-400" />
              Loss (Derived)
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={stepData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="step" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <RTooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }} />
                  <Line dataKey="trainLoss" stroke="#f87171" dot={false} strokeWidth={2} name="Train Loss" />
                  <Line dataKey="valLoss" stroke="#fbbf24" dot={false} strokeWidth={2} name="Val Loss" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Accuracy Curve */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              Accuracy
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={stepData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="step" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <RTooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }} />
                  <Line dataKey="trainAcc" stroke="#34d399" dot={false} strokeWidth={2} name="Train Acc" />
                  <Line dataKey="valAcc" stroke="#818cf8" dot={false} strokeWidth={2} name="Val Acc" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* F1 Score over time */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <Zap className="w-4 h-4 text-purple-400" />
              F1-Score Progress
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={stepData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="step" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <RTooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }} />
                  <Area dataKey="f1Score" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.1} strokeWidth={2} name="F1-Score" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* System Resources */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              System Resources
            </h3>
            <div className="space-y-4">
              {[
                { label: 'CPU', value: systemMetrics.cpu, icon: Cpu, color: 'bg-cyan-500', textColor: 'text-cyan-400' },
                { label: 'GPU', value: systemMetrics.gpu, icon: Zap, color: 'bg-purple-500', textColor: 'text-purple-400' },
                { label: 'RAM', value: systemMetrics.ram, icon: MemoryStick, color: 'bg-amber-500', textColor: 'text-amber-400' },
                { label: 'GPU Memory', value: systemMetrics.gpu_mem, icon: HardDrive, color: 'bg-emerald-500', textColor: 'text-emerald-400' },
              ].map((metric) => (
                <div key={metric.label}>
                  <div className="flex items-center justify-between text-sm mb-1.5">
                    <span className="text-gray-400 flex items-center gap-2">
                      <metric.icon className={`w-4 h-4 ${metric.textColor}`} />
                      {metric.label}
                    </span>
                    <span className={`font-mono ${metric.textColor}`}>
                      {metric.value != null ? `${metric.value.toFixed(1)}%` : 'n/a'}
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2.5">
                    <div
                      className={`${metric.color} h-2.5 rounded-full transition-all duration-500`}
                      style={{ width: `${Math.min(metric.value || 0, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {isTraining && (
              <div className="mt-4 flex items-center justify-center gap-2 text-xs text-gray-500">
                <RefreshCw className="w-3 h-3 animate-spin" />
                Updating in real-time
              </div>
            )}
          </div>
        </div>
      )}

      {!isTraining && stepData.length === 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-16 text-center">
          <Activity className="w-16 h-16 text-gray-700 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-400 mb-2">No Active Training</h2>
          <p className="text-sm text-gray-500 max-w-md mx-auto">
            Start a training job to see progress, metrics, and MLflow-backed summaries.
          </p>
        </div>
      )}

      {!isTraining && stepData.length > 0 && (
        <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="bg-emerald-500/20 p-2 rounded-lg">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <p className="font-semibold text-white">Training Complete</p>
              <p className="text-sm text-gray-400">Job {activeJobId || 'n/a'} completed</p>
            </div>
          </div>
          {latestData && (
            <div className="flex gap-6 text-sm">
              <div>
                <span className="text-gray-400">Final Loss: </span>
                <span className="font-mono text-white">
                  {latestData.valLoss != null ? latestData.valLoss.toFixed(4) : 'n/a'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Final Accuracy: </span>
                <span className="font-mono text-white">
                  {latestData.valAcc != null ? `${(latestData.valAcc * 100).toFixed(2)}%` : 'n/a'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Final F1: </span>
                <span className="font-mono text-emerald-400">
                  {latestData.f1Score != null ? latestData.f1Score.toFixed(3) : 'n/a'}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
