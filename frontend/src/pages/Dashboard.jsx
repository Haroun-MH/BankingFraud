import { Link } from 'react-router-dom';
import { useEffect, useMemo, useState } from 'react';
import {
  Brain,
  Database,
  BarChart3,
  Activity,
  Wand2,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
} from 'lucide-react';
import { api } from '../services/api';

const quickActions = [
  { label: 'Select Models', to: '/models', icon: Brain, desc: 'Choose algorithms to train' },
  { label: 'AutoML', to: '/automl', icon: Wand2, desc: 'Auto-find best model' },
  { label: 'View Data', to: '/data', icon: Database, desc: 'Explore your dataset' },
  { label: 'See Results', to: '/results', icon: BarChart3, desc: 'Visualize performance' },
  { label: 'Monitor Training', to: '/monitoring', icon: Activity, desc: 'Real-time progress' },
];

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [runs, setRuns] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [dataset, setDataset] = useState(null);
  const [activeModel, setActiveModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    const loadAll = async () => {
      setIsLoading(true);
      try {
        const [summaryData, runsData, jobsData, datasetData, activeModelData] = await Promise.all([
          api.getSummary(),
          api.getRuns(50),
          api.listJobs(),
          api.getDatasetSummary(),
          api.getActiveModel(),
        ]);
        if (!mounted) return;
        setSummary(summaryData);
        setRuns(Array.isArray(runsData) ? runsData : []);
        setJobs(Array.isArray(jobsData) ? jobsData : []);
        setDataset(datasetData);
        setActiveModel(activeModelData || null);
      } catch (err) {
        if (mounted) {
          setSummary(null);
          setRuns([]);
          setJobs([]);
          setDataset(null);
          setActiveModel(null);
        }
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    };

    loadAll();
    return () => {
      mounted = false;
    };
  }, []);

  const completedRuns = useMemo(
    () => runs.filter((exp) => exp.status === 'completed'),
    [runs]
  );

  const bestRun = useMemo(() => {
    if (activeModel?.run_id) {
      const match = runs.find((run) => run.id === activeModel.run_id);
      if (match) return match;
    }
    return completedRuns
      .slice()
      .sort((a, b) => (b.metrics?.f1 || 0) - (a.metrics?.f1 || 0))[0];
  }, [activeModel, completedRuns, runs]);

  const bestMetrics = bestRun?.metrics || {};
  const bestModelName = bestRun?.algorithm || 'Model';
  const bestModelVersion = bestRun?.modelVersion || 'n/a';

  const modelCount = Object.keys(summary?.models || {}).length;
  const bestAuc = Math.max(
    0,
    ...Object.values(summary?.models || {}).map((m) => m.metrics?.auc || 0)
  );
  const activeTraining = jobs.filter((job) =>
    ['running', 'queued', 'paused', 'cancelling'].includes(job.status)
  ).length;

  const quickStats = [
    { label: 'Models Trained', value: modelCount || '0', icon: Brain, color: 'text-indigo-400', bg: 'bg-indigo-500/10' },
    { label: 'Best AUC', value: bestAuc ? bestAuc.toFixed(3) : 'n/a', icon: TrendingUp, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
    { label: 'Experiments', value: runs.length.toString(), icon: BarChart3, color: 'text-amber-400', bg: 'bg-amber-500/10' },
    { label: 'Active Training', value: activeTraining.toString(), icon: Activity, color: 'text-cyan-400', bg: 'bg-cyan-500/10' },
  ];

  const recentExperiments = runs.slice(0, 4);

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <p className="text-gray-400 mt-1">
          {dataset?.name || 'Dataset'} — Overview
        </p>
        {isLoading && <p className="text-xs text-gray-500 mt-1">Loading live metrics...</p>}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {quickStats.map((stat) => (
          <div key={stat.label} className="bg-gray-900 border border-gray-800 rounded-xl p-5 flex items-center gap-4">
            <div className={`${stat.bg} p-3 rounded-lg`}>
              <stat.icon className={`w-6 h-6 ${stat.color}`} />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stat.value}</p>
              <p className="text-sm text-gray-400">{stat.label}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {quickActions.map((action) => (
            <Link
              key={action.to}
              to={action.to}
              className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-indigo-500/50 hover:shadow-lg hover:shadow-indigo-500/5 transition-all duration-300 group"
            >
              <action.icon className="w-8 h-8 text-gray-500 group-hover:text-indigo-400 transition-colors mb-3" />
              <p className="font-medium text-white text-sm">{action.label}</p>
              <p className="text-xs text-gray-500 mt-1">{action.desc}</p>
            </Link>
          ))}
        </div>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Experiments */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">Recent Experiments</h2>
            <Link to="/mlops" className="text-sm text-indigo-400 hover:text-indigo-300">
              View All →
            </Link>
          </div>
          <div className="space-y-3">
            {recentExperiments.length === 0 && (
              <p className="text-sm text-gray-500">No experiments yet.</p>
            )}
            {recentExperiments.map((exp) => (
              <div key={exp.id} className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
                <div className="flex items-center gap-3">
                  {exp.status === 'completed' ? (
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <Clock className="w-4 h-4 text-amber-400 animate-pulse" />
                  )}
                  <div>
                    <p className="text-sm font-medium text-gray-200">{exp.name}</p>
                    <p className="text-xs text-gray-500">{exp.algorithm}</p>
                  </div>
                </div>
                <div className="text-right">
                  {exp.metrics?.f1 != null ? (
                    <p className="text-sm font-mono text-emerald-400">F1: {exp.metrics.f1.toFixed(3)}</p>
                  ) : (
                    <p className="text-sm text-amber-400">Training...</p>
                  )}
                  <p className="text-xs text-gray-500">{exp.date || 'n/a'}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Best Model Summary */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Best Model Performance</h2>
          {bestRun ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-emerald-500/10 p-2 rounded-lg">
                  <TrendingUp className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <p className="font-medium text-white">{bestModelName}</p>
                  <p className="text-xs text-gray-500">{bestRun.id} · {bestModelVersion}</p>
                </div>
              </div>

              {[
                { label: 'Accuracy', value: bestMetrics.accuracy, color: 'bg-indigo-500' },
                { label: 'F1-Score', value: bestMetrics.f1, color: 'bg-emerald-500' },
                { label: 'AUC-ROC', value: bestMetrics.auc, color: 'bg-cyan-500' },
                { label: 'Precision', value: bestMetrics.precision, color: 'bg-amber-500' },
                { label: 'Recall', value: bestMetrics.recall, color: 'bg-purple-500' },
              ].map((metric) => (
                <div key={metric.label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">{metric.label}</span>
                    <span className="font-mono text-white">
                      {metric.value != null ? `${(metric.value * 100).toFixed(1)}%` : 'n/a'}
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div
                      className={`${metric.color} h-2 rounded-full transition-all duration-1000`}
                      style={{ width: `${metric.value ? metric.value * 100 : 0}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No completed runs yet.</p>
          )}
        </div>
      </div>

      {/* Dataset Alert */}
      <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4 flex items-center gap-4">
        <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
        <div>
          <p className="text-sm text-amber-200 font-medium">Class Imbalance Detected</p>
          <p className="text-xs text-amber-300/70 mt-0.5">
            Fraud rate: {dataset?.fraud_rate != null ? `${dataset.fraud_rate}%` : 'n/a'}. Consider class weights or
            stratified sampling for better results.
          </p>
        </div>
      </div>
    </div>
  );
}
