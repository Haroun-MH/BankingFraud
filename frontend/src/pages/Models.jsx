import { useEffect, useState } from 'react';
import { api } from '../services/api';
import { ALGORITHMS } from '../data/mockData';
import { useNotifications } from '../context/NotificationContext';
import Tooltip from '../components/Tooltip';
import {
  CheckSquare,
  Square,
  ChevronDown,
  ChevronUp,
  Play,
  Upload,
  ThumbsUp,
  ThumbsDown,
  Info,
  Layers,
} from 'lucide-react';

export default function Models() {
  const [selectedModels, setSelectedModels] = useState([]);
  const [expandedModel, setExpandedModel] = useState(null);
  const [trainMode, setTrainMode] = useState('scratch'); // 'scratch' | 'pretrained'
  const [summary, setSummary] = useState(null);
  const { addNotification } = useNotifications();

  useEffect(() => {
    let mounted = true;
    const loadSummary = async () => {
      try {
        const data = await api.getSummary();
        if (mounted) {
          setSummary(data);
        }
      } catch (err) {
        if (mounted) {
          setSummary(null);
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

  const handleTrain = async () => {
    if (selectedModels.length === 0) {
      addNotification('Please select at least one model to train.', 'warning');
      return;
    }
    const names = selectedModels.map((id) => ALGORITHMS.find((a) => a.id === id)?.shortName).join(', ');
    addNotification(`Starting training for: ${names}`, 'info');
    try {
      const map = {
        rf: 'random_forest',
        svm: 'svm',
        knn: 'knn',
        lr: 'logreg',
        ada: 'adaboost',
        xgb: 'xgboost',
      };
      const selected = selectedModels.map((id) => map[id]).filter(Boolean);
      if (selected.length === 0) {
        addNotification('Selected models are not wired to backend training yet.', 'warning');
        return;
      }
      const result = await api.startTraining({
        selected_models: selected,
        run_task4: selected.includes('random_forest'),
      });
      addNotification(`Training job queued: ${result.job_id}`, 'success', 6000);
    } catch (err) {
      addNotification('Failed to start training job', 'error');
    }
  };

  const handleModelUpload = async (file) => {
    if (!file) return;
    addNotification('Uploading pretrained model...', 'info');
    try {
      const result = await api.uploadModel(file);
      addNotification(`Model uploaded: ${result.filename || file.name}`, 'success');
    } catch (err) {
      addNotification('Failed to upload model', 'error');
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            Model Selection
            <Tooltip text="Choose one or more ML algorithms to train. Select multiple to compare side by side." />
          </h1>
          <p className="text-gray-400 mt-1">Choose algorithms, compare documentation, and start training</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-400">
            {selectedModels.length} selected
          </span>
          <button
            onClick={handleTrain}
            className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-5 py-2.5 rounded-lg transition-colors flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            Train Selected
          </button>
        </div>
      </div>

      {/* Training Mode Toggle */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center gap-4">
        <Layers className="w-5 h-5 text-indigo-400 shrink-0" />
        <span className="text-sm text-gray-300 font-medium">Training Mode:</span>
        <div className="flex gap-2">
          <button
            onClick={() => setTrainMode('scratch')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              trainMode === 'scratch'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-gray-200'
            }`}
          >
            Train from Scratch
          </button>
          <button
            onClick={() => setTrainMode('pretrained')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              trainMode === 'pretrained'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-gray-200'
            }`}
          >
            <Upload className="w-4 h-4 inline mr-1.5" />
            Load Pre-trained
          </button>
        </div>
        {trainMode === 'pretrained' && (
          <div className="flex items-center gap-2 ml-auto">
            <input
              type="file"
              className="hidden"
              id="model-upload"
              accept=".pkl,.joblib,.h5"
              onChange={(e) => handleModelUpload(e.target.files?.[0])}
            />
            <label
              htmlFor="model-upload"
              className="bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-2 rounded-lg text-sm cursor-pointer border border-gray-700 transition-colors"
            >
              Upload Model (.pkl / .joblib)
            </label>
          </div>
        )}
      </div>

      {/* Algorithm Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {ALGORITHMS.map((algo) => {
          const isSelected = selectedModels.includes(algo.id);
          const isExpanded = expandedModel === algo.id;
          const result = summary?.models?.[algo.id];
          const f1 = result?.metrics?.f1;
          const duration = result?.durationSec != null ? `${Math.round(result.durationSec)}s` : null;

          return (
            <div
              key={algo.id}
              className={`bg-gray-900 border rounded-xl transition-all duration-300 ${
                isSelected ? 'border-indigo-500 shadow-lg shadow-indigo-500/10' : 'border-gray-800 hover:border-gray-700'
              }`}
            >
              {/* Card Header */}
              <div className="p-5">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <button onClick={() => toggleModel(algo.id)} className="mt-0.5">
                      {isSelected ? (
                        <CheckSquare className="w-5 h-5 text-indigo-400" />
                      ) : (
                        <Square className="w-5 h-5 text-gray-600 hover:text-gray-400" />
                      )}
                    </button>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{algo.name}</h3>
                      <span className="inline-block mt-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-500/20 text-indigo-300">
                        {algo.category}
                      </span>
                    </div>
                  </div>
                  {result && (
                    <div className="text-right">
                      <p className="text-sm font-mono text-emerald-400">F1: {f1 != null ? f1.toFixed(3) : 'n/a'}</p>
                      <p className="text-xs text-gray-500">{duration || 'n/a'}</p>
                    </div>
                  )}
                </div>
                <p className="text-sm text-gray-400 mt-3 ml-8">{algo.description}</p>

                {/* Expand button */}
                <button
                  onClick={() => setExpandedModel(isExpanded ? null : algo.id)}
                  className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 mt-3 ml-8"
                >
                  {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  {isExpanded ? 'Show Less' : 'Details & Hyperparameters'}
                </button>
              </div>

              {/* Expanded Details */}
              {isExpanded && (
                <div className="border-t border-gray-800 p-5 space-y-4">
                  {/* Pros / Cons */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-emerald-400 font-medium mb-2 flex items-center gap-1">
                        <ThumbsUp className="w-3.5 h-3.5" /> Pros
                      </p>
                      <ul className="space-y-1">
                        {algo.pros.map((pro) => (
                          <li key={pro} className="text-xs text-gray-400 flex items-start gap-1.5">
                            <span className="text-emerald-500 mt-0.5">•</span>
                            {pro}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-xs text-red-400 font-medium mb-2 flex items-center gap-1">
                        <ThumbsDown className="w-3.5 h-3.5" /> Cons
                      </p>
                      <ul className="space-y-1">
                        {algo.cons.map((con) => (
                          <li key={con} className="text-xs text-gray-400 flex items-start gap-1.5">
                            <span className="text-red-500 mt-0.5">•</span>
                            {con}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* Hyperparameters preview */}
                  <div>
                    <p className="text-xs text-gray-300 font-medium mb-2 flex items-center gap-1">
                      <Info className="w-3.5 h-3.5" /> Hyperparameters
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {algo.hyperparameters.map((hp) => (
                        <span
                          key={hp.name}
                          className="px-2 py-1 bg-gray-800 rounded text-xs text-gray-300 border border-gray-700"
                        >
                          <span className="text-indigo-400">{hp.name}</span>=
                          <span className="text-amber-300">{hp.default}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Comparison Hint */}
      {selectedModels.length >= 2 && (
        <div className="bg-indigo-500/5 border border-indigo-500/20 rounded-xl p-4 flex items-center gap-3">
          <Info className="w-5 h-5 text-indigo-400 shrink-0" />
          <p className="text-sm text-indigo-200">
            <strong>{selectedModels.length} models selected</strong> for comparison. After training,
            go to <a href="/results" className="underline hover:text-indigo-300">Results</a> to
            compare them side by side.
          </p>
        </div>
      )}
    </div>
  );
}
