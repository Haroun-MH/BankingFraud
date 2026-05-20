import { useEffect, useMemo, useState } from 'react';
import { api } from '../services/api';
import { useNotifications } from '../context/NotificationContext';
import Tooltip from '../components/Tooltip';
import {
  Database,
  Upload,
  Filter,
  Trash2,
  Eye,
  Download,
  AlertTriangle,
  CheckCircle,
  Search,
  Columns,
} from 'lucide-react';

export default function DataExplorer() {
  const [summary, setSummary] = useState(null);
  const [features, setFeatures] = useState([]);
  const [preview, setPreview] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [showUpload, setShowUpload] = useState(false);
  const [cleaningActions, setCleaningActions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const { addNotification } = useNotifications();

  const loadDataset = async () => {
    setIsLoading(true);
    setLoadError(null);
    try {
      const [summaryData, featureData, previewData] = await Promise.all([
        api.getDatasetSummary(),
        api.getDatasetFeatures(),
        api.getDatasetPreview(10),
      ]);
      setSummary(summaryData);
      setFeatures(Array.isArray(featureData) ? featureData : []);
      setPreview(Array.isArray(previewData) ? previewData : []);
      if (Array.isArray(featureData) && featureData.length > 0) {
        setSelectedColumns(featureData.map((f) => f.name));
      }
    } catch (err) {
      setLoadError('Unable to load dataset details');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDataset();
  }, []);

  const toggleColumn = (name) => {
    setSelectedColumns((prev) =>
      prev.includes(name) ? prev.filter((c) => c !== name) : [...prev, name]
    );
  };

  const filteredFeatures = useMemo(
    () =>
      features.filter((f) =>
        f.name.toLowerCase().includes(searchQuery.toLowerCase())
      ),
    [features, searchQuery]
  );

  const previewColumns = useMemo(() => {
    if (preview.length === 0) return [];
    const columns = Object.keys(preview[0]);
    const selected = new Set(selectedColumns);
    return columns.filter((col) => col === 'isFraud' || selected.has(col));
  }, [preview, selectedColumns]);

  const handleDropNulls = async (colName) => {
    try {
      await api.cleanDataset({ action: 'drop_nulls', column: colName });
      setCleaningActions((prev) => [...prev, `Drop nulls in ${colName}`]);
      await loadDataset();
      addNotification(`Null values removed from ${colName}`, 'success');
    } catch (err) {
      addNotification('Failed to drop nulls', 'error');
    }
  };

  const handleFillNulls = async (colName) => {
    try {
      await api.cleanDataset({ action: 'fill_nulls', column: colName });
      setCleaningActions((prev) => [...prev, `Fill nulls in ${colName}`]);
      await loadDataset();
      addNotification(`Null values in ${colName} filled`, 'success');
    } catch (err) {
      addNotification('Failed to fill nulls', 'error');
    }
  };

  const handleExportCSV = () => {
    const url = api.datasetExportUrl(selectedColumns);
    window.open(url, '_blank', 'noreferrer');
    addNotification('Dataset export started', 'success');
  };

  const handleUpload = async (file) => {
    if (!file) return;
    setUploading(true);
    try {
      await api.uploadDataset(file);
      setCleaningActions([]);
      await loadDataset();
      addNotification('Dataset uploaded successfully', 'success');
    } catch (err) {
      addNotification('Failed to upload dataset', 'error');
    } finally {
      setUploading(false);
    }
  };

  const handleReset = async () => {
    try {
      await api.resetDataset();
      setCleaningActions([]);
      await loadDataset();
      addNotification('Dataset reset to base version', 'info');
    } catch (err) {
      addNotification('Failed to reset dataset', 'error');
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Database className="w-8 h-8 text-cyan-400" />
            Data Explorer
            <Tooltip text="Preview, filter, and clean your dataset before training. Upload new data or select subsets." />
          </h1>
          <p className="text-gray-400 mt-1">Inspect, filter, and prepare your data</p>
          {isLoading && <p className="text-xs text-gray-500 mt-1">Loading dataset...</p>}
          {loadError && <p className="text-xs text-amber-400 mt-1">{loadError}</p>}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowUpload(!showUpload)}
            className="bg-gray-800 hover:bg-gray-700 text-gray-200 font-medium px-4 py-2.5 rounded-lg border border-gray-700 flex items-center gap-2 transition-colors"
          >
            <Upload className="w-4 h-4" />
            Upload Data
          </button>
          <button
            onClick={handleExportCSV}
            className="bg-gray-800 hover:bg-gray-700 text-gray-200 font-medium px-4 py-2.5 rounded-lg border border-gray-700 flex items-center gap-2 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Upload Panel */}
      {showUpload && (
        <div className="bg-gray-900 border border-dashed border-indigo-500/50 rounded-xl p-8 text-center">
          <Upload className="w-12 h-12 text-indigo-400 mx-auto mb-4" />
          <p className="text-white font-medium mb-1">Drag & drop your dataset here</p>
          <p className="text-sm text-gray-400 mb-4">Supports CSV, Excel, and Parquet formats</p>
          <label className="cursor-pointer inline-block bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-5 py-2.5 rounded-lg transition-colors">
            <input
              type="file"
              className="hidden"
              accept=".csv,.xlsx,.parquet"
              onChange={(e) => handleUpload(e.target.files?.[0])}
              disabled={uploading}
            />
            {uploading ? 'Uploading...' : 'Browse Files'}
          </label>
          <button
            onClick={() => setShowUpload(false)}
            className="block mx-auto mt-3 text-sm text-gray-500 hover:text-gray-300"
          >
            Cancel
          </button>
        </div>
      )}

      {/* Dataset Summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Dataset', value: summary?.name || 'n/a', color: 'text-cyan-400' },
          { label: 'Rows', value: summary?.rows != null ? summary.rows.toLocaleString() : 'n/a', color: 'text-indigo-400' },
          { label: 'Columns', value: summary?.columns != null ? summary.columns.toString() : 'n/a', color: 'text-amber-400' },
          { label: 'Fraud Rate', value: summary?.fraud_rate != null ? `${summary.fraud_rate}%` : 'n/a', color: 'text-red-400' },
        ].map((stat) => (
          <div key={stat.label} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-1">{stat.label}</p>
            <p className={`text-lg font-bold ${stat.color}`}>{stat.value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Column Selector */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Columns className="w-5 h-5 text-indigo-400" />
              Features
            </h2>
            <span className="text-xs text-gray-500">{selectedColumns.length} / {features.length}</span>
          </div>

          <div className="relative mb-3">
            <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search columns..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-9 pr-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          <div className="flex gap-2 mb-3">
            <button
              onClick={() => setSelectedColumns(features.map((f) => f.name))}
              className="text-xs text-indigo-400 hover:text-indigo-300"
            >
              Select All
            </button>
            <span className="text-gray-700">|</span>
            <button
              onClick={() => setSelectedColumns([])}
              className="text-xs text-indigo-400 hover:text-indigo-300"
            >
              Deselect All
            </button>
          </div>

          <div className="space-y-1 max-h-96 overflow-y-auto">
            {filteredFeatures.map((feat) => (
              <label
                key={feat.name}
                className={`flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer text-sm transition-colors ${
                  selectedColumns.includes(feat.name)
                    ? 'bg-indigo-500/10 text-white'
                    : 'text-gray-500 hover:bg-gray-800'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedColumns.includes(feat.name)}
                  onChange={() => toggleColumn(feat.name)}
                  className="accent-indigo-500"
                />
                <span className="flex-1">{feat.name}</span>
                <span className="text-xs text-gray-600">{feat.type}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Column Details & Cleaning */}
        <div className="lg:col-span-2 space-y-6">
          {/* Feature Details Table */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="p-5 border-b border-gray-800">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Filter className="w-5 h-5 text-amber-400" />
                Column Statistics
              </h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-800/50">
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Column</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Type</th>
                    <th className="text-right px-4 py-3 text-gray-400 font-medium">Nulls</th>
                    <th className="text-right px-4 py-3 text-gray-400 font-medium">Unique</th>
                    <th className="text-center px-4 py-3 text-gray-400 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredFeatures
                    .filter((f) => selectedColumns.includes(f.name))
                    .map((feat) => (
                      <tr key={feat.name} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/30">
                        <td className="px-4 py-3 font-medium text-white">{feat.name}</td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-0.5 bg-gray-800 rounded text-xs text-gray-300">
                            {feat.type}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right">
                          {feat.nulls > 0 ? (
                            <span className="text-amber-400 flex items-center justify-end gap-1">
                              <AlertTriangle className="w-3.5 h-3.5" />
                              {feat.nulls.toLocaleString()}
                            </span>
                          ) : (
                            <span className="text-emerald-400 flex items-center justify-end gap-1">
                              <CheckCircle className="w-3.5 h-3.5" />0
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-300">{feat.unique.toLocaleString()}</td>
                        <td className="px-4 py-3 text-center">
                          {feat.nulls > 0 && (
                            <div className="flex items-center justify-center gap-1.5">
                              <button
                                onClick={() => handleDropNulls(feat.name)}
                                className="text-xs px-2 py-1 bg-red-500/10 text-red-400 rounded hover:bg-red-500/20 transition-colors"
                              >
                                Drop
                              </button>
                              <button
                                onClick={() => handleFillNulls(feat.name)}
                                className="text-xs px-2 py-1 bg-indigo-500/10 text-indigo-400 rounded hover:bg-indigo-500/20 transition-colors"
                              >
                                Fill
                              </button>
                            </div>
                          )}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Data Preview */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="p-5 border-b border-gray-800 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Eye className="w-5 h-5 text-cyan-400" />
                Data Preview
              </h2>
              <span className="text-xs text-gray-500">Showing first 10 rows</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-800/50">
                    {previewColumns.map((col) => (
                      <th key={col} className="text-left px-4 py-3 text-gray-400 font-medium whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, idx) => (
                    <tr key={idx} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/30">
                      {previewColumns.map((col) => (
                        <td
                          key={col}
                          className={`px-4 py-2.5 whitespace-nowrap ${
                            col === 'isFraud' && row[col] === 1
                              ? 'text-red-400 font-semibold'
                              : 'text-gray-300'
                          }`}
                        >
                          {row[col] != null ? row[col].toString() : 'n/a'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Cleaning Log */}
          {cleaningActions.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Trash2 className="w-4 h-4 text-amber-400" />
                  Cleaning Actions Applied
                </h2>
                <button
                  onClick={handleReset}
                  className="text-xs text-gray-400 hover:text-gray-200"
                >
                  Undo All
                </button>
              </div>
              <div className="space-y-1">
                {cleaningActions.map((action, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-sm text-gray-400">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    {action}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
