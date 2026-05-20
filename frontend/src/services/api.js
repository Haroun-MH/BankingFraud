const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const isFormData = typeof FormData !== 'undefined' && options.body instanceof FormData;
  if (!isFormData && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers,
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  if (response.status === 204) {
    return null;
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }

  return response.text();
}

export const api = {
  getRuns: (limit = 50) => request(`/api/runs?limit=${limit}`),
  getSummary: () => request('/api/summary'),
  startTraining: (payload) =>
    request('/api/train', {
      method: 'POST',
      body: JSON.stringify(payload || {}),
    }),
  tuneModel: (payload) =>
    request('/api/tune', {
      method: 'POST',
      body: JSON.stringify(payload || {}),
    }),
  getJob: (jobId) => request(`/api/jobs/${jobId}`),
  listJobs: () => request('/api/jobs'),
  pauseJob: (jobId) => request(`/api/jobs/${jobId}/pause`, { method: 'POST' }),
  resumeJob: (jobId) => request(`/api/jobs/${jobId}/resume`, { method: 'POST' }),
  cancelJob: (jobId) => request(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),
  getSystemMetrics: () => request('/api/system'),
  getDatasetSummary: () => request('/api/dataset/summary'),
  getDatasetFeatures: () => request('/api/dataset/features'),
  getDatasetPreview: (limit = 10) => request(`/api/dataset/preview?limit=${limit}`),
  cleanDataset: (payload) =>
    request('/api/dataset/clean', {
      method: 'POST',
      body: JSON.stringify(payload || {}),
    }),
  resetDataset: () => request('/api/dataset/reset', { method: 'POST' }),
  uploadDataset: (file) => {
    const form = new FormData();
    form.append('file', file);
    return request('/api/dataset/upload', { method: 'POST', body: form });
  },
  uploadModel: (file) => {
    const form = new FormData();
    form.append('file', file);
    return request('/api/models/upload', { method: 'POST', body: form });
  },
  datasetExportUrl: (columns) => {
    const params = new URLSearchParams();
    if (columns && columns.length) {
      params.set('columns', columns.join(','));
    }
    const suffix = params.toString();
    return `${API_BASE_URL}/api/dataset/export${suffix ? `?${suffix}` : ''}`;
  },
  getActiveModel: () => request('/api/model/active'),
  setActiveModel: (runId) =>
    request('/api/model/active', {
      method: 'POST',
      body: JSON.stringify({ run_id: runId }),
    }),
  exportRunModelUrl: (runId) => `${API_BASE_URL}/api/runs/${runId}/export`,
  artifactUrl: (name) => `${API_BASE_URL}/api/artifacts/${name}`,

  // Prediction
  predict: (transaction) =>
    request('/api/predict', {
      method: 'POST',
      body: JSON.stringify(transaction),
    }),

  // MLflow Model Registry
  registerModel: (runId, modelName) =>
    request('/api/registry/register', {
      method: 'POST',
      body: JSON.stringify({ run_id: runId, model_name: modelName }),
    }),
  promoteModel: (payload = {}) =>
    request('/api/registry/promote', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getRegistryVersions: (modelName) => {
    const params = modelName ? `?model_name=${encodeURIComponent(modelName)}` : '';
    return request(`/api/registry/versions${params}`);
  },
};
