import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import NotificationToast from './components/NotificationToast';
import { NotificationProvider } from './context/NotificationContext';
import Dashboard from './pages/Dashboard';
import Models from './pages/Models';
import Hyperparameters from './pages/Hyperparameters';
import AutoML from './pages/AutoML';
import DataExplorer from './pages/DataExplorer';
import Results from './pages/Results';
import TrainingMonitor from './pages/TrainingMonitor';
import MLOps from './pages/MLOps';
import Predict from './pages/Predict';

function App() {
  return (
    <BrowserRouter>
      <NotificationProvider>
        <div className="flex min-h-screen bg-gray-950">
          <Sidebar />
          <main className="flex-1 overflow-x-hidden">
            <div className="max-w-7xl mx-auto px-6 py-8">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/models" element={<Models />} />
                <Route path="/hyperparameters" element={<Hyperparameters />} />
                <Route path="/automl" element={<AutoML />} />
                <Route path="/data" element={<DataExplorer />} />
                <Route path="/results" element={<Results />} />
                <Route path="/monitoring" element={<TrainingMonitor />} />
                <Route path="/mlops" element={<MLOps />} />
                <Route path="/predict" element={<Predict />} />
              </Routes>
            </div>
          </main>
          <NotificationToast />
        </div>
      </NotificationProvider>
    </BrowserRouter>
  );
}

export default App;
