import { ShieldAlert } from 'lucide-react';
import PredictForm from '../components/PredictForm';
import Tooltip from '../components/Tooltip';

export default function Predict() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <ShieldAlert className="w-8 h-8 text-rose-400" />
          Fraud Prediction
          <Tooltip text="Enter transaction details to get a real-time fraud probability from the active model." />
        </h1>
        <p className="text-gray-400 mt-1">
          Submit a transaction to the active model and get an instant fraud score
        </p>
      </div>

      {/* Form card */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <PredictForm />
      </div>

      {/* Info note */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 text-sm text-gray-400">
        <p className="font-medium text-gray-300 mb-1">How it works</p>
        <p>
          The form sends the transaction fields to <code className="text-indigo-400">/api/predict</code>.
          The backend preprocesses the input, runs it through the active MLflow model, and returns a
          fraud probability. Set the active model from the{' '}
          <a href="/mlops" className="text-indigo-400 hover:text-indigo-300 underline">
            MLOps
          </a>{' '}
          page before predicting.
        </p>
      </div>
    </div>
  );
}
