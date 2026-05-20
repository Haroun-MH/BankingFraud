import { useState } from 'react';
import { api } from '../services/api';
import { useNotifications } from '../context/NotificationContext';
import { ShieldAlert, ShieldCheck, Loader2, AlertCircle } from 'lucide-react';

const FIELDS = [
  { name: 'TransactionAmt', label: 'Transaction Amount ($)', type: 'number', placeholder: '100.00', step: '0.01' },
  {
    name: 'ProductCD',
    label: 'Product Code',
    type: 'select',
    options: ['W', 'H', 'C', 'S', 'R'],
  },
  { name: 'card1', label: 'Card 1', type: 'number', placeholder: '1234', step: '1' },
  { name: 'card2', label: 'Card 2', type: 'number', placeholder: '321', step: '1' },
  { name: 'card3', label: 'Card 3', type: 'number', placeholder: '150', step: '1' },
  {
    name: 'card4',
    label: 'Card Network',
    type: 'select',
    options: ['visa', 'mastercard', 'discover', 'american express'],
  },
  { name: 'card5', label: 'Card 5', type: 'number', placeholder: '226', step: '1' },
  {
    name: 'card6',
    label: 'Card Type',
    type: 'select',
    options: ['debit', 'credit', 'debit or credit', 'charge card'],
  },
  { name: 'addr1', label: 'Billing Address (addr1)', type: 'number', placeholder: '299', step: '1' },
  { name: 'addr2', label: 'Billing Country (addr2)', type: 'number', placeholder: '87', step: '1' },
  { name: 'dist1', label: 'Distance 1', type: 'number', placeholder: '0', step: '1' },
  {
    name: 'P_emaildomain',
    label: 'Purchaser Email Domain',
    type: 'select',
    options: ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'anonymous.com', 'other'],
  },
  {
    name: 'R_emaildomain',
    label: 'Recipient Email Domain',
    type: 'select',
    options: ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'anonymous.com', 'other'],
  },
];

const DEFAULT_VALUES = Object.fromEntries(
  FIELDS.map((f) => [f.name, f.type === 'select' ? f.options[0] : ''])
);

export default function PredictForm() {
  const [values, setValues] = useState(DEFAULT_VALUES);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const { addNotification } = useNotifications();

  const handleChange = (name, value) => {
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResult(null);
    setError(null);

    // Convert numeric strings to numbers
    const payload = {};
    for (const field of FIELDS) {
      const raw = values[field.name];
      if (field.type === 'number') {
        const num = parseFloat(raw);
        payload[field.name] = isNaN(num) ? 0 : num;
      } else {
        payload[field.name] = raw;
      }
    }

    try {
      const data = await api.predict(payload);
      setResult(data);
      addNotification(
        data.is_fraud ? 'Fraud detected!' : 'Transaction appears legitimate',
        data.is_fraud ? 'error' : 'success',
        5000
      );
    } catch (err) {
      const msg = err.message || 'Prediction failed';
      setError(msg);
      addNotification(msg, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setValues(DEFAULT_VALUES);
    setResult(null);
    setError(null);
  };

  const fraudPct =
    result?.fraud_probability != null ? Math.round(result.fraud_probability * 100) : null;

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Fields grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FIELDS.map((field) => (
            <div key={field.name}>
              <label className="block text-sm font-medium text-gray-300 mb-1.5">
                {field.label}
              </label>
              {field.type === 'select' ? (
                <select
                  value={values[field.name]}
                  onChange={(e) => handleChange(field.name, e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {field.options.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  step={field.step}
                  placeholder={field.placeholder}
                  value={values[field.name]}
                  onChange={(e) => handleChange(field.name, e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              )}
            </div>
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3 pt-2">
          <button
            type="submit"
            disabled={isLoading}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold px-6 py-2.5 rounded-lg flex items-center gap-2 transition-colors"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Predicting...
              </>
            ) : (
              <>
                <ShieldAlert className="w-4 h-4" />
                Predict Fraud
              </>
            )}
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="bg-gray-800 hover:bg-gray-700 text-gray-300 font-medium px-4 py-2.5 rounded-lg border border-gray-700 transition-colors"
          >
            Reset
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div
          className={`rounded-xl border p-6 flex flex-col sm:flex-row items-center gap-6 ${
            result.is_fraud
              ? 'bg-red-500/10 border-red-500/30'
              : 'bg-emerald-500/10 border-emerald-500/30'
          }`}
        >
          {result.is_fraud ? (
            <ShieldAlert className="w-14 h-14 text-red-400 shrink-0" />
          ) : (
            <ShieldCheck className="w-14 h-14 text-emerald-400 shrink-0" />
          )}

          <div className="flex-1 text-center sm:text-left">
            <p
              className={`text-2xl font-bold ${
                result.is_fraud ? 'text-red-300' : 'text-emerald-300'
              }`}
            >
              {result.is_fraud ? 'Fraud Detected' : 'Legitimate Transaction'}
            </p>
            {fraudPct !== null && (
              <p className="text-gray-400 mt-1 text-sm">
                Fraud probability:{' '}
                <span
                  className={`font-mono font-semibold ${
                    result.is_fraud ? 'text-red-300' : 'text-emerald-300'
                  }`}
                >
                  {fraudPct}%
                </span>
              </p>
            )}
          </div>

          {/* Probability bar */}
          {fraudPct !== null && (
            <div className="w-full sm:w-48">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>0%</span>
                <span>100%</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all duration-700 ${
                    result.is_fraud ? 'bg-red-500' : 'bg-emerald-500'
                  }`}
                  style={{ width: `${fraudPct}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
