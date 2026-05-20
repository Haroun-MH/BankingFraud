import { useNotifications } from '../context/NotificationContext';
import { X, CheckCircle, AlertTriangle, Info, AlertCircle } from 'lucide-react';

const iconMap = {
  success: CheckCircle,
  warning: AlertTriangle,
  error: AlertCircle,
  info: Info,
};

const colorMap = {
  success: 'border-emerald-500 bg-emerald-500/10 text-emerald-300',
  warning: 'border-amber-500 bg-amber-500/10 text-amber-300',
  error: 'border-red-500 bg-red-500/10 text-red-300',
  info: 'border-indigo-500 bg-indigo-500/10 text-indigo-300',
};

export default function NotificationToast() {
  const { notifications, removeNotification } = useNotifications();

  if (notifications.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-3 max-w-sm">
      {notifications.map((n) => {
        const Icon = iconMap[n.type] || Info;
        return (
          <div
            key={n.id}
            className={`flex items-start gap-3 px-4 py-3 rounded-lg border shadow-xl animate-slide-in ${colorMap[n.type] || colorMap.info}`}
            style={{ animation: 'slideIn 0.3s ease-out' }}
          >
            <Icon className="w-5 h-5 mt-0.5 shrink-0" />
            <p className="text-sm flex-1">{n.message}</p>
            <button onClick={() => removeNotification(n.id)} className="shrink-0 hover:opacity-70">
              <X className="w-4 h-4" />
            </button>
          </div>
        );
      })}
      <style>{`
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(100%); }
          to { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </div>
  );
}
