import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Brain,
  SlidersHorizontal,
  Wand2,
  Database,
  BarChart3,
  Activity,
  GitBranch,
  ShieldAlert,
  HelpCircle,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useState } from 'react';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/models', icon: Brain, label: 'Models' },
  { to: '/hyperparameters', icon: SlidersHorizontal, label: 'Parameters' },
  { to: '/automl', icon: Wand2, label: 'AutoML' },
  { to: '/data', icon: Database, label: 'Data Explorer' },
  { to: '/results', icon: BarChart3, label: 'Results' },
  { to: '/monitoring', icon: Activity, label: 'Training Monitor' },
  { to: '/mlops', icon: GitBranch, label: 'MLOps' },
  { to: '/predict', icon: ShieldAlert, label: 'Predict' },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`${
        collapsed ? 'w-16' : 'w-60'
      } bg-gray-900 border-r border-gray-800 h-screen flex flex-col transition-all duration-300 shrink-0 sticky top-0`}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-gray-800">
        <ShieldCheck className="w-7 h-7 text-indigo-400 shrink-0" />
        {!collapsed && <span className="font-bold text-lg text-white tracking-tight">FraudGuard</span>}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 space-y-1 px-2 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200 border border-transparent'
              }`
            }
          >
            <item.icon className="w-5 h-5 shrink-0" />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Tutorial trigger */}
      {!collapsed && (
        <div className="px-3 pb-2">
          <button className="flex items-center gap-2 px-3 py-2 w-full text-sm text-gray-500 hover:text-indigo-400 rounded-lg hover:bg-gray-800 transition-colors">
            <HelpCircle className="w-4 h-4" />
            Tutorial Guide
          </button>
        </div>
      )}

      {/* Collapse toggle */}
      <div className="border-t border-gray-800 p-2">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center py-2 text-gray-500 hover:text-gray-300 rounded-lg hover:bg-gray-800 transition-colors"
        >
          {collapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
        </button>
      </div>
    </aside>
  );
}
