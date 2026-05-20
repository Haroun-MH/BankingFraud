import { Info } from 'lucide-react';

export default function Tooltip({ text, children }) {
  return (
    <span className="relative group inline-flex items-center">
      {children || <Info className="w-4 h-4 text-gray-500 hover:text-indigo-400 cursor-help transition-colors" />}
      <span className="invisible group-hover:visible absolute z-50 bg-gray-800 text-gray-200 text-xs rounded-lg px-3 py-2 border border-gray-700 shadow-xl whitespace-normal max-w-xs bottom-full left-1/2 -translate-x-1/2 mb-2 pointer-events-none">
        {text}
        <span className="absolute top-full left-1/2 -translate-x-1/2 -mt-px border-4 border-transparent border-t-gray-700" />
      </span>
    </span>
  );
}
