/**
 * Sidebar Component
 *
 * Navigation sidebar with links to different sections of the app.
 */


interface NavItem {
  id: string;
  label: string;
  icon: string;
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'training', label: 'Training', icon: '🎯' },
  { id: 'generate', label: 'Generate', icon: '✨' },
  { id: 'analysis', label: 'Analysis', icon: '🔬' },
];

interface SidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

export function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  return (
    <nav className="sidebar">
      <ul className="nav-list">
        {navItems.map((item) => (
          <li key={item.id}>
            <button
              className={`nav-item ${activeSection === item.id ? 'active' : ''}`}
              onClick={() => onSectionChange(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          </li>
        ))}
      </ul>

      <div className="sidebar-footer">
        <div className="version-info">v0.1.0</div>
      </div>
    </nav>
  );
}

export default Sidebar;
