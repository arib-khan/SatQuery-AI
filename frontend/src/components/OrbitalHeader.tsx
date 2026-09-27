'use client';

import React, { useEffect, useState } from 'react';
import { SystemStatus } from '../types/satquery';

interface OrbitalHeaderProps {
  status?: SystemStatus | null;
}
const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || 'http://localhost:8000';

export const OrbitalHeader: React.FC<OrbitalHeaderProps> = ({ status: propStatus }) => {
  const [clock, setClock] = useState<string>('');
  const [liveStatus, setLiveStatus] = useState<SystemStatus | null>(propStatus || null);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const localTime = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
      const utcTime = now.toISOString().substring(11, 19) + ' UTC';
      setClock(`${localTime} LOCAL · ${utcTime}`);
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (propStatus) {
      setLiveStatus(propStatus);
      return;
    }
    const fetchStatus = () => {
      fetch(`${FASTAPI_BASE_URL}/api/status`)
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (data) setLiveStatus(data);
        })
        .catch(() => { });
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 12000);
    return () => clearInterval(interval);
  }, [propStatus]);

  return (
    <header className="top-navbar">
      <div className="brand-section">
        <div className="brand-icon-box">
          {/* Scientific Satellite Swath Aperture */}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2" />
            <polyline points="2 17 12 22 22 17" />
            <polyline points="2 12 12 17 22 12" />
          </svg>
        </div>
        <div>
          <div className="brand-title">
            AERO<span>LENS</span> AI
          </div>
          <div className="brand-subtitle">
            AUTONOMOUS ORBITAL EARTH OBSERVATION COCKPIT
          </div>
        </div>
      </div>

      <div className="telemetry-group">
        <div className="pill-badge active">
          <span className="pulse-dot" />
          <span>DOWNLINK: {liveStatus?.downlink_freq || '8.2 GHz (X-BAND)'}</span>
        </div>

        <div className="pill-badge">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent-amber)" strokeWidth="2.5">
            <circle cx="12" cy="12" r="10" />
            <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(45 12 12)" />
          </svg>
          <span>{liveStatus?.orbit || 'LEO 540KM · SSO 98.2°'}</span>
        </div>

        <div className="pill-badge">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <rect x="4" y="4" width="16" height="16" rx="2" />
            <line x1="9" y1="9" x2="15" y2="9" />
            <line x1="9" y1="15" x2="15" y2="15" />
          </svg>
          <span>{liveStatus?.adaptation || 'ORBITAL VLM INFERENCE CORE'}</span>
        </div>

        <div className="pill-badge">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          <span>{clock || 'SYNCING UTC...'}</span>
        </div>
      </div>
    </header>
  );
};
