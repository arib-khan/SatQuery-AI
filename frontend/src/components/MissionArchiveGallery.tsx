'use client';

import React, { useEffect, useState } from 'react';
import { BenchmarkMission } from '../types/satquery';

interface MissionArchiveGalleryProps {
  onSelectMission: (mission: BenchmarkMission) => void;
}

const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || 'http://localhost:8000';


const FALLBACK_MISSIONS: BenchmarkMission[] = [
  {
    id: 'vrsbench_grounding',
    title: 'VRSBench Text-Guided Grounding',
    category: 'Object Grounding',
    mission_tag: 'VRSBENCH-PASS',
    image_a: 'examples/scene_harbor.png',
    modality_a: 'Optical',
    image_b: null,
    modality_b: 'Auto',
    query: 'Locate and highlight all parked airplanes with a bounding box.',
  },
  {
    id: 'rsvqa_airport',
    title: 'RSVQA-LR Runway Traffic Pass',
    category: 'VQA Counting',
    mission_tag: 'RSVQA-PASS-102',
    image_a: 'examples/scene_airport.png',
    modality_a: 'Optical',
    image_b: null,
    modality_b: 'Auto',
    query: 'How many aircraft are present on the apron and tarmac? Detail their orientations.',
  },
  {
    id: 'bitemporal_flood',
    title: 'SpaceNet Flood Change-VQA',
    category: 'Bi-Temporal Change',
    mission_tag: 'DISASTER-CD-88',
    image_a: 'disaster_examples/flood_before.tiff',
    modality_a: 'Optical',
    image_b: 'disaster_examples/flood_after.tiff',
    modality_b: 'Optical',
    query: 'What structural and land-cover changes occurred between these two observation dates?',
  },
  {
    id: 'optical_sar_fusion',
    title: 'Sentinel-1 SAR / Sentinel-2 Optical Fusion',
    category: 'Cross-Modal Fusion',
    mission_tag: 'SENTINEL-XFUSE',
    image_a: 'examples/scene_industrial.png',
    modality_a: 'Optical',
    image_b: 'examples/scene_agricultural.png',
    modality_b: 'SAR',
    query: 'Use both the optical and SAR images together to identify water boundaries and built-up areas beneath clouds.',
  },
  {
    id: 'insat_cyclone',
    title: 'INSAT-3DS Cyclone Rapid Pass',
    category: 'Meteorological VQA',
    mission_tag: 'INSAT-RAPID-PASS',
    image_a: 'examples/scene_urban.png',
    modality_a: 'Optical',
    image_b: null,
    modality_b: 'Auto',
    query: 'Execute an exhaustive, high-depth scientific intelligence evaluation of this satellite imagery: synthesize platform telemetry, radiometric channel physics, cloud microphysics, terrestrial geomorphology, and tactical hazard advisories.',
  },
];

export const MissionArchiveGallery: React.FC<MissionArchiveGalleryProps> = ({ onSelectMission }) => {
  const [missions, setMissions] = useState<BenchmarkMission[]>(FALLBACK_MISSIONS);

  useEffect(() => {
    fetch(`${FASTAPI_BASE_URL}/api/examples`)
      .then((res) => {
        if (res.ok) return res.json();
        throw new Error('Fallback to local');
      })
      .then((data: any) => {
        const list = Array.isArray(data) ? data : data?.examples;
        if (list && list.length > 0) {
          setMissions(list);
        }
      })
      .catch(() => { });
  }, []);

  return (
    <div className="mission-archive-section">
      <div className="panel-header">
        <div className="panel-title">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
          <span>VERIFIED BENCHMARK MISSIONS // PRESET PASSES</span>
        </div>
        <span className="panel-title-tag">1-CLICK STAGING</span>
      </div>

      <div className="mission-cards-slider">
        {missions.map((m) => (
          <div
            key={m.id}
            className="mission-tile"
            onClick={() => onSelectMission(m)}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="mission-tile-tag">{m.mission_tag}</span>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '9px',
                    padding: '2px 5px',
                    borderRadius: '4px',
                    background: 'var(--accent-emerald-subtle)',
                    color: 'var(--accent-emerald)',
                    border: '1px solid var(--accent-emerald-border)',
                  }}
                >
                  {m.category}
                </span>
              </div>

              <div className="mission-tile-title">{m.title}</div>
              <div className="mission-tile-query">
                &ldquo;{m.query.length > 70 ? m.query.slice(0, 70) + '...' : m.query}&rdquo;
              </div>
            </div>

            <div className="mission-tile-footer">
              <span>STAGE SENSORS & QUERY</span>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
