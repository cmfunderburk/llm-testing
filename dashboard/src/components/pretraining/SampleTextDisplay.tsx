/**
 * Sample Text Display Component
 *
 * Shows a scrollable history of generated samples during training.
 * Generations are produced every 2000 steps and at epoch end.
 */

import { useEffect, useRef } from 'react';
import { useTraining } from '../../context/TrainingContext';

export function SampleTextDisplay() {
  const { generations, status } = useTraining();
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest generation
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [generations]);

  const isEmpty = generations.length === 0;

  return (
    <div className="sample-text-display">
      <h3>
        Generated Samples
        {!isEmpty && <span className="generation-count">({generations.length})</span>}
      </h3>

      {isEmpty ? (
        <div className="sample-empty">
          <p>
            {status.state === 'idle'
              ? 'Start training to see generated samples.'
              : status.state === 'loading'
              ? 'Loading corpus...'
              : 'Samples generated every 2000 steps...'}
          </p>
        </div>
      ) : (
        <div className="generations-list" ref={scrollRef}>
          {generations.map((gen, idx) => (
            <div key={`${gen.epoch}-${gen.step}`} className="generation-item">
              <div className="generation-header">
                <span className="generation-number">#{idx + 1}</span>
                <span className="generation-meta">
                  Epoch {gen.epoch} · Step {gen.step.toLocaleString()}
                </span>
              </div>
              <pre className="generation-text">{gen.text}</pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default SampleTextDisplay;
