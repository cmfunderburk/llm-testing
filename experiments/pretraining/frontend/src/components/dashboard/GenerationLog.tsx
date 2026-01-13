/**
 * Generation Log Component
 *
 * Displays generated text samples during training.
 */

import { useTraining } from '../../context/TrainingContext';

export function GenerationLog() {
  const { generations } = useTraining();

  if (generations.length === 0) {
    return (
      <div className="generation-log">
        <h3>Generated Samples</h3>
        <p className="empty-state">No samples generated yet. Start training to see generated text.</p>
      </div>
    );
  }

  return (
    <div className="generation-log">
      <h3>Generated Samples</h3>

      <div className="generations-list">
        {generations.slice().reverse().map((gen, idx) => (
          <div key={idx} className="generation-item">
            <div className="generation-header">
              <span className="generation-epoch">Epoch {gen.epoch}</span>
              <span className="generation-step">Step {gen.step}</span>
            </div>
            <pre className="generation-text">{gen.text}</pre>
          </div>
        ))}
      </div>
    </div>
  );
}

export default GenerationLog;
