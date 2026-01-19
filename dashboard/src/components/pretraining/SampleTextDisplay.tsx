/**
 * Sample Text Display Component
 */

import { useTraining } from '../../context/TrainingContext';

export function SampleTextDisplay() {
  const { generations, status } = useTraining();

  const latestGen = generations[generations.length - 1];

  if (!latestGen) {
    return (
      <div className="sample-text-display">
        <h3>Latest Generation</h3>
        <div className="sample-empty">
          <p>
            {status.state === 'idle'
              ? 'Start training to see generated samples.'
              : 'Waiting for first epoch to complete...'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="sample-text-display">
      <h3>Latest Generation</h3>
      <div className="sample-meta">
        <span>Epoch {latestGen.epoch}</span>
        <span>Step {latestGen.step}</span>
      </div>
      <div className="sample-content">
        <pre>{latestGen.text}</pre>
      </div>
    </div>
  );
}

export default SampleTextDisplay;
