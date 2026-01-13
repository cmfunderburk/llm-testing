/**
 * Sample Text Display Component
 *
 * Shows the most recent generated text sample from training.
 * Displays epoch-end generation samples to track model progress.
 */

import { useTraining } from '../../context/TrainingContext';

export function SampleTextDisplay() {
  const { generations, status } = useTraining();

  // Get the most recent generation
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
