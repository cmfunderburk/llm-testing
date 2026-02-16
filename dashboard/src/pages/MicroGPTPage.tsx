/**
 * MicroGPT Page
 *
 * Canonical educational entrypoint driven by misc/microgpt.py.
 */

import { useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { Card, LoadingSpinner } from '../components/shared';
import type { MicroGPTArtifactResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface GuidedSection {
  title: string;
  startLine: number;
  endLine: number;
  systemView: string;
  mathView: string;
}

const GUIDED_SECTIONS: GuidedSection[] = [
  {
    title: 'Data Bootstrap and Character Vocabulary',
    startLine: 14,
    endLine: 27,
    systemView: 'Loads a corpus, shuffles document order, and builds a character-level vocabulary with one extra BOS token.',
    mathView: 'Defines a finite alphabet Sigma from dataset support. Tokenization is a bijection between Sigma U {BOS} and {0,...,|Sigma|}.',
  },
  {
    title: 'Scalar Autograd Engine',
    startLine: 29,
    endLine: 73,
    systemView: 'Implements a dynamic scalar computation graph and reverse-mode automatic differentiation.',
    mathView: 'For each node v = f(u), stores local Jacobian pieces d v / d u_i and applies chain rule in reverse topological order.',
  },
  {
    title: 'Parameter Initialization and State Dict',
    startLine: 74,
    endLine: 90,
    systemView: 'Creates all trainable matrices (embeddings, attention projections, MLP projections, LM head) and flattens params.',
    mathView: 'Samples theta from N(0, sigma^2) per scalar, yielding a random initialization over a non-convex objective.',
  },
  {
    title: 'Linear Algebra Primitives',
    startLine: 92,
    endLine: 107,
    systemView: 'Defines linear map, numerically-stable softmax, and RMSNorm used by the transformer block.',
    mathView: 'Computes softmax(z)_i = exp(z_i - m)/sum_j exp(z_j - m), where m = max_j z_j for stability.',
  },
  {
    title: 'Single-Token GPT Forward Pass',
    startLine: 108,
    endLine: 145,
    systemView: 'Runs token+position embedding, causal self-attention with KV cache, MLP, residuals, and outputs next-token logits.',
    mathView: 'Implements h_{t+1} = Block(h_t, K_{<=t}, V_{<=t}) and logits = W_out h_t, with attention scaling by 1/sqrt(d_head).',
  },
  {
    title: 'Adam State and Hyperparameters',
    startLine: 146,
    endLine: 150,
    systemView: 'Configures optimizer constants and first/second moment buffers.',
    mathView: 'Tracks exponential moving averages m_t and v_t of gradients and squared gradients for adaptive preconditioning.',
  },
  {
    title: 'Training Loop and Cross-Entropy Objective',
    startLine: 151,
    endLine: 185,
    systemView: 'Builds sequence loss from next-token predictions, backpropagates, applies Adam with linear LR decay, and logs progress.',
    mathView: 'Minimizes mean negative log-likelihood: L = -(1/n) sum_t log p_theta(x_{t+1}|x_{<=t}).',
  },
  {
    title: 'Autoregressive Sampling',
    startLine: 186,
    endLine: 200,
    systemView: 'Generates new sequences by repeatedly sampling from temperature-scaled predictive distribution until BOS stop.',
    mathView: 'Samples x_t ~ softmax(logits_t / T), where T controls entropy and sharpness of the categorical distribution.',
  },
];

function shortHash(sha256: string): string {
  if (sha256.length < 16) return sha256;
  return `${sha256.slice(0, 12)}...${sha256.slice(-8)}`;
}

function rangesOverlap(aStart: number, aEnd: number, bStart: number, bEnd: number): boolean {
  return Math.max(aStart, bStart) <= Math.min(aEnd, bEnd);
}

function extractGuideSectionMarkdown(markdown: string, startLine: number, endLine: number): string {
  const lines = markdown.split('\n');
  const headingMatches: Array<{ idx: number; start: number; end: number }> = [];
  const headingRegex = /^###\s+Lines\s+(\d+)-(\d+):/;

  lines.forEach((line, idx) => {
    const match = headingRegex.exec(line.trim());
    if (match) {
      headingMatches.push({
        idx,
        start: Number.parseInt(match[1], 10),
        end: Number.parseInt(match[2], 10),
      });
    }
  });

  if (headingMatches.length === 0) {
    return markdown;
  }

  const selectedBlocks: string[] = [];
  for (let i = 0; i < headingMatches.length; i += 1) {
    const current = headingMatches[i];
    const nextIdx = i + 1 < headingMatches.length ? headingMatches[i + 1].idx : lines.length;
    if (rangesOverlap(startLine, endLine, current.start, current.end)) {
      selectedBlocks.push(lines.slice(current.idx, nextIdx).join('\n').trim());
    }
  }

  if (selectedBlocks.length === 0) {
    return '_No matching guide section found for this line range._';
  }

  return selectedBlocks.join('\n\n');
}

function normalizeMathDelimiters(markdown: string): string {
  let normalized = markdown.replace(
    /\\\[([\s\S]+?)\\\]/g,
    (_match, expr: string) => `\n$$\n${expr.trim()}\n$$\n`,
  );
  normalized = normalized.replace(
    /\\\(([\s\S]+?)\\\)/g,
    (_match, expr: string) => `$${expr.trim()}$`,
  );
  return normalized;
}

export function MicroGPTPage() {
  const [artifact, setArtifact] = useState<MicroGPTArtifactResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSectionIdx, setActiveSectionIdx] = useState(0);
  const [showFullGuide, setShowFullGuide] = useState(false);
  const [showFullSource, setShowFullSource] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const fetchArtifact = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/api/education/microgpt`);
        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || 'Failed to load MicroGPT educational artifact');
        }
        const data: MicroGPTArtifactResponse = await response.json();
        if (!cancelled) setArtifact(data);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    fetchArtifact();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeSection = GUIDED_SECTIONS[activeSectionIdx];
  const sourceLines = useMemo(
    () => artifact?.source.split('\n') ?? [],
    [artifact],
  );
  const visibleLines = useMemo(
    () => sourceLines.slice(activeSection.startLine - 1, activeSection.endLine),
    [activeSection.endLine, activeSection.startLine, sourceLines],
  );
  const sectionGuideMarkdown = useMemo(
    () => (artifact
      ? extractGuideSectionMarkdown(artifact.docs_markdown, activeSection.startLine, activeSection.endLine)
      : ''),
    [activeSection.endLine, activeSection.startLine, artifact],
  );
  const renderedSectionGuideMarkdown = useMemo(
    () => normalizeMathDelimiters(sectionGuideMarkdown),
    [sectionGuideMarkdown],
  );
  const renderedFullGuideMarkdown = useMemo(
    () => (artifact ? normalizeMathDelimiters(artifact.docs_markdown) : ''),
    [artifact],
  );

  return (
    <div className="microgpt-page">
      <div className="page-header">
        <h2>MicroGPT Canonical Entry</h2>
        <p className="page-description">
          Read-through mode: source code and rendered guide side-by-side, aligned by line ranges from
          <code> misc/microgpt.py</code>.
        </p>
      </div>

      {isLoading && (
        <Card>
          <LoadingSpinner message="Loading canonical MicroGPT source and educational guide..." />
        </Card>
      )}

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {artifact && !isLoading && (
        <>
          <Card title="Reading Workspace">
            <div className="microgpt-meta-grid">
              <div className="microgpt-meta-item">
                <span className="microgpt-meta-label">Source Path</span>
                <code>{artifact.source_path}</code>
              </div>
              <div className="microgpt-meta-item">
                <span className="microgpt-meta-label">Documentation Path</span>
                <code>{artifact.docs_path}</code>
              </div>
              <div className="microgpt-meta-item">
                <span className="microgpt-meta-label">Source Hash (SHA-256)</span>
                <code>{shortHash(artifact.source_sha256)}</code>
              </div>
              <div className="microgpt-meta-item">
                <span className="microgpt-meta-label">Line Counts</span>
                <span>{artifact.source_line_count} source lines, {artifact.docs_line_count} doc lines</span>
              </div>
            </div>

            <div className="microgpt-section-bar">
              <span className="microgpt-meta-label">Section Focus</span>
              <div className="button-group">
                {GUIDED_SECTIONS.map((section, idx) => (
                  <button
                    key={section.title}
                    className={`btn-select ${idx === activeSectionIdx ? 'selected' : ''}`}
                    onClick={() => setActiveSectionIdx(idx)}
                  >
                    {section.startLine}-{section.endLine}
                  </button>
                ))}
              </div>
            </div>

            <div className="microgpt-section-summary">
              <h3>{activeSection.title}</h3>
              <p><strong>Systems:</strong> {activeSection.systemView}</p>
              <p><strong>Math:</strong> {activeSection.mathView}</p>
            </div>
          </Card>

          <div className="microgpt-grid">
            <Card title={`Code (${activeSection.startLine}-${activeSection.endLine})`}>
              <div className="microgpt-code-viewer">
                {visibleLines.map((line, idx) => {
                  const lineNo = activeSection.startLine + idx;
                  return (
                    <div key={lineNo} className="microgpt-code-row">
                      <span className="microgpt-line-number">{lineNo.toString().padStart(3, ' ')}</span>
                      <span className="microgpt-code-text">{line || ' '}</span>
                    </div>
                  );
                })}
              </div>
            </Card>

            <Card title={`Guide (${activeSection.startLine}-${activeSection.endLine})`}>
              <div className="microgpt-markdown">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {renderedSectionGuideMarkdown}
                </ReactMarkdown>
              </div>
            </Card>
          </div>

          <Card title="Reference Views">
            <div className="microgpt-reference-actions">
              <button
                className="btn btn-secondary"
                onClick={() => setShowFullGuide((v) => !v)}
              >
                {showFullGuide ? 'Hide Full Guide' : 'Show Full Guide'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => setShowFullSource((v) => !v)}
              >
                {showFullSource ? 'Hide Full Source' : 'Show Full Source'}
              </button>
            </div>

            {showFullGuide && (
              <div className="microgpt-markdown microgpt-reference-pane">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {renderedFullGuideMarkdown}
                </ReactMarkdown>
              </div>
            )}

            {showFullSource && (
              <div className="microgpt-code-viewer microgpt-reference-pane">
                {sourceLines.map((line, idx) => {
                  const lineNo = idx + 1;
                  return (
                    <div key={lineNo} className="microgpt-code-row">
                      <span className="microgpt-line-number">{lineNo.toString().padStart(3, ' ')}</span>
                      <span className="microgpt-code-text">{line || ' '}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}

export default MicroGPTPage;
