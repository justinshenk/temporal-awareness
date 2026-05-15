# RQ47 Temporal Awareness Final Presentation Notes

Generated deck: `presentations/rq47_temporal_awareness_final.pptx`
Slide count: 22
Embedded image count: 12

## Speaker Notes

### Slide 1: Title
- Set the story: temporal horizon starts as a decodable representation, then becomes a causal and online monitoring target.

### Slide 2: The Whole Story
- This gives the audience a mental map before figures arrive. Each section answers one stronger question than the last.

### Slide 3: Experimental Map
- Point out that the deck uses the same artifact chain as the codebase: scripts produce results, notebooks summarize, deck tells the story.

### Slide 4: Dataset And Probe Setup
- Explain the key intuition: implicit training plus explicit validation is a semantic generalization test.

### Slide 5: Probe Methods
- Make LR/DMM agreement intuitive: a strong learned classifier and a simple mean direction often point to the same underlying separation.

### Slide 6: Finding 1: The Signal Generalizes
- Tell the audience how to read cells, then why cross-dataset transfer is the first big result.

### Slide 7: Finding 1 Detail: Method Comparison
- This slide translates the heatmap into a model-by-model comparison.

### Slide 8: Finding 2: The Signal Has A Depth Profile
- This replaces the previously cluttered layer-dynamics slide with one large readable figure and explanations below.

### Slide 9: Finding 2 Detail: Semantic Layer Coverage
- Explain coverage as robustness of where the signal can be read.

### Slide 10: Finding 3: Explicit And Implicit Curves Track Layer Behavior
- Use Qwen LR as a readable example of the earlier notebooks' per-case diagnostics.

### Slide 11: Finding 4: Cross-Validation Supports The Probe Signal
- This slide adds confidence: layer selection is not based on a single noisy split.

### Slide 12: Bridge: From Reading A Signal To Testing A Mechanism
- Clarify why causal experiments are necessary after strong probe validation.

### Slide 13: Causal Finding: Steering Moves The Logits
- Explain steering as a dose-response causal test, not merely another classifier metric.

### Slide 14: Causal Finding: Activation Patching Recovers Behavior
- Use recovery as the intuitive metric: clean internal state repairs corrupted output tendency.

### Slide 15: Causal Finding: Attribution Patching Is A Fast Cross-Check
- Frame attribution patching as a triangulation tool, not as stronger than activation patching.

### Slide 16: Causal Finding: Ablation Shows Dependence
- Mention the denominator caveat verbally; ablation is useful but can have outliers.

### Slide 17: Causal Synthesis
- This slide makes the causal story coherent and conservative.

### Slide 18: Online Oversight Setup
- Describe this as an unfolding-process experiment: probe scores over generation time.

### Slide 19: Online Finding: Pre-Event Drift Appears
- Keep the distinction clear: promising precursor signal, but keyword detector means no final safety claim.

### Slide 20: Online Finding Detail: Top Pre-Event Drifts
- Use examples to make the online monitor tangible.

### Slide 21: Limitations
- This slide keeps the story honest and helps the technical audience trust the positive results.

### Slide 22: Conclusion And Next Work
- End with the crisp claim: useful mechanistic oversight scaffold, with clear next experiments.

## Artifact Sources

- Probe validation figures: `results/figures/probe_validation_multimethod/`
- All-method validation figures: `results/figures/probe_validation_all_methods/`
- RQ47 intervention and oversight figures: `results/figures/rq47/`
- RQ47 summary tables: `results/tables/rq47/`

## Verification Checklist

- Generated PPTX is non-empty.
- Slide count matches the expanded story structure.
- Embedded image count is at least 12.
- Notes markdown exists and records slide-level speaker guidance.