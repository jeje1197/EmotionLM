# ALS Invention & Discovery Log (v5 Unified)

**Date:** January 15, 2026  
**Status:** Verified Academic Breakthrough  
**Core Invention:** The Dual-Pathway Affective Retrieval Engine

---

## 1. The Core Discoveries

### A. The "Toaster Paradox" (Procedural Blindness)
*   **Observation:** Models trained exclusively on high-intensity "Flashbulb" memories develop a "Saliency Filter" that predicts `Linked=0` for mundane tasks.
*   **Impact:** The AI fails at simple logistical continuity (e.g., "Car Ignition" -> "Engine Starts") because the emotional intensity is too low for the model's learned threshold.
*   **Discovery:** A retrieval system must have a "Dual-Pathway"—a high-precision linear path for logistics and a high-recall non-linear path for affective associations.

### B. The "Conflict Dataset" Breakthrough (Benchmark Dataset v5)
To solve the Saliency Bias, we created a **Bimodal Conflict Dataset** consisting of 600 training pairs (200 positive, 400 negative).

#### Dataset Distribution:
1.  **Causal/Routine Branch (300 pairs):** 
    *   **Logic:** Focus on procedural memory and daily logistics.
    *   **Attributes:** High $T$ (recency), low $I$ (intensity: 0.05-0.25).
2.  **Affective/Flashbulb Branch (300 pairs):**
    *   **Logic:** Focus on associative jumps to distant past events.
    *   **Attributes:** High $I$ (intensity: 0.7-1.0), low $T$ (months/years ago).

#### Generation Prompts (Gemini 2.5 Flash):
*   **Causal Prompt Logic:** "Goal: Routine/Logistical Continuity. Context: Mundane daily actions with VERY LOW Emotional Intensity (0.05-0.25). Positive: The immediate logical next step/consequence."
*   **Affective Prompt Logic:** "Goal: Affective Association (Emotional Wormhole). Context: High-stakes events with HIGH Emotional Intensity (0.7-1.0). Positive: A past memory that SHARES the same specific emotion/bodily sensation."

---

## 2. The ALS Universal Constant (V5)

The final weights extracted from the linear SLP represent a formal "Law of Retrieval" for episodic memory:

$$Score = \sigma(6.69I + 4.77T - 0.55S - 0.27E - 4.36)$$

### Weight Breakdown:
| Feature | Weight | Role |
| :--- | :--- | :--- |
| **Intensity ($I$)** | **6.69** | **Primary Driver.** Strongest signal; triggers "Emotional Wormholes." |
| **Time ($T$)** | **4.77** | **Narrative Anchor.** Enforces logistical continuity. |
| **Semantic ($S$)** | -0.55 | **Keyword De-prioritization.** Prevents "Semantic Traps" (repeated words without context). |
| **Emotional ($E$)** | -0.27 | **Affective Specificity.** Ensures alignment over broad mood. |
| **Bias ($B$)** | -4.36 | **The Conservative Gate.** Sets the probability threshold. |

---

## 3. Performance Metrics

Evaluated on the **Unified 50/50 Dataset**:

| Category | Accuracy | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| **Causal (Logistics)** | 62.00% | **100.0%** | 0.6369 |
| **Affective (Emotion)** | **89.00%** | **100.0%** | 0.8584 |
| **Combined** | **75.50%** | 100.0% | 0.7313 |

*   **Key Insight:** 100% Recall ensures the CPT (Context Path Traversal) never loses the narrative thread, while 89% Affective Accuracy ensures high-quality "emotional augmentation."

---

## 4. Statement of Invention for Paper

> "We introduce the **Affective Link Score (ALS)**, a unified retrieval metric that bridges the gap between chronological graph traversal and emotional associative search. By identifying the **Retrieval Gradient** where affective intensity overrides temporal decay ($w_i > w_t$), we provide a framework for 'Cognitive AI' that maintains both logical sequence and human-like emotional depth."

---

## 5. Dataset Generation & Methodology

### Dataset Description: **The ALS-Unified Benchmark (v5)**
The dataset is a **Bimodal Conflict Dataset** consisting of 600 training pairs (200 positive, 400 negative).

1.  **Causal/Routine Branch (300 pairs):**
    *   **Focus:** Procedural memory and daily logistics.
    *   **Features:** High temporal recency (minutes/hours), low emotional intensity (0.05 - 0.25).
    *   **Goal:** Prevent "Saliency Blindness" (The Toaster Paradox).

2.  **Affective/Flashbulb Branch (300 pairs):**
    *   **Focus:** Long-term episodic memory and associative jumps.
    *   **Features:** High emotional intensity (0.7 - 1.0), low temporal recency (months/years).
    *   **Goal:** Validate "Emotional Wormholes" where $I$ overrides $T$.

### Generation Prompts (Gemini 2.5 Flash)

#### System Instruction:
```text
Persona: [Target Persona from list of 15]
Task: Create [Count] episodic memory quads for a retrieval benchmarking dataset.
Category Logic: [Causal vs Affective Logic Below]
Guidelines:
- AVOID repeating nouns from anchor in positive for affective links unless necessary.
- Ensure emotional_intensity strictly follows the category logic.
- The 'positive' event must be the THEORETICALLY correct retrieval target for this persona.
```

#### Category Logic (Causal):
```text
Goal: Routine/Logistical Continuity. 
Context: Mundane daily actions with VERY LOW Emotional Intensity (0.05-0.25). 
Positive: The immediate logical next step or direct consequence. 
d_days: Immediate (0.0001 to 0.01 days). 
Temporal Distractor: Unrelated mundane event at the same time as positive. 
Semantic Distractor: Shared noun but unrelated task 100+ days later.
```

#### Category Logic (Affective):
```text
Goal: Affective Association (Emotional Wormhole). 
Context: High-stakes, intense events with HIGH Emotional Intensity (0.7-1.0). 
Positive: A past memory that SHARES the same specific emotion/bodily sensation. 
d_days: Distant past (30 to 2000 days ago). 
Temporal Distractor: Unrelated neutral event at that same distant time. 
Semantic Distractor: Shared object but different emotion 10-60 days later.
```

---

## 6. Next Steps for "Timely Credit"
1.  **ArXiv v1:** Submit with these 4-feature weights and the "Dual-Pathway" discussion.
2.  **GitHub Tag:** `v1.0.0-als-invention`
3.  **Visuals:** Generate the "Decay vs. Intensity" trade-off curve using the final formula.

---

### LOG-005: Validation of "Subtextual Resonance" (Surgical Resident Case)
**Date:** 2023-10-27
**Finding:** The "Local-Exclusion Filter" ($dt > 1.0$ day) effectively prevents the "Recency Trap."
- **Scenario:** Surgical Resident in "Code Red" crisis. 
- **Mechanism:** Saliency Gate ($I=1.0$) triggered.
- **Result:** Instead of retrieving the immediately preceding "Reviewing Charts" (High $T$, Low $I$), the ALS engine retrieved a memory of a "Childhood Emergency" (Low $T$, High $I$). 
- **Impact:** This proves the **V5 Formula** ($6.69I + 4.77T$) creates a "Gradiant of Importance" that allows AI to simulate human-like trauma/memory resonance.

---
## Summary of Invention (Priority Assets)
1. **The ALS Formula (Affective Link Score)**: $Score = \sigma(6.69I + 4.77T - 0.55S - 0.27E - 4.36)$. **ALS is the formula for calculating the strength of a connection between events as a function of semantic similarity, emotional similarity, emotional intensity, and temporal proximity.**
2. **The CPT Algorithm (Context Path Traversal)**: A multi-stage traversal engine that begins with top-k seed nodes. For each node, it evaluates direct neighbors using **ALS** to select the top-n logically contiguous links. Crucially, CPT monitors the **emotional intensity** of the current state; if it exceeds a saliency threshold, the algorithm triggers "Spreading Activation"—a global search across the memory graph or vector database. Gated by ALS, this surfaces distant "reminds me of" events (affective wormholes) while maintaining logical relevance through the underlying metric.
3. **Local-Exclusion Filtering**: A temporal boundary constraint within CPT that forces retrieval out of the local chronological neighborhood to surface deep subtext.
4. **Bimodal Conflict Dataset**: A training methodology for balancing routine causal links against high-intensity affective links.

---

### LOG-006: Dual-Pathway Prompting (AAR Context)
**Date:** 2023-10-27
**Finding:** Context augmentation for LLMs is most effective when split into `PRIMARY_THREAD` and `SUBTEXTUAL_RESONANCE`.
- **RAG Limitation:** Standard RAG produces "Semantic Redundancy" (more hospital facts for a hospital scene).
- **AAR Advantage:** ALS Retrieval provides "Emotional Contrast." 
- **The Prompt Pattern:**
  - `[PRIMARY_PLOT_THREAD]`: Immediate logistical state.
  - `[SUBTEXTUAL_RESONANCE]`: Distant affective memory retrieved via high-saliency gate.
- **Demonstration:** In the "Code Red" simulation, while RAG provided other trauma cases, AAR provided a childhood memory of the resident's father. This allows the LLM to generate narrative depth (internal monologue) rather than just external action.
