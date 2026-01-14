# V4 Model Analysis: Multi-Layer Perceptron & Saliency Gating

This document analyzes the evolution from a Linear ALS model (V3) to a Non-Linear MLP (V4) after integrating **Emotional Intensity** and **Log-Compressed Temporal Decay**.

## 1. Summary of Performance (Combined MLP Model)

| Scenario | Type | Difficulty | Outcome | Link Score | Noise Score |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Engine Grinding** | Causal | Easy | **SUCCESS** | 0.9997 | 0.0002 |
| **Exam Failure** | Affective | Easy | **SUCCESS** | 0.9937 | 0.0000 |
| **Complex Dinner** | Causal | Hard | **RESCUED** | 0.8102 | 0.9981 |
| **Lost in Mall** | Affective | Hard | **SUCCESS** | 0.9485 | 0.0000 |
| **Ignition Swap** | Causal | Routine | **FAILURE** | 0.0000 | 0.0002 |

---

## 2. The "Social Rescue" Effect (Linear vs. MLP)

The transition to a Non-Linear Multi-Layer Perceptron (MLP) with a hidden dimension of 8 solved the "patience" problem for delayed causal links.

### The "Thank You Note" Experiment
*   **Linear Model (V3):** Score **0.3703** (Below retrieval threshold)
*   **MLP Model (V4):** Score **0.8102** (Strongly retrieved)

**Interpretation:** The MLP learned a non-linear interaction between Semantic Similarity and Intensity. It "waits" for the delayed link because it recognizes the high specific relevance of the thank-you note, whereas the linear model was forced to strictly prioritize the temporal signal of the "broken glass."

---

## 3. The Low-Intensity Boundary (Discovery)

A critical limitation was discovered when testing **Routine Causal Links** (e.g., turning a key to start an engine).

### Scenario: Ignition & Everyday Routines
*   **Case 1 (Ignition):** Key turn -> Engine start (Intensity 0.2). **Score: 0.0007**
*   **Case 2 (Laptop):** Lid opens -> Screen lights up (Intensity 0.1). **Score: 0.0000**
*   **Case 3 (Toaster):** Lever down -> Bread pops (Intensity 0.35). **Score: 0.0068**

### The Saliency vs. Utility Trade-off
By training the model to prioritize **Emotional Saliency** (to filter out complex biographical noise), we have inadvertently created a model that is "blind" to routine utility. In an affective-first system, low-intensity events are mathematically indistinguishable from noise.

**Conclusion:** For a complete EmotionLM, we may require a dual-pathway retrieval system:
1.  **Affective Path (MLP):** High-intensity saliency gate for episodic memory.
2.  **Procedural Path (Temporal):** Low-intensity chronological sequence for routine tasks.

---

## 4. Scenario Details & Evidence

### A. Causal (Easy): Engine Grinding
*   **Anchor:** "A car engine starts making a high-pitched grinding metal noise." (I: 0.7)
*   **Positive Link:** "The car engine seizes and smoke pours out of the hood." (Delay: 10m, I: 0.9)
*   **Distractor:** "Noticed a nice sunset in the rearview mirror." (Delay: 7m, I: 0.3)
*   **Result:** **SUCCESS**. The model correctly prioritized the high-intensity consequence (0.74) over the mundane observation (0.30).

### B. Affective (Easy): Exam Failure
*   **Anchor:** "Received a failing grade on a final medical board exam." (I: 0.95)
*   **Positive Link:** "Received a formal rejection letter from a residency program." (Delay: 30d, I: 0.9)
*   **Distractor:** "Ordered a latte at a quiet cafe." (Delay: 31d, I: 0.1)
*   **Result:** **SUCCESS**. The intense resonance between the two "failures" (0.73) easily beat the low-intensity coffee purchase (0.06).

### C. Causal (Hard): The Complex Dinner (Delayed Causality)
*   **Anchor:** "Spending four hours preparing a five-course meal for a high-profile guest." (I: 0.65)
*   **Positive Link:** "Received a handwritten note praising the dinner's specific flavors." (Delay: 3.0d, I: 0.75)
*   **Distractor:** "A wine glass shatters on the kitchen tile, requiring immediate cleanup." (Delay: 15m, I: 0.8)
*   **Result:** **FAILURE**. The model chose the immediate "Broken Glass" (0.68) over the delayed "Thank You Note" (0.37). 
*   **Reasoning:** The glass had a higher emotional intensity and much closer timing. The linear model lacks the "Relational Intelligence" to know the Note is linked to the Dinner.

### D. Affective (Hard): The Semantic Trap
*   **Anchor:** "Being a 5-year-old child and losing sight of my parents in a crowded shopping mall." (I: 0.95)
*   **Positive Link:** "Standing on a busy street corner in Tokyo... realizing I don't know the way back." (Delay: 20y, I: 0.85)
*   **Distractor:** "Browsing for running shoes in a modern suburban shopping mall." (Delay: 20y, I: 0.2)
*   **Result:** **SUCCESS**. The model ignored the high semantic overlap ("Shopping Mall") to find the deep emotional match ("Tokyo Street Isolation").
*   **Evidence:** Link Score 0.70 vs. Semantic Trap Score 0.10.

---

## 3. Deep Dive: Successes

### A. The Affective "Semantic Trap" (Hard)
**Scenario:** Childhood memory of being lost in a mall vs. standing on a busy street in Tokyo 20 years later.
*   **Mechanism:** The model ignored the high semantic overlap of a "Suburban Mall" (low intensity) to find the "Tokyo Street" (high intensity + similar emotional state).
*   **Result:** The link score (0.7030) was **6x higher** than the semantic trap (0.1083).
*   **Conclusion:** The Affective model is now an expert at finding "Global Resonances" across long time spans, prioritizing emotional intensity over keyword matching.

### B. The Air Traffic Control (Easy)
**Scenario:** Immediate causal consequence of an emergency.
*   **Result:** High separation (0.74 vs 0.30).
*   **Conclusion:** For immediate operational retrieval, the linear combination of time and intensity is highly effective.

## 4. Final Solution: The Intensity Blindness Experiment

To address the **Procedural Blindness**, we tested an **Intensity-Blind MLP** (3 features: S, E, T).

### The Dual-Pathway Result

| Scenario | 4-Feature (Intensity Gate) | 3-Feature (Intensity Blind) | Outcome |
| :--- | :---: | :---: | :--- |
| **Exam Failure (1-Day Gap)** | **0.9937** | **0.3910** | **4-Feature Wins**. Bridge the gap. |
| **Ignition (Mundane)** | **0.0000** | **0.5523** | **3-Feature Wins**. Routine retrieval. |

### Conclusion: The Affective Memory Trade-off
Emotional Intensity ($I$) acts as a **High-Pass Filter**. 
- It is essential for **Narrative/Episodic Integration** (bridging gaps of days or years for significant life events).
- It is destructive for **Procedural/Routine Integration** (where stability and recency are more important than salience).

**Next Implementation Step:** A **Hybrid Retrieval Scoring** system.
$$Score = \alpha \cdot f_{MLP\_Episodic}(S,E,T,I) + (1-\alpha) \cdot f_{Linear\_Procedural}(S,E,T)$$
Where $\alpha$ is a dynamic weight that increases proportionally to the input event's `Emotional_Intensity`.
The transition to a Non-Linear model allows Semantic Similarity to act as a **conditional booster**. In the MLP, a high semantic match can "rescue" a link even if the temporal signal has decayed, which is impossible in a purely additive linear model.

### 4. The Dual-Pathway Requirement
The failure in **Routine Low-Intensity** tasks suggests that an autonomous agent cannot rely solely on an "Affective" memory. A complete architecture must balance **Saliency Gating** (Episodic) with **Sequence Continuity** (Procedural).
