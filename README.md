<div align="center">

## 🚘 Counterfactual Stability Analysis of Classical Lane Perception

</div>

> A classical lane perception pipeline (Canny → ROI mask → 
probabilistic Hough → slope-based grouping → least-squares line fit → vanishing point) 
tested under semantically plausible perturbations, using *temporal stability of the 
vanishing point* as a candidate uncertainty signal.

**Core finding:** a synthetic cast shadow leaves per-frame VP recovery at 77/77 (so 
per-frame confidence metrics see nothing wrong), yet roughly doubles frame-to-frame 
VP jitter — i.e. temporal incoherence reveals degradation that single-frame metrics miss.

### Contents
- `CS131_Project_Proposal.pdf` — original project proposal
- `CS131_Project_Milestone.pdf` — milestone report (progress, results, updated plan)
- `milestone/` — milestone code, figures, and outputs

Tested on KITTI raw sequence `2011_09_26_drive_0002` (77 frames, city driving). 

---
Lea M. Hadzic  
CS 131 – *Computer Vision: Foundations and Applications*
