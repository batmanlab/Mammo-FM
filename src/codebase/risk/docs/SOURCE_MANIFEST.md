# Risk component source manifest

This component was extracted from the `Mammo-FM-risk` research handoff for
integration into the main Mammo-FM repository.

## Included

- `mayo_mammofm_risk`: Stage 1 MIRAI probability distillation, Stage 2
  five-horizon risk training, prediction, losses, and the frozen Mammo-FM
  image-encoder wrapper.
- `mammo_clip_metadataset_mayo_KD.py`: the preserved four-view Mayo DICOM
  loader used by the risk runner.
- `mammo_sae`: SAE feature extraction, patient-level splitting, training,
  activation maps, mammogram overlays, risk intervention, selected/full-latent
  Shapley estimation, and text sentence attribution.
- Portable shell entry points, SAE configurations, source licenses, data
  contracts, checkpoint provenance, and synthetic tests.

