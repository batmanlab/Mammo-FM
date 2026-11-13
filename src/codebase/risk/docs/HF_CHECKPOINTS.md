# Mammo-FM BU risk heads

The public
[`risk_checkpoints/BU_S1_KD_S2`](https://huggingface.co/batmanLab/Mammo-FM/tree/main/risk_checkpoints/BU_S1_KD_S2)
folder now contains exactly two newly trained BU risk heads:

- **S1→S2 (default):** Stage 1 knowledge distillation from MIRAI probabilities,
  followed by Stage 2 training on BU five-horizon outcomes.
- **S2-only:** the matched Stage 2 comparison initialized from the same seeded
  random head without Stage 1.

The five historical fold files and their old manifest were removed. The folder
name is retained for URL compatibility even though it also contains the S2-only
comparison.

## Downloads and provenance

The release is pinned to Hugging Face commit
[`7605810fab99c90eb9c9ff23206c237bafa3f500`](https://huggingface.co/batmanLab/Mammo-FM/commit/7605810fab99c90eb9c9ff23206c237bafa3f500).

| Protocol | Inference head | Best epoch | Bytes | Release SHA-256 | Source training-checkpoint SHA-256 |
| --- | --- | ---: | ---: | --- | --- |
| S1→S2 (default) | [Download](https://huggingface.co/batmanLab/Mammo-FM/resolve/7605810fab99c90eb9c9ff23206c237bafa3f500/risk_checkpoints/BU_S1_KD_S2/Mammo-FM_BU_S1_KD_S2_head.pt) | 6 | 23,381,538 | `6992aa888affb61b8cd897b1ee9e17fae8ef54bf0cb41f38d8935e63675c8273` | `2c8b5f7260e6e85e45470f87ee5f99ba8fc2e4a0c9e93c64d2b7e6d4407c7f8d` |
| S2-only | [Download](https://huggingface.co/batmanLab/Mammo-FM/resolve/7605810fab99c90eb9c9ff23206c237bafa3f500/risk_checkpoints/BU_S1_KD_S2/Mammo-FM_BU_S2_only_head.pt) | 7 | 23,381,474 | `cf3e602acf5b0269ca03c7f851cad5d9b933d4a11d17b84a7b334f431a93fa9c` | `204fc27f020a10f70b66994105fc789ca6f82d39d436e20b4f2d4d4768bdb3e4` |

