# SmartPolice Literature And Benchmark Map

## Material Passport

- Origin skills: `research-project-lead`, `academic-research-suite/deep-research`
- Created: 2026-06-13
- Verification status: REPO_REFERENCES_MAPPED
- Source: `docs/references.md` and current project benchmark framing

## Use Of Literature

This project should use literature and public benchmarks to shape evaluation protocols, not to borrow claims or scores.

## Benchmark Families

| Family | What it contributes | How SmartPolice uses it | Boundary |
| --- | --- | --- | --- |
| GenImage | Cross-generator and degraded-image framing | Generator labels plus JPEG/crop/watermark/screenshot-resave robustness matrix | Do not claim full GenImage leaderboard performance unless locally reproduced. |
| AIGIBench | External blind-test and source-aware evaluation | `dataset_source` holdout and label-covered diagnostics | Current strict source-holdout remains weak. |
| SIDA/SID-Set | Social-media distribution shift | Future social-platform domain import and licensing checks | Not yet an active mixed training source. |
| RRDataset / ITW-SM | In-the-wild propagation, repeated upload, recapture/retake | Future robustness conditions beyond screenshot-resave | Current project only partially proxies these conditions. |
| UniversalFakeDetect / Synthbuster-like baselines | Generic fake-image detector comparison | Useful external baseline candidates | Need clean, reproducible comparison protocol before claiming superiority. |

## Policy And Police-Relevant Context

Use these as problem-background and compliance context:

- 生成式人工智能服务管理暂行办法
- 互联网信息服务深度合成管理规定
- 人工智能生成合成内容标识办法
- public rumor cases involving AI-generated fire/disaster/public-safety images

Writing rule:

> Policy sources justify why AIGC rumor governance matters; they do not prove model forensic validity.

## Current Research Gap

Existing public AIGC detection and attribution work often evaluates clean or benchmarked images. The SmartPolice project gap is the police workflow after social-platform disturbance:

- original metadata may be stripped
- screenshots and reuploads change low-level traces
- false positives have operational consequences
- model output must become an auditable clue, not a final conclusion

## Search Terms For Next Deep-Research Pass

- AI-generated image detection robustness
- AI image forensics social media compression
- generated image attribution source holdout
- synthetic image detection real-world benchmark
- C2PA provenance AI-generated images
- deep synthesis governance China
- police misinformation image verification
- AIGC rumor governance public safety

## Next Literature Tasks

1. Verify the latest versions and claims of GenImage, AIGIBench, SIDA/SID-Set, RRDataset, and ITW-SM before citing.
2. Add 3-5 papers on social-media compression/recapture robustness.
3. Add 2-3 sources on provenance standards such as C2PA/watermarking.
4. Add 2-3 public-safety rumor cases from official or high-trust sources.
5. Build a short "what we borrow / what we do not claim" table for the final report.
