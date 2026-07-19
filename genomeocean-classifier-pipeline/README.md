# GenomeOcean Classifier Pipeline

독립적인 Main과 Sub 패키지를 연결하여 사용자 FASTA contig를 최종 네 그룹으로
분류합니다.

- `Cellular`
- `NCLDV`
- `Mirus`
- `Other Viruses`

## 처리 흐름

```text
사용자 FASTA
    |
    v
Main: Cellular / NCLDV-Mirus / Other Viruses
    |
    +-- Cellular --------------------> Cellular
    +-- Other Viruses ---------------> Other Viruses
    +-- NCLDV/Mirus --> Sub
                         +------------> NCLDV
                         +------------> Mirus
```

Pipeline은 모델 예측 코드를 복사하지 않고 설치된
`genomeocean-main-classifier`와 `genomeocean-sub-classifier` 패키지를
호출합니다.

## 로컬 개발 설치

세 저장소가 같은 `deploy/` 디렉터리에 있을 때 다음 순서로 설치합니다.

```bash
cd /path/to/01_gv_genomeocean/deploy

python -m pip install -e ./genomeocean-main-classifier
python -m pip install -e ./genomeocean-sub-classifier
python -m pip install -e "./genomeocean-classifier-pipeline[dev]"
```

Main/Sub가 아직 PyPI에 등록되지 않았기 때문에 반드시 먼저 설치해야 합니다.

## 모델 설정

`configs/models.yaml`의 `your-org`를 실제 Hugging Face 조직 또는 사용자명으로
교체합니다. 각 fold 디렉터리는 tokenizer와 model 파일을 포함한 독립적인
Hugging Face `from_pretrained` 디렉터리여야 합니다.

```yaml
main:
  model_id: your-org/genomeocean-main-100m-5kb
  revision: main
  folds: [fold1, fold2, fold3, fold4, fold5]

sub:
  model_id: your-org/genomeocean-sub-100m-5kb
  revision: main
  folds: [fold1, fold2, fold3, fold4, fold5]

inference:
  chunk_size: 5000
  stride: 5000
```

Hugging Face 업로드 전에는 명령행으로 로컬 모델 경로를 지정할 수 있습니다.

## 사용법

설정 파일의 5개 fold ensemble을 사용할 때:

```bash
genomeocean-classify \
  --input examples/example.fna \
  --output-dir outputs/example
```

로컬 모델을 직접 지정할 때:

```bash
genomeocean-classify \
  --input sample.fna \
  --output-dir outputs/sample \
  --main-model-id /absolute/path/to/exported-main-model \
  --sub-model-id /absolute/path/to/exported-sub-model \
  --local-files-only
```

앞의 export 스크립트로 준비한 로컬 fold 디렉터리 5개를 직접 지정하려면
각 옵션을 반복합니다. 원본 학습 checkpoint 경로를 바로 지정하지 마십시오.

```bash
genomeocean-classify \
  --input sample.fna \
  --output-dir outputs/sample \
  --main-model-id /models/main-100m-5kb/fold1 \
  --main-model-id /models/main-100m-5kb/fold2 \
  --main-model-id /models/main-100m-5kb/fold3 \
  --main-model-id /models/main-100m-5kb/fold4 \
  --main-model-id /models/main-100m-5kb/fold5 \
  --sub-model-id /models/sub-100m-5kb/fold1 \
  --sub-model-id /models/sub-100m-5kb/fold2 \
  --sub-model-id /models/sub-100m-5kb/fold3 \
  --sub-model-id /models/sub-100m-5kb/fold4 \
  --sub-model-id /models/sub-100m-5kb/fold5 \
  --local-files-only
```

Main은 모든 contig에 5개 fold를 사용하고, Sub는 Main ensemble이
`NCLDV/Mirus`로 판단한 후보에만 5개 fold를 사용합니다. 모델을 동시에
메모리에 유지하지 않고 fold별로 순차 실행하므로 GPU peak memory는 단일 모델에
가깝지만 추론 시간은 대략 5배입니다.

## 결과

```text
outputs/sample/
├── main/
│   ├── chunk_predictions.tsv
│   ├── contig_predictions.tsv
│   └── ncldv_mirus_candidates.fna
├── sub/
│   ├── chunk_predictions.tsv
│   └── contig_predictions.tsv
├── final_contig_predictions.tsv
├── final_file_predictions.tsv
└── run_metadata.json
```

`final_contig_predictions.tsv`에는 Main 결과, 필요한 경우 Sub 결과, 최종
4-class 결과가 한 행에 함께 저장됩니다. `main_ensemble_size`와
`sub_ensemble_size`로 실제 사용한 fold 수를 확인할 수 있습니다.

NCLDV/Mirus 분기의 최종 confidence는 현재
`main_confidence × sub_confidence`로 기록합니다. 이는 편리한 계층형 점수이며
확률 보정(calibration)이 완료된 임상적/통계적 확률을 의미하지 않습니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

`pytest`를 설치했다면 `pytest`로 실행해도 됩니다.

테스트에서는 가짜 predictor를 사용하여 Main의 NCLDV/Mirus 후보만 Sub로
전달되는지 검사합니다. 실제 모델을 다운로드하지 않습니다.

## 배포 전 남은 작업

1. Main/Sub 최종 모델 export
2. Hugging Face 모델 저장소 2개 생성
3. `models.yaml`의 실제 모델 ID와 revision 고정
4. 라이선스와 NOTICE 추가
5. 새로운 환경에서 clone/install/end-to-end 검사
