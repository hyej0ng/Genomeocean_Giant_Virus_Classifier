# GenomeOcean Sub Classifier

Main 모델이 `NCLDV/Mirus`로 분류한 FASTA contig를 다시 다음 두 그룹으로
분류하는 독립 배포 패키지입니다.

- `NCLDV`
- `Mirus`

Sub 저장소만 단독으로 사용할 수도 있지만, 입력이 이미 NCLDV/Mirus 후보라는
전제를 가집니다.

## 설치

```bash
python -m pip install -e ".[dev]"
```

## 5-fold ensemble 사용법

Main과 같은 방식으로 여러 모델의 확률을 평균합니다. Hugging Face 저장소
하나에 self-contained `fold1`~`fold5` 폴더를 올린 경우:

```bash
genomeocean-sub predict \
  --input examples/example_ncldv_mirus.fna \
  --output-dir outputs/example \
  --model-id your-org/genomeocean-sub-100m-5kb \
  --subfolder fold1 \
  --subfolder fold2 \
  --subfolder fold3 \
  --subfolder fold4 \
  --subfolder fold5 \
  --device auto
```

fold checkpoint를 별도 경로로 관리하면 `--model-id`를 fold 수만큼 반복합니다.
한 개만 주면 단일 모델로 동작합니다.

### 기존 학습 checkpoint를 업로드용으로 준비하기

학습 checkpoint의 custom code 원격 참조를 제거하려면 다음을 실행합니다.

```bash
python scripts/prepare_hf_folds.py \
  --output-dir /path/to/genomeocean-sub-100m-5kb \
  --checkpoint /path/to/sub/fold1/checkpoint-44240 \
  --checkpoint /path/to/sub/fold2/checkpoint-48620 \
  --checkpoint /path/to/sub/fold3/checkpoint-46530 \
  --checkpoint /path/to/sub/fold4/checkpoint-44195 \
  --checkpoint /path/to/sub/fold5/checkpoint-46130
```

중간에 export가 중단되어 같은 output 폴더를 다시 사용할 때는 명령 끝에
`--force`를 추가합니다. 원본 checkpoint는 변경하지 않습니다.

생성된 output 디렉터리를 Hugging Face에 업로드합니다. Sub의 `fold6`은
교차검증 fold가 아니라 전체 train 데이터 재학습용이므로 이 5-fold ensemble에는
포함하지 않습니다.

로컬 모델 export도 사용할 수 있습니다.

```bash
genomeocean-sub predict \
  --input candidates.fna \
  --output-dir outputs/sub \
  --model-id /absolute/path/to/exported-sub-model
```

## 전처리

Main과 동일하게 A/C/G/T/N 이외의 문자를 제거하고 `N`을 제거한 뒤, 정제된
서열에서 완전한 5,000bp chunk만 만듭니다. 5kb보다 짧은 contig는
`skipped_records.tsv`에 기록됩니다.

Main과 Sub의 전처리가 달라지면 통합 pipeline 결과가 잘못될 수 있으므로 두
저장소의 `test_preprocessing.py`를 함께 실행해야 합니다.

## 결과 파일

| 파일 | 설명 |
|---|---|
| `chunk_predictions.tsv` | fold 평균 NCLDV/Mirus 확률 |
| `contig_predictions.tsv` | chunk 다수결 contig 결과 |
| `file_predictions.tsv` | contig 다수결 FASTA 파일 결과 |
| `skipped_records.tsv` | 예측하지 못한 짧은/빈 contig |
| `run_metadata.json` | 모델 목록, fold 수, 입력, 실행 설정 |

## 테스트

```bash
python -m unittest discover -s tests -v
```

`pytest`를 설치했다면 `pytest`로 실행해도 됩니다.

기본 테스트는 실제 모델을 다운로드하지 않고 전처리와 집계 규칙을 확인합니다.

## 라이선스

공개 배포 전에 코드 라이선스와 기반 모델의 라이선스/NOTICE를 추가해야 합니다.
