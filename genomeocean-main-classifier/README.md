# GenomeOcean Main Classifier

사용자가 제공한 FASTA를 다음 세 그룹으로 분류하는 배포용 패키지입니다.

- `Cellular`
- `NCLDV/Mirus`
- `Other Viruses`

학습용 fold CSV와 정답 label은 필요하지 않습니다. 모델 가중치는 GitHub에
포함하지 않고 Hugging Face 모델 저장소 또는 로컬 export 디렉터리에서
불러옵니다.

## 현재 상태

코드와 테스트는 준비되어 있지만 기본 공개 모델 ID는 아직 지정되지 않았습니다.
실행할 때 `--model-id`에 Hugging Face 모델 ID나 로컬 모델 경로를 전달해야
합니다.

여러 fold를 지정하면 같은 chunk를 각 모델에 통과시킨 뒤 클래스 확률을 평균하는
soft-voting ensemble을 사용합니다. 모델은 GPU 메모리를 아끼기 위해 한 번에
하나씩 로드하고 해제합니다.

## 설치

Python 3.11 환경에서 저장소 루트로 이동한 뒤 설치합니다.

```bash
python -m pip install -e ".[dev]"
```

`-e`는 개발 모드입니다. `src/` 아래 Python 파일을 수정하면 다시 설치하지
않아도 변경 내용이 반영됩니다.

## 사용법

```bash
genomeocean-main predict \
  --input examples/example.fna \
  --output-dir outputs/example \
  --model-id your-org/genomeocean-main-100m-5kb \
  --subfolder fold1 \
  --subfolder fold2 \
  --subfolder fold3 \
  --subfolder fold4 \
  --subfolder fold5 \
  --device auto
```

위 예시는 Hugging Face 저장소 하나 안에 `fold1`~`fold5`가 있고 각 폴더가
모델 파일과 tokenizer 파일을 모두 포함한다고 가정합니다. 각 fold를 별도
저장소나 로컬 checkpoint로 관리한다면 `--model-id`를 반복합니다.

```bash
genomeocean-main predict \
  --input sample.fna \
  --output-dir outputs/sample \
  --model-id /models/main-100m-5kb/fold1 \
  --model-id /models/main-100m-5kb/fold2 \
  --model-id /models/main-100m-5kb/fold3 \
  --model-id /models/main-100m-5kb/fold4 \
  --model-id /models/main-100m-5kb/fold5 \
  --local-files-only
```

`--model-id`를 한 번만 지정하면 단일 모델 추론으로 동작합니다.

### 기존 학습 checkpoint를 업로드용으로 준비하기

현재 학습 checkpoint에는 `modeling_mistral.py`는 있지만 `config.json`의
custom code 경로가 원본 DOEJGI 저장소를 가리키도록 되어 있습니다.
아래 스크립트는 optimizer 파일을 제외한 추론 파일만 `fold1`~`fold5`로
복사하고 custom code 경로를 로컬 파일로 고칩니다.

```bash
python scripts/prepare_hf_folds.py \
  --output-dir /path/to/genomeocean-main-100m-5kb \
  --checkpoint /path/to/main/fold1/checkpoint-416820 \
  --checkpoint /path/to/main/fold2/checkpoint-407235 \
  --checkpoint /path/to/main/fold3/checkpoint-427060 \
  --checkpoint /path/to/main/fold4/checkpoint-420540 \
  --checkpoint /path/to/main/fold5/checkpoint-425970
```

생성된 output 디렉터리 전체를 Hugging Face 모델 저장소에 업로드합니다.
각 fold 안에 `config.json`, `configuration_mistral.py`,
`modeling_mistral.py`, tokenizer, weight 파일이 있어야 합니다.

`--input`은 단일 `.fna`, `.fa`, `.fasta`, gzip 파일 또는 이러한 파일이 들어
있는 디렉터리일 수 있습니다.

아직 Hugging Face에 업로드하지 않은 로컬 checkpoint도 사용할 수 있습니다.

```bash
genomeocean-main predict \
  --input sample.fna \
  --output-dir outputs/sample \
  --model-id /absolute/path/to/exported-main-model
```

## 전처리

기존 학습 저장소의 `prep_go_dataset.py`와 같은 규칙을 사용합니다.

1. A/C/G/T/N 이외의 문자를 제거합니다.
2. `N`을 제거합니다.
3. 정제된 서열을 5,000bp, stride 5,000으로 나눕니다.
4. 5,000bp보다 짧은 contig는 예측하지 않고 `skipped_records.tsv`에 기록합니다.
5. 마지막 5,000bp 미만 tail은 예측에서 제외하고 길이를 결과에 기록합니다.

`chunk_start`와 `chunk_end`는 원본이 아닌 **정제된 서열 기준 좌표**입니다.

## 결과 파일

| 파일 | 설명 |
|---|---|
| `chunk_predictions.tsv` | 5kb chunk별 평균 확률과 ensemble 예측 |
| `contig_predictions.tsv` | chunk 다수결로 결정한 contig 결과 |
| `file_predictions.tsv` | contig 다수결로 결정한 FASTA 파일 결과 |
| `skipped_records.tsv` | 짧거나 정제 후 빈 contig |
| `ncldv_mirus_candidates.fna` | Sub 모델로 보낼 후보 |
| `ncldv_mirus_candidate_manifest.tsv` | 후보와 원본 contig 매핑 |
| `run_metadata.json` | 모델 목록, fold 수, 입력, 설정, 결과 개수 |

`chunk_predictions.tsv`의 `ensemble_size`는 사용한 fold 수, `ensemble_votes`는
그 결과를 선택한 fold 수, `ensemble_agreement`는 fold 동의 비율입니다.
`confidence_std`가 클수록 fold별 예측이 서로 다르다는 뜻입니다.

동률일 때는 기존 코드의 `idxmax` 동작과 동일하게 더 작은 label ID를
선택합니다.

## 테스트

모델 다운로드 없이 FASTA와 전처리, 집계 기능을 검사합니다.

```bash
python -m unittest discover -s tests -v
```

`pytest`를 설치했다면 `pytest`로 실행해도 됩니다.

실제 모델을 사용한 전체 테스트는 모델 export 또는 Hugging Face 업로드 후
위의 `genomeocean-main predict` 명령으로 수행합니다.

## 라이선스

공개 배포 전에 이 패키지의 코드 라이선스와 기반
`DOEJGI/GenomeOcean-100M-v1.2`의 라이선스/NOTICE를 확인하여 추가해야 합니다.
